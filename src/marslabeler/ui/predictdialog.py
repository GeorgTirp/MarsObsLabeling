"""Prediction dialog: progress bar while a model runs inference over an observation."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)

from marslabeler.classes import ClassScheme
from marslabeler.inference.engine import (
    compute_global_quantization,
    run_block_inference_with_pixel_maps,
    run_block_scores,
)
from marslabeler.io.raster import RasterSource
from marslabeler.model.grid import BlockInfo


def _remap_pixel_maps(
    pixel_maps: dict[str, np.ndarray], model_index_to_id: dict[int, int]
) -> dict[str, np.ndarray]:
    """Vectorized per-array remap of model channel indices -> classes.yaml ids
    (the scalar equivalent of `model_index_to_id.get(model_idx)` used for the
    majority-voted label, applied to every pixel via a lookup table instead of
    a python-level loop -- these arrays are block_size x block_size each)."""
    if not pixel_maps:
        return {}
    max_seen = max(int(crop.max()) for crop in pixel_maps.values() if crop.size)
    max_known = max(model_index_to_id.keys(), default=-1)
    lut = np.full(max(max_seen, max_known) + 1, -1, dtype=np.int16)
    for model_idx, class_id in model_index_to_id.items():
        lut[model_idx] = class_id

    remapped: dict[str, np.ndarray] = {}
    for block_id, crop in pixel_maps.items():
        mapped_crop = lut[crop]
        unmapped = mapped_crop < 0
        if unmapped.any():
            bad_idx = int(crop[unmapped].flat[0])
            raise ValueError(
                f"Model predicted channel index {bad_idx}, which has no "
                f"matching class in classes.yaml. Add a class with "
                f"model_index: {bad_idx} (or id: {bad_idx})."
            )
        remapped[block_id] = mapped_crop.astype(np.uint8)
    return remapped


class PredictWorker(QThread):
    """Worker thread: load the checkpoint, then run block-level inference.

    Computes two things from the same loaded model: the class prediction (needs
    class-mapping via ClassScheme) and a softmax-confidence score (needs no
    calibration -- available for any model, trained or not), so the class Summary
    window always has *some* per-block confidence data once inference has run.
    """

    progress = Signal(int, int)  # done, total
    status = Signal(str)
    # block_id -> class id (classes.yaml id space), block_id -> mean softmax
    # confidence [0,1], block_id -> (h_px, w_px) uint8 per-pixel class-id crop
    # (classes.yaml id space) for the pixel-wise view -- see MainWindow's
    # prediction_render_mode.
    finished_ok = Signal(dict, dict, dict)
    failed = Signal(str)

    def __init__(
        self,
        raster: RasterSource,
        blocks: list[BlockInfo],
        block_size: int,
        checkpoint_path: Path,
        classes_scheme: ClassScheme,
        inference_config: dict,
    ):
        super().__init__()
        self.raster = raster
        self.blocks = blocks
        self.block_size = block_size
        self.checkpoint_path = checkpoint_path
        self.classes_scheme = classes_scheme
        self.inference_config = inference_config
        self._cancelled = False
        # class_id -> component -> [LocalGalleryItem]; populated during run().
        # Read by the dialog after finished_ok rather than sent through the
        # signal, so the existing 3-dict finished_ok contract is untouched.
        self.local_npca: dict = {}

    def cancel(self) -> None:
        self._cancelled = True

    def _build_local_npca(self, bundle, embedding_sink, raw_predictions: dict) -> dict:
        """Rank this observation's blocks on the checkpoint's stored PCA bases.

        Returns {} whenever it cannot be done -- a v1 `.npca.pt` with no bases,
        no artifact at all, or a cancelled run -- since this is an optional
        enrichment of the Summary window, never a reason to fail inference.
        """
        try:
            from marslabeler.inference.local_npca import (
                build_local_gallery,
                classifier_weight_vectors,
                stack_block_embeddings,
            )
            from marslabeler.inference.modelio import sidecar_path
            from vision_backend.pc_align.neural_pca import load_gallery_bases

            bases = load_gallery_bases(sidecar_path(self.checkpoint_path, "npca.pt"))
            if not bases:
                return {}

            embeddings, aligned = stack_block_embeddings(embedding_sink, self.blocks)
            if not len(aligned):
                return {}

            # Restrict each class's exemplars to blocks the model actually called
            # that class; otherwise a component surfaces whichever block projects
            # highest regardless of whether the class is present at all.
            eligible: dict[int, set[str]] = {}
            for block_id, model_index in raw_predictions.items():
                eligible.setdefault(int(model_index), set()).add(block_id)

            weights = classifier_weight_vectors(bundle.model, list(bases.keys()))
            return build_local_gallery(
                embeddings, aligned, bases, weights, eligible_block_ids=eligible
            )
        except Exception as exc:  # optional feature -- never break inference
            self.status.emit(f"Local Neural-PCA unavailable: {exc}")
            return {}

    def run(self) -> None:
        try:
            if not self.blocks:
                self.finished_ok.emit({}, {}, {})
                return

            self.status.emit(f"Loading model {self.checkpoint_path.name}...")
            from marslabeler.inference.modelio import (
                build_inference_plan,
                load_model_bundle,
                make_confidence_score_fn,
                make_predict_fn_with_embeddings,
            )

            bundle = load_model_bundle(
                self.checkpoint_path,
                device=self.inference_config.get("device", "auto"),
                ai4exomars_path=self.inference_config.get("ai4exomars_path"),
            )
            plan = build_inference_plan(
                bundle,
                block_size=self.block_size,
                batch_size=self.inference_config.get("batch_size", 4),
                context_multiplier=self.inference_config.get("context_multiplier", 4),
                gsd_ratio=self.inference_config.get("gsd_ratio", 1.0),
            )
            model_index_to_id = self.classes_scheme.model_index_to_id()

            quantization_bounds = None
            if self.raster.dtype != "uint8":
                self.status.emit(
                    f"Non-uint8 imagery ({self.raster.dtype}) -- computing a global "
                    "brightness stretch to 8-bit (approximation; see docs)..."
                )
                quantization_bounds = compute_global_quantization(
                    self.raster, self.raster.width, self.raster.height
                )

            self.status.emit(
                f"Running inference ({bundle.model_kind}, {bundle.num_classes} classes, "
                f"device={bundle.device})..."
            )

            def progress_cb(done: int, total: int) -> None:
                self.progress.emit(done, total)
                self.status.emit(f"Predicting block {done}/{total}...")

            # Capture each block's pooled feature phi(x) off the SAME forward pass
            # that produces the predictions (measured cost: within noise). These
            # feed the per-observation Neural-PCA gallery, so its thumbnails point
            # at blocks of THIS image instead of the training corpus.
            predict_fn, embedding_sink = make_predict_fn_with_embeddings(
                bundle, quantization_bounds
            )
            raw, raw_pixel_maps = run_block_inference_with_pixel_maps(
                self.raster,
                self.blocks,
                plan,
                predict_fn,
                progress_cb=progress_cb,
                should_cancel=lambda: self._cancelled,
            )
            self.local_npca = self._build_local_npca(bundle, embedding_sink, raw)

            if self._cancelled:
                self.status.emit("Cancelled")
                self.finished_ok.emit({}, {}, {})
                return

            mapped: dict[str, int] = {}
            for block_id, model_idx in raw.items():
                class_id = model_index_to_id.get(model_idx)
                if class_id is None:
                    raise ValueError(
                        f"Model predicted channel index {model_idx}, which has no "
                        f"matching class in classes.yaml. Add a class with "
                        f"model_index: {model_idx} (or id: {model_idx})."
                    )
                mapped[block_id] = class_id
            pixel_maps = _remap_pixel_maps(raw_pixel_maps, model_index_to_id)

            def confidence_progress_cb(done: int, total: int) -> None:
                self.progress.emit(done, total)
                self.status.emit(f"Scoring confidence: block {done}/{total}...")

            confidence = run_block_scores(
                self.raster,
                self.blocks,
                plan,
                make_confidence_score_fn(bundle, quantization_bounds),
                progress_cb=confidence_progress_cb,
                should_cancel=lambda: self._cancelled,
            )

            self.status.emit(f"Predicted {len(mapped)} blocks")
            self.finished_ok.emit(mapped, confidence, pixel_maps)

        except Exception as e:
            self.failed.emit(str(e))


class PredictDialog(QDialog):
    """Modal progress dialog wrapping PredictWorker; mirrors PreprocessDialog."""

    def __init__(
        self,
        raster: RasterSource,
        blocks: list[BlockInfo],
        block_size: int,
        checkpoint_path: Path,
        classes_scheme: ClassScheme,
        inference_config: dict,
    ):
        super().__init__()
        self.setWindowTitle("Running Model Inference")
        self.setModal(True)
        self.setMinimumWidth(420)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)

        self.predictions: dict[str, int] = {}
        self.confidence: dict[str, float] = {}
        self.pixel_predictions: dict[str, np.ndarray] = {}
        self.error: str | None = None

        layout = QVBoxLayout()

        title = QLabel(f"Predicting with {checkpoint_path.name}")
        title.setStyleSheet("font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        self.status_label = QLabel("Starting...")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.button_box.rejected.connect(self._on_cancel)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

        self.worker = PredictWorker(
            raster, blocks, block_size, checkpoint_path, classes_scheme, inference_config
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished_ok.connect(self._on_finished_ok)
        self.worker.failed.connect(self._on_failed)

    def start(self) -> None:
        self.worker.start()

    def _on_progress(self, done: int, total: int) -> None:
        pct = int((done / total) * 100) if total else 100
        self.progress_bar.setValue(pct)

    def _on_cancel(self) -> None:
        self.worker.cancel()
        self.button_box.setEnabled(False)
        self.status_label.setText("Cancelling...")

    def _on_finished_ok(self, predictions: dict, confidence: dict, pixel_predictions: dict) -> None:
        self.predictions = predictions
        self.confidence = confidence
        self.pixel_predictions = pixel_predictions
        # Built during the same pass; {} when the checkpoint has no PCA bases.
        self.local_npca = getattr(self.worker, "local_npca", {}) or {}
        if predictions:
            self.accept()
        else:
            self.reject()

    def _on_failed(self, message: str) -> None:
        self.error = message
        self.status_label.setText(f"Error: {message}")
        self.reject()

    def get_predictions(self) -> dict[str, int]:
        return self.predictions

    def get_confidence(self) -> dict[str, float]:
        return self.confidence

    def get_local_npca(self) -> dict:
        """class_id -> component -> ranked blocks of THIS observation ({} if unavailable)."""
        return getattr(self, "local_npca", {})

    def get_pixel_predictions(self) -> dict[str, np.ndarray]:
        """block_id -> (h_px, w_px) uint8 per-pixel predicted class-id crop
        (classes.yaml id space), for the pixel-wise prediction view."""
        return self.pixel_predictions
