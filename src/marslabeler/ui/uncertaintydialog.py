"""Uncertainty dialog: progress bar while computing the Mahalanobis (epistemic)
uncertainty heatmap for the Uncertainty Heatmap toggle.

Unlike class prediction / softmax confidence, this needs a fitted calibration
artifact (`MahalanobisStats`, produced offline by
`AI4ExoMars/vision_backend/uncertainty/fit_gaussians.py` from a trained checkpoint +
labeled crops) -- conventionally a `<checkpoint_stem>.uncertainty.pt` sidecar next to
the checkpoint. Until a trained model has been calibrated, that sidecar won't exist;
this dialog reports that clearly (via `failed`) rather than fabricating a heatmap.

A Stage-4 block classifier is scored per block from its phi instead
(`inference.blockanalysis`): from the prediction pass's phi when the caller has
it, otherwise after one feature pass over the blocks.
"""

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)

from marslabeler.inference.engine import compute_global_quantization, run_block_scores_with_pixel_maps
from marslabeler.inference.pixelmaps import PixelMapStore
from marslabeler.io.raster import RasterSource
from marslabeler.model.grid import BlockInfo


class UncertaintyWorker(QThread):
    """Worker thread: load the checkpoint + fitted stats, then score blocks."""

    progress = Signal(int, int)  # done, total
    status = Signal(str)
    finished_ok = Signal(dict)  # block_id -> uncertainty in [0, 1]
    failed = Signal(str)

    def __init__(
        self,
        raster: RasterSource,
        blocks: list[BlockInfo],
        block_size: int,
        checkpoint_path: Path,
        inference_config: dict,
    ):
        super().__init__()
        self.raster = raster
        self.blocks = blocks
        self.block_size = block_size
        self.checkpoint_path = checkpoint_path
        self.inference_config = inference_config
        self._cancelled = False
        self.pixel_scores = {}
        self.maximum = 1.0
        self.method = inference_config.get("method", "mahalanobis")
        # Stage-4 phi/logits: given by the caller when the prediction pass kept
        # them, otherwise collected here (and handed back) in one feature pass.
        self.block_features = inference_config.get("block_features")

    def cancel(self) -> None:
        self._cancelled = True

    def _load_model(self):
        """(bundle, plan, quantization bounds) for a pass over self.blocks."""
        from marslabeler.inference.modelio import build_inference_plan, load_model_bundle

        self.status.emit(f"Loading model {self.checkpoint_path.name}...")
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
        quantization_bounds = None
        if self.raster.dtype != "uint8":
            self.status.emit(
                f"Non-uint8 imagery ({self.raster.dtype}) -- computing a global "
                "brightness stretch to 8-bit (approximation; see docs)..."
            )
            quantization_bounds = compute_global_quantization(
                self.raster, self.raster.width, self.raster.height
            )
        return bundle, plan, quantization_bounds

    def _progress(self, verb: str):
        def progress_cb(done: int, total: int) -> None:
            self.progress.emit(done, total)
            self.status.emit(f"{verb} block {done}/{total}...")
        return progress_cb

    def _missing_calibration(self, path: Path) -> None:
        self.failed.emit(
            f"No {self.method} calibration found for this model at:\n{path}\n\n"
            "Use calibration fitted for this exact checkpoint. "
            "A training prior or calibration from an older model is not interchangeable."
        )

    def _collect_block_features(self):
        """One block-classifier pass keeping phi and logits; None if cancelled."""
        from marslabeler.inference.blockanalysis import BlockFeatures
        from marslabeler.inference.engine import run_block_inference_with_pixel_maps
        from marslabeler.inference.modelio import make_prediction_analysis_fn

        bundle, plan, quantization_bounds = self._load_model()
        if not bundle.block_classifier:
            raise ValueError("Block-level uncertainty needs a Stage-4 block classifier checkpoint.")
        predict_fn, phi_sink, _, _, logit_sink = make_prediction_analysis_fn(
            bundle, quantization_bounds, blocks=self.blocks, plan=plan,
        )
        run_block_inference_with_pixel_maps(
            self.raster, self.blocks, plan, predict_fn,
            progress_cb=self._progress("Extracting features of"),
            should_cancel=lambda: self._cancelled, keep_pixel_maps=False,
        )
        if self._cancelled:
            return None
        return BlockFeatures.from_batches(self.blocks, phi_sink, logit_sink, bundle.model)

    def _run_block_level(self, stats_path: Path) -> None:
        from marslabeler.inference.blockanalysis import (
            MAHALANOBIS_DISPLAY_MAXIMUM,
            block_laplace_scores,
            block_mahalanobis_scores,
        )

        if self.block_features is None:
            self.block_features = self._collect_block_features()
            if self.block_features is None:
                self.status.emit("Cancelled")
                self.finished_ok.emit({})
                return
        ai4exomars_path = self.inference_config.get("ai4exomars_path")
        if self.method == "laplace":
            self.status.emit(f"Sampling the Laplace posterior ({stats_path.name})...")
            scores = block_laplace_scores(
                self.block_features, stats_path, ai4exomars_path=ai4exomars_path,
                progress_cb=self._progress("Scoring"), should_cancel=lambda: self._cancelled,
            )
        else:
            scores = block_mahalanobis_scores(self.block_features, stats_path, ai4exomars_path=ai4exomars_path)
            self.maximum = MAHALANOBIS_DISPLAY_MAXIMUM
        if self._cancelled:
            self.status.emit("Cancelled")
            self.finished_ok.emit({})
            return
        # Blocks retired as nodata since the prediction pass get no score.
        wanted = {block.block_id for block in self.blocks}
        scores = {block_id: score for block_id, score in scores.items() if block_id in wanted}
        self.status.emit(f"Scored {len(scores)} blocks")
        self.finished_ok.emit(scores)

    def run(self) -> None:
        try:
            if not self.blocks:
                self.finished_ok.emit({})
                return

            from marslabeler.inference.modelio import (
                load_uncertainty_stats,
                make_uncertainty_score_fn,
                sidecar_path,
            )

            suffix = "laplace.pt" if self.method == "laplace" else "uncertainty.pt"
            stats_path = sidecar_path(self.checkpoint_path, suffix)
            self.status.emit(f"Loading uncertainty calibration ({stats_path.name})...")
            if not stats_path.exists():
                self._missing_calibration(stats_path)
                return
            if self.block_features is not None or self.inference_config.get("block_classifier"):
                self._run_block_level(stats_path)
                return

            stats = None
            if self.method == "mahalanobis":
                try:
                    stats = load_uncertainty_stats(
                        stats_path, ai4exomars_path=self.inference_config.get("ai4exomars_path")
                    )
                except FileNotFoundError:
                    self._missing_calibration(stats_path)
                    return

            bundle, plan, quantization_bounds = self._load_model()

            if self.method == "laplace":
                from marslabeler.inference.laplace import make_laplace_score_fn
                score_fn = make_laplace_score_fn(bundle, stats_path, quantization_bounds)
            else:
                score_fn = make_uncertainty_score_fn(bundle, stats, quantization_bounds, normalize=False)
                self.maximum = float(stats.reference_max_distance or 1.0)

            self.status.emit("Computing uncertainty heatmap...")

            scores, pixels = run_block_scores_with_pixel_maps(
                self.raster,
                self.blocks,
                plan,
                score_fn,
                progress_cb=self._progress("Scoring"),
                should_cancel=lambda: self._cancelled,
                map_store=PixelMapStore(),
            )

            if self._cancelled:
                self.status.emit("Cancelled")
                self.finished_ok.emit({})
                return

            self.status.emit(f"Scored {len(scores)} blocks")
            self.pixel_scores = pixels
            self.finished_ok.emit(scores)
        except Exception as e:
            self.failed.emit(str(e))


class UncertaintyDialog(QDialog):
    """Modal progress dialog wrapping UncertaintyWorker; mirrors PredictDialog."""

    def __init__(
        self,
        raster: RasterSource,
        blocks: list[BlockInfo],
        block_size: int,
        checkpoint_path: Path,
        inference_config: dict,
    ):
        super().__init__()
        name = "Laplace uncertainty" if inference_config.get("method") == "laplace" else "Mahalanobis distance"
        self.setWindowTitle(f"Computing {name}")
        self.setModal(True)
        self.setMinimumWidth(420)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)

        self.scores: dict[str, float] = {}
        self.error: str | None = None

        layout = QVBoxLayout()

        scope = "per block" if inference_config.get("block_classifier") else "pixel map"
        title = QLabel(f"{name} · {scope} ({checkpoint_path.name})")
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

        self.worker = UncertaintyWorker(raster, blocks, block_size, checkpoint_path, inference_config)
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

    def _on_finished_ok(self, scores: dict) -> None:
        self.scores = scores
        if scores:
            self.accept()
        else:
            self.reject()

    def _on_failed(self, message: str) -> None:
        self.error = message
        self.status_label.setText(message)
        self.reject()

    def get_scores(self) -> dict[str, float]:
        return self.scores

    def get_pixel_scores(self):
        return self.worker.pixel_scores

    def get_maximum(self):
        return self.worker.maximum

    def get_block_features(self):
        """Stage-4 phi/logits used for the scores (blockanalysis.BlockFeatures), else None."""
        return self.worker.block_features
