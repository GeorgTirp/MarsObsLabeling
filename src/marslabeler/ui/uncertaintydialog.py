"""Uncertainty dialog: progress bar while computing the Mahalanobis (epistemic)
uncertainty heatmap for the Uncertainty Heatmap toggle.

Unlike class prediction / softmax confidence, this needs a fitted calibration
artifact (`MahalanobisStats`, produced offline by
`AI4ExoMars/vision_backend/uncertainty/fit_gaussians.py` from a trained checkpoint +
labeled crops) -- conventionally a `<checkpoint_stem>.uncertainty.pt` sidecar next to
the checkpoint. Until a trained model has been calibrated, that sidecar won't exist;
this dialog reports that clearly (via `failed`) rather than fabricating a heatmap.
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

from marslabeler.inference.engine import compute_global_quantization, run_block_scores
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

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            if not self.blocks:
                self.finished_ok.emit({})
                return

            from marslabeler.inference.modelio import (
                VisionBackendNotFound,
                build_inference_plan,
                load_model_bundle,
                load_uncertainty_stats,
                make_uncertainty_score_fn,
                sidecar_path,
            )

            stats_path = sidecar_path(self.checkpoint_path, "uncertainty.pt")
            self.status.emit(f"Loading uncertainty calibration ({stats_path.name})...")
            try:
                stats = load_uncertainty_stats(
                    stats_path, ai4exomars_path=self.inference_config.get("ai4exomars_path")
                )
            except FileNotFoundError:
                self.failed.emit(
                    f"No uncertainty calibration found for this model at:\n{stats_path}\n\n"
                    "Fit it once from a trained checkpoint with:\n"
                    "AI4ExoMars/vision_backend/uncertainty/fit_gaussians.py "
                    f"--checkpoint {self.checkpoint_path}"
                )
                return

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

            score_fn = make_uncertainty_score_fn(bundle, stats, quantization_bounds)

            self.status.emit("Computing uncertainty heatmap...")

            def progress_cb(done: int, total: int) -> None:
                self.progress.emit(done, total)
                self.status.emit(f"Scoring block {done}/{total}...")

            scores = run_block_scores(
                self.raster,
                self.blocks,
                plan,
                score_fn,
                progress_cb=progress_cb,
                should_cancel=lambda: self._cancelled,
            )

            if self._cancelled:
                self.status.emit("Cancelled")
                self.finished_ok.emit({})
                return

            self.status.emit(f"Scored {len(scores)} blocks")
            self.finished_ok.emit(scores)

        except VisionBackendNotFound as e:
            self.failed.emit(str(e))
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
        self.setWindowTitle("Computing Uncertainty Heatmap")
        self.setModal(True)
        self.setMinimumWidth(420)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)

        self.scores: dict[str, float] = {}
        self.error: str | None = None

        layout = QVBoxLayout()

        title = QLabel(f"Mahalanobis Uncertainty ({checkpoint_path.name})")
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
