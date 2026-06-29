"""Preprocessing dialog: progress bar while computing invalid masks and skip decisions."""

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QDialogButtonBox,
)

from marslabeler.io.raster import RasterSource
from marslabeler.model.grid import Grid
from marslabeler.io.preprocess import compute_invalid_mask


class PreprocessWorker(QThread):
    """Worker thread for preprocessing (invalid mask computation)."""

    progress = Signal(int)
    status = Signal(str)
    finished = Signal(dict)

    def __init__(self, raster: RasterSource, grid: Grid, config: dict):
        super().__init__()
        self.raster = raster
        self.grid = grid
        self.config = config
        self.skip_decisions = {}

    def run(self):
        """Compute invalid masks and skip decisions for all blocks."""
        try:
            nodata_threshold = self.config.get("skip", {}).get("nodata_skip_threshold", 0.5)
            skip_low_var = self.config.get("skip", {}).get("skip_low_variance", False)
            variance_threshold = self.config.get("skip", {}).get("variance_skip_threshold", 0.0)

            total_blocks = self.grid.num_blocks()

            for idx, block in enumerate(self.grid.iter_blocks()):
                # Update progress
                progress = int((idx / total_blocks) * 100)
                self.progress.emit(progress)
                self.status.emit(f"Processing block {idx + 1}/{total_blocks}...")

                # Compute nodata fraction
                nodata_frac = self.raster.nodata_fraction(
                    block.x_px, block.y_px, block.w_px, block.h_px
                )

                # Decide if block should be skipped
                should_skip = nodata_frac > nodata_threshold

                # Check variance if enabled
                if not should_skip and skip_low_var and variance_threshold > 0:
                    variance = self.raster.variance(
                        block.x_px, block.y_px, block.w_px, block.h_px
                    )
                    should_skip = variance < variance_threshold

                self.skip_decisions[block.block_id] = {
                    "nodata_fraction": nodata_frac,
                    "should_skip": should_skip,
                }

            self.progress.emit(100)
            self.status.emit("Preprocessing complete!")
            self.finished.emit(self.skip_decisions)

        except Exception as e:
            self.status.emit(f"Error: {str(e)}")
            self.finished.emit({})


class PreprocessDialog(QDialog):
    """Dialog showing preprocessing progress before main window launches."""

    def __init__(self, raster: RasterSource, grid: Grid, config: dict):
        super().__init__()
        self.setWindowTitle("Processing Observation")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)

        self.skip_decisions = {}

        # Layout
        layout = QVBoxLayout()

        # Title
        title = QLabel("Preprocessing Invalid Pixels & Nodata Detection")
        title.setStyleSheet("font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # Status label
        self.status_label = QLabel("Starting preprocessing...")
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)

        # Dialog buttons (disabled during processing)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

        # Worker thread
        self.worker = PreprocessWorker(raster, grid, config)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self._on_preprocessing_finished)

    def start_preprocessing(self):
        """Start the preprocessing worker thread."""
        self.worker.start()

    def _on_preprocessing_finished(self, skip_decisions: dict):
        """Called when preprocessing completes."""
        self.skip_decisions = skip_decisions
        # Close dialog automatically on success
        if skip_decisions:
            self.accept()
        else:
            self.reject()

    def get_skip_decisions(self) -> dict:
        """Get the computed skip decisions."""
        return self.skip_decisions
