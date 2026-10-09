"""Cancellable crop export using a separate raster reader in a worker thread."""

import traceback
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QProgressBar, QVBoxLayout

from marslabeler.model.cropexport import ExportCancelled, export_probe_set
from marslabeler.model.export import export_coarse_geotiff


class ExportWorker(QThread):
    progress = Signal(int, int)

    def __init__(self, source, snapshot, classes_path, output, grid, predictions_mode):
        super().__init__()
        self.source = source
        self.snapshot = snapshot
        self.classes_path = classes_path
        self.output = output
        self.grid = grid
        self.predictions_mode = predictions_mode
        self.count = 0
        self.error = None
        self.cancelled = False

    def run(self):
        try:
            def archives(folder):
                # The main window is modal throughout this worker. Export the
                # immutable label snapshot, never live session labels.
                from marslabeler.model.labelstore import LabelStore

                labels = LabelStore.load_parquet(self.snapshot, self.grid, strict=True)
                export_coarse_geotiff(labels, self.grid, folder / "labels_coarse.tif")

            self.count = export_probe_set(
                self.source, self.snapshot, self.classes_path, self.output,
                progress_cb=self.progress.emit,
                should_cancel=self.isInterruptionRequested,
                prepare_archives=archives,
                label_origin="prediction_review" if self.predictions_mode else "labeling_session",
            )
        except ExportCancelled:
            self.cancelled = True
        except Exception as exc:
            self.error = str(exc)
            traceback.print_exc()


class ExportDialog(QDialog):
    def __init__(self, source, snapshot, classes_path, output, grid, predictions_mode, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export crops and labels")
        self.setMinimumWidth(420)
        self.setModal(True)
        layout = QVBoxLayout(self)
        self.status = QLabel("Checking labels and preparing image crops…")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        layout.addWidget(self.progress)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.worker = ExportWorker(source, snapshot, classes_path, output, grid, predictions_mode)
        self.worker.progress.connect(self._progress)
        self.worker.finished.connect(self._finished)

    def _progress(self, done, total):
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        if not self.worker.isInterruptionRequested():
            self.status.setText(
                "Verifying dataset files and label mappings…" if done == total
                else f"Exporting crop {done:,} of {total:,}"
            )

    def _finished(self):
        if self.worker.error or self.worker.cancelled:
            super().reject()
        else:
            self.accept()

    def reject(self):
        # Escape, window close, and Cancel all wait for the worker to release
        # its reader and staging files before the dialog can be destroyed.
        if self.worker.isRunning():
            self.worker.requestInterruption()
            self.status.setText("Cancelling export…")
            self.buttons.setEnabled(False)
        else:
            super().reject()

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)
