"""Main window: ties together all UI components."""

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QStatusBar,
    QLabel,
    QFileDialog,
    QMenu,
    QMenuBar,
    QProgressDialog,
)
from PySide6.QtGui import QAction, QKeyEvent

from marslabeler.io.raster import RasterSource
from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import LabelStore
from marslabeler.model.session import Session
from marslabeler.classes import load_classes
from marslabeler.config import load_config
from marslabeler.ui.panelcanvas import PanelCanvas
from marslabeler.ui.sidepreview import SidePreview
from marslabeler.ui.legendpanel import LegendPanel
from marslabeler.ui.historypanel import HistoryPanel
from marslabeler.ui.controller import KeyboardController


class PanelLoadWorker(QThread):
    """Worker thread for loading panel images (non-blocking)."""

    panel_loaded = Signal(np.ndarray, str)  # image, panel_id

    def __init__(self, raster: RasterSource, grid: Grid, panel_idx: int):
        super().__init__()
        self.raster = raster
        self.grid = grid
        self.panel_idx = panel_idx

    def run(self):
        """Load panel image."""
        x, y, w, h = self.grid.get_panel_coords(self.panel_idx)
        panel_data = self.raster.read_window(x, y, w, h, 1600, 1600)
        panel_id = f"Panel {self.panel_idx}"
        self.panel_loaded.emit(panel_data, panel_id)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, config_path: Path = None):
        super().__init__()
        self.setWindowTitle("Mars Obs Labeler")
        self.setGeometry(100, 100, 1920, 1080)

        # Load config
        if config_path is None:
            config_path = Path("configs/app.yaml")
        self.config = load_config(config_path)

        # State
        self.session: Optional[Session] = None
        self.classes_scheme = None
        self.current_panel_idx = 0
        self.panel_load_worker: Optional[PanelLoadWorker] = None
        self.controller: Optional[KeyboardController] = None
        self.autosave_timer = None

        # UI Components
        self._setup_ui()
        self._setup_menu()

    def _setup_ui(self):
        """Build UI layout."""
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout: history | canvas | legend+preview
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        # Left: History panel (placeholder until session loads)
        self.history_panel = QLabel("(No session)")
        self.history_panel.setMaximumWidth(200)
        main_layout.addWidget(self.history_panel)

        # Center: Panel canvas
        self.canvas = PanelCanvas()
        self.canvas.on_block_clicked = self._on_block_clicked
        main_layout.addWidget(self.canvas, 1)

        # Right: Legend and preview (vertical stack)
        right_layout = QVBoxLayout()

        # Legend panel (placeholder)
        self.legend_panel = QLabel("(No session)")
        right_layout.addWidget(self.legend_panel)

        # Side preview
        self.preview = SidePreview()
        right_layout.addWidget(self.preview)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(right_widget)

        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.status_label = QLabel("Ready")
        self.statusBar.addWidget(self.status_label)

    def _setup_menu(self):
        """Build menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open JP2...", self)
        open_action.triggered.connect(self._on_open_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _on_open_file(self):
        """File→Open JP2 dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open JP2 Observation",
            "",
            "JP2 Images (*.jp2);;All Files (*)",
        )

        if not path:
            return

        self._load_observation(Path(path))

    def _load_observation(self, jp2_path: Path):
        """Load a JP2 observation and create a session."""
        try:
            # Open raster
            raster = RasterSource(jp2_path)
            raster.open()

            # Create grid
            grid = Grid(
                img_width=raster.width,
                img_height=raster.height,
                panel_size=self.config.geometry.panel_size,
                block_size=self.config.geometry.block_size,
                obs_id=jp2_path.stem,
                transform=raster.transform,
                crs=raster.crs,
            )

            # Load classes
            self.classes_scheme = load_classes(self.config.paths.classes_file)

            # Create or load session
            labels_dir = Path(self.config.paths.labels_dir)
            self.session = Session.load_or_create(
                jp2_path,
                grid,
                self.config.to_dict(),
                labels_dir,
                labeler=self.config.labeler or "unknown",
            )

            # Create keyboard controller
            self.controller = KeyboardController(self.session, self.classes_scheme)
            self.controller.on_label_changed = self._on_labels_changed
            self.controller.on_panel_changed = self._on_panel_changed_kb
            self.controller.on_cursor_changed = self._on_cursor_changed

            # Setup autosave timer
            self._setup_autosave()

            # Update UI
            self._update_history_panel()
            self._update_legend_panel()
            self._load_current_panel()

            self.status_label.setText(f"Loaded: {jp2_path.stem}")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def _update_history_panel(self):
        """Replace history placeholder with actual panel."""
        # Remove old widget
        layout = self.centralWidget().layout()
        old_item = layout.itemAt(0)
        if old_item:
            old_item.widget().deleteLater()

        # Add new history panel
        history = HistoryPanel(self.session.grid, self.session.labels)
        history.on_panel_selected = self._on_panel_selected
        layout.insertWidget(0, history)
        self.history_panel = history

    def _update_legend_panel(self):
        """Replace legend placeholder with actual panel."""
        layout = self.centralWidget().layout().itemAt(2).widget().layout()
        old_item = layout.itemAt(0)
        if old_item:
            old_item.widget().deleteLater()

        # Add new legend panel
        legend = LegendPanel(self.classes_scheme)
        layout.insertWidget(0, legend)
        self.legend_panel = legend

    def _load_current_panel(self):
        """Load and display the current panel."""
        if not self.session:
            return

        self.status_label.setText(f"Loading panel {self.current_panel_idx}...")

        # Load in worker thread
        self.panel_load_worker = PanelLoadWorker(
            self.session.raster,
            self.session.grid,
            self.current_panel_idx,
        )
        self.panel_load_worker.panel_loaded.connect(self._on_panel_loaded)
        self.panel_load_worker.start()

    def _on_panel_loaded(self, panel_data: np.ndarray, panel_id: str):
        """Called when panel image is loaded."""
        if not self.session:
            return

        # Display panel image
        self.canvas.set_panel_image(panel_data, stretch_percentiles=(1, 99))

        # Set grid
        grid = self.session.grid
        self.canvas.set_grid(grid.blocks_per_panel_row, grid.blocks_per_panel_col)

        # Build label overlay (class_id → color)
        panel_blocks = grid.get_panel_blocks(self.current_panel_idx)
        block_data = np.full(
            (grid.blocks_per_panel_row, grid.blocks_per_panel_col),
            -3,  # unlabeled
            dtype=np.int16,
        )

        for block in panel_blocks:
            record = self.session.labels.get_record(block.block_id)
            output_row = block.block_row
            output_col = block.block_col
            block_data[output_row, output_col] = record.class_id

        class_colors = {
            cls_id: cls_obj.color for cls_id, cls_obj in self.classes_scheme.classes.items()
        }
        class_colors[self.classes_scheme.abstain.id] = self.classes_scheme.abstain.color
        class_colors[self.classes_scheme.nodata.id] = self.classes_scheme.nodata.color

        self.canvas.set_label_overlay(block_data, class_colors)

        # Set current block highlight
        current_block = self.session.current_block()
        self.canvas.set_current_block_highlight(current_block.block_row, current_block.block_col)

        # Load side preview
        block_data_native = self.session.raster.read_window(
            current_block.x_px,
            current_block.y_px,
            current_block.w_px,
            current_block.h_px,
            current_block.w_px,
            current_block.h_px,
        )
        self.preview.set_block_image(block_data_native, current_block.block_id)

        self.status_label.setText(f"Panel {self.current_panel_idx} loaded")

    def _on_block_clicked(self, block_row: int, block_col: int):
        """Handle panel canvas block click."""
        if not self.session:
            return

        # Navigate to clicked block
        grid = self.session.grid
        panel_blocks = grid.get_panel_blocks(self.current_panel_idx)
        local_idx = block_row * grid.blocks_per_panel_col + block_col
        if local_idx < len(panel_blocks):
            block_idx = self.current_panel_idx * grid.blocks_per_panel + local_idx
            self.session.move_to_block(block_idx)
            self._load_current_panel()

    def _on_panel_selected(self, panel_idx: int):
        """Handle history panel selection."""
        self.current_panel_idx = panel_idx
        self._load_current_panel()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle keyboard input."""
        if self.controller and self.controller.handle_key_press(event):
            return
        super().keyPressEvent(event)

    def _on_labels_changed(self) -> None:
        """Callback: labels have changed, refresh UI."""
        self._load_current_panel()  # Reload to show updated overlays

    def _on_panel_changed_kb(self) -> None:
        """Callback: panel changed via keyboard, load new panel."""
        # Update panel index based on current block
        self.current_panel_idx = self.session.current_block().panel_idx
        self._load_current_panel()

    def _on_cursor_changed(self) -> None:
        """Callback: cursor moved, update preview only."""
        if not self.session:
            return

        current_block = self.session.current_block()
        self.canvas.set_current_block_highlight(current_block.block_row, current_block.block_col)

        # Update preview
        block_data_native = self.session.raster.read_window(
            current_block.x_px,
            current_block.y_px,
            current_block.w_px,
            current_block.h_px,
            current_block.w_px,
            current_block.h_px,
        )
        self.preview.set_block_image(block_data_native, current_block.block_id)

    def _setup_autosave(self) -> None:
        """Setup autosave timer."""
        if self.autosave_timer:
            self.autosave_timer.stop()

        self.autosave_timer = QTimer()
        autosave_interval = self.config.autosave.every_seconds * 1000  # ms
        self.autosave_timer.setInterval(autosave_interval)
        self.autosave_timer.timeout.connect(self._maybe_autosave)
        self.autosave_timer.start()

    def _maybe_autosave(self) -> None:
        """Check if autosave should trigger."""
        if not self.session or not self.controller:
            return

        if self.controller.should_autosave():
            self._do_autosave()

    def _do_autosave(self) -> None:
        """Perform autosave."""
        if not self.session:
            return

        try:
            labels_dir = Path(self.config.paths.labels_dir)
            self.session.save_session(labels_dir)
            self.controller.reset_autosave()
            self.status_label.setText(f"Auto-saved (panel {self.current_panel_idx})")
        except Exception as e:
            self.status_label.setText(f"Autosave error: {str(e)}")
