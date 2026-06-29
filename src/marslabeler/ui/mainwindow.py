"""Main window: ties together all UI components."""

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QStatusBar,
    QLabel,
    QFileDialog,
    QDialog,
    QApplication,
    QPushButton,
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
from marslabeler.ui.preprocessdialog import PreprocessDialog
from marslabeler.ui.helpdialog import HelpDialog
from marslabeler.ui.loadingoverlay import LoadingOverlay
from marslabeler.model.export import export_coarse_geotiff, export_class_metadata


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
        self.controller: Optional[KeyboardController] = None
        self.autosave_timer = None
        self.skip_decisions = {}
        self.help_shown_on_startup = False
        self.loading_overlay: Optional[LoadingOverlay] = None
        self.na_class_id: Optional[int] = None
        self.na_class_name: Optional[str] = None
        self.saved_complete_panels: set[int] = set()

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
        self.main_layout = main_layout

        # Left: History panel (placeholder until session loads)
        self.history_panel = QLabel("(No session)")
        self.history_panel.setMaximumWidth(200)
        main_layout.addWidget(self.history_panel)

        # Center: Panel canvas
        self.canvas = PanelCanvas()
        self.canvas.on_block_clicked = self._on_block_clicked
        main_layout.addWidget(self.canvas, 1)

        # Right: Preview (top) and legend (below) vertical stack
        right_layout = QVBoxLayout()
        self.right_layout = right_layout

        # Side preview (top)
        self.preview = SidePreview()
        right_layout.addWidget(self.preview)

        # Legend panel (below preview, placeholder until session loads)
        self.legend_panel = QLabel("(No session)")
        right_layout.addWidget(self.legend_panel)

        right_layout.addStretch()

        # Action buttons (disabled until a session loads)
        self.next_panel_button = QPushButton("Next Panel ▶  (fills rest as NA)")
        self.next_panel_button.setEnabled(False)
        self.next_panel_button.clicked.connect(self._go_to_next_panel)
        right_layout.addWidget(self.next_panel_button)

        self.export_button = QPushButton("Export Labels")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export)
        right_layout.addWidget(self.export_button)

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

        self.export_action = QAction("Export Labels...", self)
        self.export_action.triggered.connect(self._export)
        self.export_action.setEnabled(False)
        file_menu.addAction(self.export_action)

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
            self._resolve_na_class()

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
            self.controller.on_show_help = self._show_help
            self.controller.on_next_panel = self._go_to_next_panel

            # Setup autosave timer
            self._setup_autosave()

            # Show preprocessing dialog
            preprocess_dialog = PreprocessDialog(raster, grid, self.config.to_dict())
            preprocess_dialog.start_preprocessing()

            if preprocess_dialog.exec() != QDialog.DialogCode.Accepted:
                self.status_label.setText("Loading cancelled")
                raster.close()
                return

            # Store skip decisions for later use
            self.skip_decisions = preprocess_dialog.get_skip_decisions()

            # Update UI
            self._update_history_panel()
            self._update_legend_panel()

            # Create loading overlay to block interaction during panel load (after UI is set up)
            self.loading_overlay = LoadingOverlay(self)
            self.loading_overlay.setGeometry(self.rect())
            self.loading_overlay.set_status("Rendering first panel...")
            self.loading_overlay.set_progress(50)
            self.loading_overlay.raise_()
            self.loading_overlay.show()
            # Let the overlay paint before the (blocking) synchronous read
            QApplication.processEvents()

            self._load_current_panel()

            # Enable session-dependent actions
            self.next_panel_button.setEnabled(True)
            self.export_button.setEnabled(True)
            self.export_action.setEnabled(True)
            self.saved_complete_panels = set()

            self.status_label.setText(f"Loaded: {jp2_path.stem}")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def _update_history_panel(self):
        """Swap the history placeholder for the real panel (by reference)."""
        history = HistoryPanel(self.session.grid, self.session.labels)
        history.on_panel_selected = self._on_panel_selected
        history.setMaximumWidth(200)
        old_item = self.main_layout.replaceWidget(self.history_panel, history)
        if old_item is not None and old_item.widget() is not None:
            old_item.widget().deleteLater()
        self.history_panel = history

    def _update_legend_panel(self):
        """Swap the legend placeholder for the real legend (by reference)."""
        legend = LegendPanel(self.classes_scheme)
        old_item = self.right_layout.replaceWidget(self.legend_panel, legend)
        if old_item is not None and old_item.widget() is not None:
            old_item.widget().deleteLater()
        self.legend_panel = legend

    def _load_current_panel(self):
        """Load and display the panel the cursor is in (synchronous, decimated read)."""
        if not self.session:
            return

        # Keep the displayed panel in sync with the cursor
        self.current_panel_idx = self.session.current_block().panel_idx
        self.status_label.setText(f"Loading panel {self.current_panel_idx}...")

        # Synchronous decimated read — fast via GDAL overviews, no thread to crash
        grid = self.session.grid
        x, y, w, h = grid.get_panel_coords(self.current_panel_idx)
        panel_data = self.session.raster.read_window(x, y, w, h, 1600, 1600)
        self._render_panel(panel_data)

    def _render_panel(self, panel_data: np.ndarray):
        """Render the given panel image and its overlays."""
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

        # Hide loading overlay after first panel loads
        if self.loading_overlay:
            self.loading_overlay.hide()
            self.loading_overlay.deleteLater()
            self.loading_overlay = None

        # Defer help dialog to next event loop iteration so the canvas paints first
        if not self.help_shown_on_startup:
            self.help_shown_on_startup = True
            QTimer.singleShot(100, self._show_help)

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
        """Handle history panel click: move the cursor into that panel and show it.

        This is a review/jump action (e.g. going back to fix a panel) — it does
        NOT NA-fill; only the explicit Next Panel action does that.
        """
        if not self.session:
            return
        self.session.move_to_panel(panel_idx)  # cursor → first block of that panel
        self._load_current_panel()             # re-syncs current_panel_idx from cursor
        self._refresh_history()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle keyboard input."""
        # Ignore input while the loading overlay is up (except letting Esc through)
        if self.loading_overlay is not None:
            super().keyPressEvent(event)
            return
        if self.controller and self.controller.handle_key_press(event):
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:
        """Keep the loading overlay covering the whole window on resize."""
        super().resizeEvent(event)
        if self.loading_overlay is not None:
            self.loading_overlay.setGeometry(self.rect())

    def _on_labels_changed(self) -> None:
        """Callback: labels have changed, refresh UI."""
        prev_panel = self.current_panel_idx  # panel the just-labeled block was in
        self._load_current_panel()  # reloads + re-syncs current_panel_idx to the cursor
        self._refresh_history()
        # If that panel is now fully done (e.g. last block labeled), autosave it
        self._maybe_save_completed_panel(prev_panel)

    def _on_panel_changed_kb(self) -> None:
        """Callback: cursor rolled into another panel (auto-advance / PageUp). Just redraw."""
        self._load_current_panel()
        self._refresh_history()

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

    def _show_help(self) -> None:
        """Show keyboard shortcuts help dialog."""
        if not self.classes_scheme:
            return
        dialog = HelpDialog(self.classes_scheme)
        dialog.exec()

    # ------------------------------------------------------------------ #
    # Panel completion: NA-fill, autosave, history marking, export
    # ------------------------------------------------------------------ #

    def _resolve_na_class(self) -> None:
        """Find the 'Not Available' (NA) user class used to fill remaining blocks."""
        self.na_class_id = None
        self.na_class_name = None
        if not self.classes_scheme:
            return
        for cid, cobj in self.classes_scheme.classes.items():
            if cobj.name.strip().lower() in ("not available", "na", "n/a"):
                self.na_class_id = cid
                self.na_class_name = cobj.name
                return

    def _panel_complete(self, panel_idx: int) -> bool:
        """True when no block in the panel is still unlabeled."""
        for block in self.session.grid.get_panel_blocks(panel_idx):
            if self.session.labels.get_record(block.block_id).status == "unlabeled":
                return False
        return True

    def _finalize_panel(self, panel_idx: int) -> None:
        """Fill the panel's unlabeled blocks with NA, then autosave the session."""
        if not self.session:
            return

        # Fill remaining unlabeled blocks as NA (single undo snapshot)
        if self.na_class_id is not None:
            unlabeled_ids = [
                block.block_id
                for block in self.session.grid.get_panel_blocks(panel_idx)
                if self.session.labels.get_record(block.block_id).status == "unlabeled"
            ]
            if unlabeled_ids:
                self.session.labels.bulk_assign(
                    unlabeled_ids, self.na_class_id, self.na_class_name
                )

        self.saved_complete_panels.add(panel_idx)
        self._autosave_session(note=f"panel {panel_idx} done")

    def _maybe_save_completed_panel(self, panel: int) -> None:
        """Autosave when the given panel becomes complete (e.g. last block labeled)."""
        if not self.session:
            return
        if self._panel_complete(panel):
            if panel not in self.saved_complete_panels:
                self.saved_complete_panels.add(panel)
                self._autosave_session(note=f"panel {panel} complete")
        else:
            # Panel reopened/edited below complete — allow it to save again later
            self.saved_complete_panels.discard(panel)

    def _autosave_session(self, note: str = "") -> None:
        """Save the session parquet + cursor JSON."""
        if not self.session:
            return
        try:
            labels_dir = Path(self.config.paths.labels_dir)
            self.session.save_session(labels_dir)
            if self.controller:
                self.controller.reset_autosave()
            msg = "Saved" if not note else f"Saved ({note})"
            self.status_label.setText(msg)
        except Exception as e:
            self.status_label.setText(f"Save error: {str(e)}")

    def _go_to_next_panel(self) -> None:
        """Button/handler: finalize current panel (NA-fill + save), then advance."""
        if not self.session:
            return

        leaving = self.current_panel_idx
        self._finalize_panel(leaving)

        # Advance to the next panel (stays put if already on the last one)
        self.session.move_to_panel(leaving + 1)
        self.current_panel_idx = self.session.current_block().panel_idx
        self._load_current_panel()
        self._refresh_history()

    def _refresh_history(self) -> None:
        """Refresh the history panel's progress bars and done markers in place."""
        if isinstance(self.history_panel, HistoryPanel):
            self.history_panel.refresh()

    def _export(self) -> None:
        """Export labels to exports/<obs_id>/ (coarse GeoTIFF + parquet + classes)."""
        if not self.session:
            return
        try:
            obs_id = self.session.grid.obs_id
            out_dir = Path("exports") / obs_id
            out_dir.mkdir(parents=True, exist_ok=True)

            self.session.labels.save_parquet(out_dir / f"{obs_id}_labels.parquet")
            export_coarse_geotiff(
                self.session.labels, self.session.grid, out_dir / f"{obs_id}_coarse.tif"
            )
            export_class_metadata(
                self.session.labels, self.classes_scheme, out_dir / "classes.json"
            )
            self.status_label.setText(f"Exported to {out_dir}/")
        except Exception as e:
            self.status_label.setText(f"Export error: {str(e)}")
