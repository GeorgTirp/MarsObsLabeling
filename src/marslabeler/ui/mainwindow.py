"""Main window: ties together all UI components."""

import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QSplitter,
    QStatusBar,
    QLabel,
    QFileDialog,
    QDialog,
    QApplication,
    QPushButton,
    QMessageBox,
    QFrame,
    QSizePolicy,
    QScrollArea,
)
from PySide6.QtGui import QAction, QKeyEvent, QKeySequence

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
from marslabeler.ui.theme import apply_theme
from marslabeler.model.export import export_coarse_geotiff, export_class_metadata


class MainWindow(QMainWindow):
    """Main application window."""

    # Above this total block count, warn before loading (fine --resolution values
    # can produce millions of tiles: slow to build, heavy to render, tedious to label).
    TILE_COUNT_WARN_THRESHOLD = 1_000_000

    def __init__(
        self,
        config_path: Path = None,
        resolution_m: Optional[float] = None,
        ignore_cached_predictions: bool = False,
        match_training_gsd: bool | None = None,
        predictions_mode: bool = False,
    ):
        super().__init__()
        apply_theme()
        self.setWindowTitle("Mars Obs · Observation workspace")
        self.resize(1440, 900)

        # Load config
        if config_path is None:
            config_path = Path("configs/app.yaml")
        self.config = load_config(config_path)

        # Predictions mode (mars-inference): labels_dir points at a per-model predictions
        # cache instead of the configured labels folder, and a Save Predictions button
        # persists them explicitly (see load_for_inference()). Otherwise this is exactly
        # the labeling window -- predicted blocks are edited with the same hotkeys.
        self.predictions_mode = predictions_mode
        self.model_path: Optional[Path] = None
        self._model_sig: Optional[str] = None
        # Kept alive + non-modal (see _show_class_summary) so a Neural-PCA
        # thumbnail click can navigate this window while the summary stays
        # open and visible alongside it.
        self._summary_dialog = None

        # Analysis layers (mars-inference): which overlay the canvas is showing, and
        # the per-block data behind it. All empty/None until inference actually runs
        # (or, for npca_gallery, until a calibration artifact is found) -- see
        # load_for_inference(), _run_prediction(), _toggle_uncertainty_layer().
        self.display_layer = "classes"  # "classes" | "uncertainty"
        self.block_confidence: dict[str, float] = {}  # block_id -> mean softmax confidence
        self.block_uncertainty: dict[str, float] = {}  # block_id -> mean epistemic (Mahalanobis) uncertainty
        self.npca_gallery: Optional[dict] = None  # class model_index -> component -> ranked thumbnails
        # Same shape, but ranked over THIS observation's blocks (built during
        # the prediction pass); its items carry block ids, so clicking one
        # navigates the open image instead of the training corpus.
        self.local_npca_gallery: dict = {}

        # Which granularity the "classes" display_layer renders at: "pixelwise"
        # (each pixel colored by its own predicted class -- the model's raw
        # output, computed for free during inference and free to render) or
        # "blockwise" (one solid color per block -- LabelStore's own majority-
        # voted/edited class, today's original behavior). block_pixel_predictions
        # holds the per-block raw crops "pixelwise" stitches together; empty
        # until _run_prediction() runs (a cache-hit reload of previously-saved
        # predictions never repopulates it -- only the block-level class survives
        # a save, see _save_predictions -- so pixelwise falls back to blockwise
        # per-panel whenever no crop is cached for it).
        self.prediction_render_mode = "pixelwise"  # "pixelwise" | "blockwise"
        self.block_pixel_predictions: dict[str, np.ndarray] = {}

        # Optional classification tile resolution override (meters/tile-side), from
        # --resolution. None -> use config.geometry.block_size unchanged.
        self.resolution_m = resolution_m
        # --fresh: ignore any saved predictions/session for the observation, so
        # inference re-runs (restoring the pixel-wise view, per-block confidence
        # and the local Neural-PCA gallery, all of which are memory-only products
        # of the prediction pass) and the requested --resolution actually applies.
        self.ignore_cached_predictions = bool(ignore_cached_predictions)
        # None -> ask when a mismatch is found; True/False -> decided on the CLI.
        self.match_training_gsd = match_training_gsd
        # observation_gsd / training_gsd actually in force; 1.0 = native pixels.
        self.gsd_ratio = 1.0

        # Where labels are saved to / resumed from. Starts at the config default;
        # changeable via File -> Set Labels Folder... (applies to the next Open).
        self.labels_dir = Path(self.config.paths.labels_dir)

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
        self.empty_panels: set[int] = set()  # fully off-swath panels: hidden + skipped
        # Region labeling: held class key (drag-paint brush) + last used class
        self.held_class_id: Optional[int] = None
        self.last_class_id: Optional[int] = None
        # Active marquee selection (r0, c0, r1, c1) awaiting a class key to fill
        self.selection_rect: Optional[tuple[int, int, int, int]] = None
        # View mode: "panel" (single, labelable) | "multi" (NxN panels) | "overview" (whole image)
        self.view_mode = "panel"
        self.multi_span = 2  # panels per side in "multi" mode (2 or 4)
        self._multi_origin = (0, 0)  # (panel_row0, panel_col0) of the multi region

        # UI Components
        self._setup_ui()
        self._setup_menu()

    def _setup_ui(self):
        """Build UI layout."""
        workspace = QWidget()
        shell = QVBoxLayout(workspace)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        self.setCentralWidget(workspace)
        self._setup_workspace_header(shell)

        # Main layout: history | canvas | legend | preview+actions, as a splitter
        # so each column can be dragged wider/narrower live. The legend gets its
        # own full-height column (rather than being stacked under the preview)
        # so all classes are readable at once without scrolling.
        main_layout = QSplitter(Qt.Horizontal)
        main_layout.setChildrenCollapsible(False)
        main_layout.setHandleWidth(3)
        shell.addWidget(main_layout, 1)
        self.main_layout = main_layout

        # Left: History panel (placeholder until session loads)
        self.history_panel = self._empty_panel("PANELS", "Open an observation\nto navigate its panels.")
        self.history_panel.setMinimumWidth(140)
        self.history_panel.setMaximumWidth(200)
        main_layout.addWidget(self.history_panel)

        # Center: Panel canvas
        self.canvas = PanelCanvas()
        self.canvas.setMinimumWidth(280)
        self.canvas.on_block_clicked = self._on_block_clicked
        self.canvas.on_block_paint = self._on_block_paint
        self.canvas.on_block_paint_end = self._on_block_paint_end
        self.canvas.on_selection_made = self._on_selection_made
        main_layout.addWidget(self.canvas)
        self._setup_empty_workspace()
        # The Class Summary replaces the canvas IN THIS COLUMN rather than opening
        # its own window: on macOS a non-modal child of a fullscreen/maximised
        # window is placed on a different Space and never surfaces, so "open"
        # succeeds while nothing appears. Swapping the centre pane (the same
        # insert-then-detach idiom the legend/history swaps use) can't be lost
        # behind anything, and leaves the splitter's own geometry untouched.
        self._summary_page = None
        self._showing_summary = False

        # Legend column (placeholder until a session loads and classes are known)
        self.legend_panel = self._empty_panel("TERRAIN CLASSES", "Your class scheme and\nshortcuts will appear here.")
        self.legend_panel.setMinimumWidth(210)
        main_layout.addWidget(self.legend_panel)

        # Right: Preview (top) + action buttons
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(14, 16, 14, 14)
        right_layout.setSpacing(8)
        self.right_layout = right_layout

        # Side preview (top)
        self.preview = SidePreview()
        right_layout.addWidget(self.preview)

        right_layout.addWidget(self._section_label("VIEW & ANALYSIS"))

        # Action buttons (disabled until a session loads)
        self.overview_button = QPushButton("Observation overview     O")
        self.overview_button.setEnabled(False)
        self.overview_button.setCheckable(True)
        self.overview_button.clicked.connect(self._toggle_overview)
        right_layout.addWidget(self.overview_button)

        # Only shown in predictions mode (mars-inference): toggles the class-color
        # overlay for a Mahalanobis-distance epistemic-uncertainty heatmap.
        self.uncertainty_button = QPushButton("Uncertainty heatmap")
        self.uncertainty_button.setEnabled(False)
        self.uncertainty_button.setCheckable(True)
        self.uncertainty_button.setVisible(self.predictions_mode)
        self.uncertainty_button.clicked.connect(self._toggle_uncertainty_layer)
        right_layout.addWidget(self.uncertainty_button)

        # Only shown in predictions mode: pixel-wise (raw per-pixel model output,
        # the default) vs. blockwise (LabelStore's solid-per-block class, same
        # as the classic labeling view) coloring. Disabled until pixel-level
        # crops actually exist (_run_prediction populates them; a cache-hit
        # reload of previously-saved predictions never does, see
        # block_pixel_predictions's docstring in __init__).
        # Pixel maps, per-block confidence and the local Neural-PCA gallery are
        # products of the prediction pass held in memory only (the .parquet cache
        # stores one class id per block), so a cache hit leaves all three empty.
        # This re-runs inference to rebuild them without restarting the app.
        self.rerun_button = QPushButton("Re-run inference")
        self.rerun_button.setEnabled(False)
        self.rerun_button.setVisible(self.predictions_mode)
        self.rerun_button.clicked.connect(self._rerun_inference)
        right_layout.addWidget(self.rerun_button)

        self.render_mode_button = QPushButton("Pixel-wise view")
        self.render_mode_button.setEnabled(False)
        self.render_mode_button.setCheckable(True)
        self.render_mode_button.setVisible(self.predictions_mode)
        self.render_mode_button.clicked.connect(self._toggle_render_mode)
        right_layout.addWidget(self.render_mode_button)

        right_layout.addStretch(1)
        right_layout.addWidget(self._section_label("SESSION"))
        self.next_panel_button = QPushButton("Complete && next panel")
        self.next_panel_button.setToolTip("Fill remaining unlabeled blocks as NA, save, and advance to the next panel.")
        self.next_panel_button.setEnabled(False)
        self.next_panel_button.clicked.connect(self._go_to_next_panel)
        right_layout.addWidget(self.next_panel_button)
        next_hint = QLabel("Remaining blocks are filled as NA.")
        next_hint.setProperty("role", "muted")
        right_layout.addWidget(next_hint)

        self.export_button = QPushButton("Export Labels")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export)
        right_layout.addWidget(self.export_button)

        # Only shown in predictions mode (mars-inference)
        self.save_predictions_button = QPushButton("Save predictions")
        self.save_predictions_button.setProperty("role", "primary")
        self.save_predictions_button.setEnabled(False)
        self.save_predictions_button.setVisible(self.predictions_mode)
        self.save_predictions_button.clicked.connect(self._save_predictions)
        right_layout.addWidget(self.save_predictions_button)

        right_widget = QWidget()
        right_widget.setObjectName("inspector")
        right_widget.setMinimumWidth(268)
        right_widget.setLayout(right_layout)
        inspector_scroll = QScrollArea()
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inspector_scroll.setMinimumWidth(280)
        inspector_scroll.setWidget(right_widget)
        main_layout.addWidget(inspector_scroll)
        for button in right_widget.findChildren(QPushButton):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Canvas gets the extra space when the window resizes; history/legend/
        # right columns stay put unless the user drags a handle themselves.
        main_layout.setStretchFactor(main_layout.indexOf(self.canvas), 1)
        main_layout.setSizes([156, 742, 250, 280])
        # setChildrenCollapsible(False) alone leaves isCollapsible() reporting
        # True per pane in this Qt build (childrenCollapsible is the runtime
        # default; isCollapsible() reflects each pane's own override, which
        # stays at Qt's True default until set explicitly) -- set it per pane
        # too so a stray drag can't hide the canvas or legend entirely.
        for i in range(main_layout.count()):
            main_layout.setCollapsible(i, False)

        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.status_label = QLabel("Ready · Open a JP2 or GeoTIFF observation to begin")
        self.status_label.setMinimumWidth(0)
        self.status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.statusBar.addWidget(self.status_label, 1)
        self.statusBar.addPermanentWidget(QLabel("LOCAL WORKSPACE"))

    @staticmethod
    def _section_label(text):
        label = QLabel(text)
        label.setProperty("role", "section")
        label.setContentsMargins(0, 8, 0, 4)
        return label

    def _empty_panel(self, title, message):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.addWidget(self._section_label(title))
        description = QLabel(message)
        description.setWordWrap(True)
        description.setProperty("role", "muted")
        layout.addWidget(description)
        layout.addStretch()
        return panel

    def _workspace_button(self, text, callback, *, primary=False):
        button = QPushButton(text)
        button.setProperty("role", "primary" if primary else "compact")
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.clicked.connect(callback)
        return button

    def _setup_workspace_header(self, shell):
        header = QFrame()
        header.setObjectName("workspaceHeader")
        row = QHBoxLayout(header)
        row.setContentsMargins(20, 12, 20, 12)
        row.setSpacing(14)
        mark = QLabel("M")
        mark.setObjectName("brandMark")
        row.addWidget(mark)
        identity = QVBoxLayout()
        identity.setSpacing(2)
        brand = QLabel("MARS OBS")
        brand.setObjectName("brand")
        identity.addWidget(brand)
        tagline = QLabel("SURFACE OBSERVATION WORKSPACE")
        tagline.setProperty("role", "section")
        identity.addWidget(tagline)
        row.addLayout(identity)
        row.addSpacing(24)
        self.observation_label = QLabel("No observation open")
        self.observation_label.setProperty("role", "muted")
        self.observation_label.setMinimumWidth(0)
        self.observation_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row.addWidget(self.observation_label, 1)
        badge = QLabel("PREDICTION REVIEW" if self.predictions_mode else "TERRAIN LABELING")
        badge.setObjectName("modeBadge")
        row.addWidget(badge)
        row.addWidget(self._workspace_button("Open observation", self._on_open_file))
        self.workspace_save_button = self._workspace_button(
            "Save predictions" if self.predictions_mode else "Save labels",
            lambda: self._save_predictions() if self.predictions_mode else self._autosave_session(),
            primary=True,
        )
        self.workspace_save_button.setEnabled(False)
        row.addWidget(self.workspace_save_button)
        shell.addWidget(header)

        toolbar = QFrame()
        toolbar.setObjectName("viewToolbar")
        tools = QHBoxLayout(toolbar)
        tools.setContentsMargins(16, 6, 16, 6)
        tools.setSpacing(6)
        self.view_label = QLabel("IMAGE VIEWER")
        self.view_label.setProperty("role", "section")
        tools.addWidget(self.view_label)
        tools.addStretch()
        hint = QLabel("Class key to label  ·  Shift + drag to select")
        hint.setProperty("role", "muted")
        tools.addWidget(hint)
        tools.addSpacing(16)
        self.zoom_out_button = self._workspace_button("−", lambda: self._workspace_zoom(False))
        self.zoom_out_button.setToolTip("Zoom out (−)")
        self.zoom_in_button = self._workspace_button("+", lambda: self._workspace_zoom(True))
        self.zoom_in_button.setToolTip("Zoom in (+)")
        self.fit_button = self._workspace_button("Fit panel", self._fit_panel)
        self.help_button = self._workspace_button("Shortcuts  ?", self._show_help)
        for button in (self.zoom_out_button, self.zoom_in_button, self.fit_button, self.help_button):
            button.setEnabled(False)
            tools.addWidget(button)
        shell.addWidget(toolbar)

    def _setup_empty_workspace(self):
        self.empty_workspace = QWidget(self.canvas.viewport())
        self.empty_workspace.setObjectName("emptyWorkspace")
        wrapper = QVBoxLayout(self.canvas.viewport())
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.addWidget(self.empty_workspace)
        content = QVBoxLayout(self.empty_workspace)
        content.setContentsMargins(30, 30, 30, 30)
        content.setSpacing(16)
        content.addStretch()
        eyebrow = QLabel("EXPLORE · LABEL · REVIEW")
        eyebrow.setObjectName("welcomeEyebrow")
        content.addWidget(eyebrow)
        title = QLabel("A closer look at Mars.")
        title.setObjectName("welcomeTitle")
        title.setWordWrap(True)
        content.addWidget(title)
        description = QLabel("Open a HiRISE observation to label terrain, inspect native image crops, and review model predictions.")
        description.setWordWrap(True)
        description.setProperty("role", "muted")
        description.setMaximumWidth(410)
        content.addWidget(description)
        content.addWidget(self._workspace_button("Open observation", self._on_open_file, primary=True), 0, Qt.AlignmentFlag.AlignLeft)
        formats = QLabel("JP2 / GeoTIFF  ·  Existing sessions resume automatically")
        formats.setWordWrap(True)
        formats.setProperty("role", "muted")
        content.addWidget(formats)
        content.addStretch()

    def _fit_panel(self):
        if not self.session:
            return
        if self._showing_summary:
            self._close_class_summary()
        self._set_view("panel")
        self._set_zoom(1)

    def _workspace_zoom(self, zoom_in):
        if self._showing_summary:
            self._close_class_summary()
        if zoom_in:
            self._zoom_in()
        else:
            self._zoom_out()

    def _sync_workspace(self):
        """Keep shell metadata in sync without scanning labels or reading imagery."""
        self._sync_legend_selection()
        loaded = self.session is not None
        self.empty_workspace.setVisible(not loaded)
        for button in (self.workspace_save_button, self.zoom_out_button, self.zoom_in_button, self.fit_button):
            button.setEnabled(loaded)
        self.help_button.setEnabled(self.classes_scheme is not None)
        if not loaded:
            self.observation_label.setText("No observation open")
            self.observation_label.setToolTip("")
            self.view_label.setText("IMAGE VIEWER")
            return
        grid = self.session.grid
        self.observation_label.setText(grid.obs_id)
        self.observation_label.setToolTip(str(self.session.raster.path))
        if self._showing_summary:
            view_text = "CLASS SUMMARY   ·   COVERAGE & MODEL ANALYSIS"
        elif self.view_mode == "overview":
            view_text = f"OBSERVATION OVERVIEW   ·   {grid.num_panels} PANELS"
        elif self.view_mode == "multi":
            view_text = f"REGIONAL VIEW   ·   {self.multi_span} × {self.multi_span} PANELS"
        else:
            view_text = (
                f"PANEL {self.current_panel_idx + 1:03d} / {grid.num_panels:03d}"
                f"   ·   {grid.block_size} PX BLOCKS"
            )
        self.view_label.setText(view_text)

    def _setup_menu(self):
        """Build menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open observation...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_open_file)
        file_menu.addAction(open_action)

        self.save_action = QAction("Save Labels", self)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(
            lambda: self._save_predictions() if self.predictions_mode else self._autosave_session()
        )
        file_menu.addAction(self.save_action)

        self.labels_folder_action = QAction("Set Labels Folder...", self)
        self.labels_folder_action.triggered.connect(self._on_set_labels_folder)
        file_menu.addAction(self.labels_folder_action)

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
            "Open observation",
            "",
            "Observations (*.jp2 *.JP2 *.tif *.TIF *.tiff *.TIFF);;All Files (*)",
        )

        if not path:
            return

        self._load_observation(Path(path))
        self._sync_workspace()

    def _on_set_labels_folder(self):
        """File→Set Labels Folder: choose where labels are saved to / resumed from."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Labels Folder",
            str(self.labels_dir),
        )
        if not path:
            return

        self.labels_dir = Path(path)
        if self.session is not None:
            # A session is already open; saves now target the new folder. It does not
            # retroactively pull in labels there — reopen the image to resume those.
            QMessageBox.information(
                self,
                "Labels folder changed",
                f"Labels will now be saved to:\n{self.labels_dir}\n\n"
                "To resume labels already stored there, reopen the observation "
                "(File → Open JP2).",
            )
        self.status_label.setText(f"Labels folder: {self.labels_dir}")

    def _offer_open_without_labels(self, jp2_path: Path, error: Exception) -> Optional[Path]:
        """Offer a fresh session with a separate save destination for recovery."""
        destination = self.labels_dir / f"{jp2_path.stem}-fresh-{datetime.now():%Y%m%d-%H%M%S-%f}"
        warning = QMessageBox(self)
        warning.setIcon(QMessageBox.Icon.Warning)
        warning.setWindowTitle("Could not load saved labels")
        warning.setText(f"The saved labels for '{jp2_path.name}' could not be loaded.")
        warning.setInformativeText(
            f"{error}\n\nYou can still open the observation without those labels. "
            "Your existing files will be kept, and new labels will be saved to:\n"
            f"{destination}"
        )
        open_button = warning.addButton(
            "Open without saved labels", QMessageBox.ButtonRole.AcceptRole
        )
        warning.addButton(QMessageBox.StandardButton.Cancel)
        warning.setDefaultButton(open_button)
        warning.exec()
        if warning.clickedButton() is open_button:
            return destination
        self.status_label.setText("Loading cancelled")
        return None

    def _load_observation(self, jp2_path: Path) -> bool:
        """Load an observation, optionally starting fresh if saved labels fail."""
        previous_session = self.session
        if previous_session and not self.predictions_mode:
            if not self._autosave_session(note="before opening another observation"):
                return False
        labels_dir = self.labels_dir
        resume = not (self.predictions_mode and self.ignore_cached_predictions)
        raster = None
        new_session = None

        try:
            # Open raster
            raster = RasterSource(jp2_path)
            raster.open()

            # Classification tile size: derive from --resolution (meters/tile-side) and
            # the observation's native GSD if given, else fall back to the configured
            # pixel block_size unchanged.
            panel_size = self.config.geometry.panel_size
            if self.resolution_m is not None:
                block_size = max(1, round(self.resolution_m / raster.gsd))
                # Panels must tile into whole blocks; the derived block size rarely
                # divides the configured panel exactly, so shrink the panel to the
                # nearest whole multiple of it (no partial blocks dropped at edges).
                panel_size = max(block_size, (panel_size // block_size) * block_size)
                self.status_label.setText(
                    f"Tile: {block_size}px "
                    f"({block_size * raster.gsd:g} m/side at {raster.gsd:g} m/px)"
                )
            else:
                block_size = self.config.geometry.block_size

            # Reconcile with any saved labels for this observation: they may have been
            # made for a different image (wrong picture) or a different tile resolution.
            obs_id = jp2_path.stem
            saved = None
            if resume:
                try:
                    saved = Session.read_saved_metadata(labels_dir, obs_id)
                    if saved is not None:
                        if not isinstance(saved, dict):
                            raise ValueError("Saved label metadata is not a valid object.")
                        for field in ("block_size", "panel_size", "img_width", "img_height"):
                            value = saved.get(field)
                            if value is not None and (type(value) is not int or value <= 0):
                                raise ValueError(f"Saved labels have invalid {field}.")
                        if (saved.get("panel_size") and saved.get("block_size")
                                and saved["panel_size"] % saved["block_size"]):
                            raise ValueError("Saved panel size is not divisible by the block size.")
                except Exception as error:
                    labels_dir = self._offer_open_without_labels(jp2_path, error)
                    if labels_dir is None:
                        return False
                    saved = None
                    resume = False
            if saved is not None:
                # Wrong-picture guard: saved image dimensions differ from this raster.
                saved_w, saved_h = saved.get("img_width"), saved.get("img_height")
                if (
                    saved_w is not None
                    and saved_h is not None
                    and (saved_w != raster.width or saved_h != raster.height)
                ):
                    labels_dir = self._offer_open_without_labels(
                        jp2_path,
                        ValueError(
                            f"Saved labels were made for a {saved_w}×{saved_h} px image, "
                            f"but '{jp2_path.name}' is {raster.width}×{raster.height} px."
                        ),
                    )
                    if labels_dir is None:
                        return False
                    saved = None
                    resume = False

            if saved is not None:
                # Wrong-resolution guard: adopt the saved tile geometry so the existing
                # labels line up (the requested --resolution is ignored for this image).
                saved_block, saved_panel = saved.get("block_size"), saved.get("panel_size")
                if (
                    not self.ignore_cached_predictions
                    and saved_block is not None
                    and saved_panel is not None
                    and (saved_block != block_size or saved_panel != panel_size)
                ):
                    QMessageBox.information(
                        self,
                        "Resuming at saved resolution",
                        f"Saved labels for '{obs_id}' use {saved_block}px tiles "
                        f"({saved_block * raster.gsd:g} m), but this run requested "
                        f"{block_size}px ({block_size * raster.gsd:g} m).\n\n"
                        "Loading at the saved resolution so the existing labels align. "
                        "To label at the requested resolution instead, choose a "
                        "different labels folder or remove the saved files first.",
                    )
                    block_size, panel_size = saved_block, saved_panel

            # Scale decision before anything expensive: it changes the tile size,
            # so it has to happen before the grid and the preprocessing pass.
            gsd_ratio = self._resolve_gsd_scaling(raster)
            if gsd_ratio != 1.0:
                from marslabeler.inference.modelio import (
                    REQUIRED_STRIDE,
                    native_block_size_for_gsd,
                )

                # One model window now covers fewer native pixels, so the tile has
                # to shrink to match -- otherwise a block would extend beyond the
                # window that predicted it.
                stride = REQUIRED_STRIDE.get("simmim", 256)
                block_size = native_block_size_for_gsd(block_size, stride, gsd_ratio)
                panel_size = max(block_size, (panel_size // block_size) * block_size)
                self.status_label.setText(
                    f"Scale-matched tiles: {block_size}px "
                    f"({block_size * raster.gsd:g} m/side)"
                )

            # Warn before committing to an impractically large tile count (mirrors the
            # Grid's own block-indexing: full blocks_per_panel for every panel).
            panels_across = (raster.width + panel_size - 1) // panel_size
            panels_down = (raster.height + panel_size - 1) // panel_size
            blocks_per_panel = (panel_size // block_size) ** 2
            total_blocks = panels_across * panels_down * blocks_per_panel
            if total_blocks > self.TILE_COUNT_WARN_THRESHOLD:
                reply = QMessageBox.warning(
                    self,
                    "Large tile count",
                    f"This resolution produces {total_blocks:,} tiles "
                    f"({blocks_per_panel:,} per panel, {block_size}px each).\n\n"
                    "Loading may be slow and there will be a lot of tiles to label. "
                    "Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.status_label.setText("Loading cancelled")
                    return False

            # Create grid
            grid = Grid(
                img_width=raster.width,
                img_height=raster.height,
                panel_size=panel_size,
                block_size=block_size,
                obs_id=jp2_path.stem,
                transform=raster.transform,
                crs=raster.crs,
            )

            # Load classes
            classes_scheme = load_classes(self.config.paths.classes_file)

            # Create or load session
            try:
                new_session = Session.load_or_create(
                    jp2_path, grid, self.config.to_dict(), labels_dir,
                    labeler=self.config.labeler or "unknown", resume=resume,
                )
            except Exception as error:
                if not resume or not (labels_dir / f"{obs_id}.parquet").exists():
                    raise
                labels_dir = self._offer_open_without_labels(jp2_path, error)
                if labels_dir is None:
                    return False
                new_session = Session.load_or_create(
                    jp2_path, grid, self.config.to_dict(), labels_dir,
                    labeler=self.config.labeler or "unknown", resume=False,
                )

            # Commit the new session only after preprocessing is accepted, keeping
            # the previous observation usable when opening is cancelled.
            preprocess_dialog = PreprocessDialog(raster, grid, self.config.to_dict())
            preprocess_dialog.start_preprocessing()
            result = preprocess_dialog.exec()
            preprocess_dialog.worker.wait()
            if result != QDialog.DialogCode.Accepted:
                self.status_label.setText("Loading cancelled")
                return False

            recovered = labels_dir != self.labels_dir
            self.session = new_session
            self.labels_dir = labels_dir
            self.gsd_ratio = gsd_ratio
            self.classes_scheme = classes_scheme
            self._resolve_na_class()
            if previous_session:
                previous_session.raster.close()

            if self.predictions_mode:
                self.display_layer = "classes"
                self.block_confidence = {}
                self.block_uncertainty = {}
                self.uncertainty_button.setChecked(False)
                self.uncertainty_button.setEnabled(False)
                self.prediction_render_mode = "pixelwise"
                self.block_pixel_predictions = {}
                self.render_mode_button.setChecked(False)
                self.render_mode_button.setText("Pixel-wise view")
                self.render_mode_button.setEnabled(False)

            # A restored session carries the class NAMES that were current when it
            # was saved; class_id is authoritative, so re-derive them rather than
            # letting a renamed class show its old label in exports and details.
            if self.classes_scheme is not None:
                id_to_name = {c.id: c.name for c in self.classes_scheme.classes.values()}
                id_to_name[self.classes_scheme.abstain.id] = self.classes_scheme.abstain.name
                id_to_name[self.classes_scheme.nodata.id] = self.classes_scheme.nodata.name
                renamed = self.session.labels.refresh_class_names(id_to_name)
                if renamed:
                    print(f"[session] refreshed {renamed} stale class names from "
                          f"{self.config.paths.classes_file}")

            # Create keyboard controller
            self.controller = KeyboardController(self.session, self.classes_scheme)
            self.controller.on_label_changed = self._on_labels_changed
            self.controller.on_panel_changed = self._on_panel_changed_kb
            self.controller.on_cursor_changed = self._on_cursor_changed
            self.controller.on_show_help = self._show_help
            self.controller.on_next_panel = self._go_to_next_panel

            # Setup autosave timer
            self._setup_autosave()

            # Store skip decisions for later use
            self.skip_decisions = preprocess_dialog.get_skip_decisions()
            raster.close()
            # Share them so navigation doesn't re-decode a window per block.
            self.session.skip_decisions = self.skip_decisions

            # Detect fully off-swath panels: retire them (mark nodata), hide + skip
            self.empty_panels = self._compute_empty_panels()
            self._retire_empty_panels()
            # Start on the first panel that actually has image content
            if self.session.current_block().panel_idx in self.empty_panels:
                first = self._first_nonempty_panel()
                if first is not None:
                    self.session.move_to_panel(first)

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
            self.overview_button.setEnabled(True)
            self.saved_complete_panels = set()

            self.status_label.setText(f"Loaded: {jp2_path.stem}")
            if recovered:
                self.status_label.setText(
                    f"Opened without saved labels · New labels folder: {self.labels_dir}"
                )
            # Only a real observation open schedules startup help. Rendering a
            # panel (including hidden test/review windows) must not open modals.
            if not self.help_shown_on_startup:
                self.help_shown_on_startup = True
                QTimer.singleShot(100, self._show_help)
            return True

        except Exception as e:
            # Surface the failure instead of silently leaving a blank "(No session)"
            # window; keep the full traceback on stderr for debugging.
            traceback.print_exc()
            self.status_label.setText(f"Error: {str(e)}")
            QMessageBox.critical(
                self,
                "Could not load observation",
                f"Failed to load {jp2_path.name}:\n\n{e}",
            )
            return False
        finally:
            if raster is not None:
                raster.close()
            if new_session is not None and new_session is not self.session:
                new_session.raster.close()
            self._sync_workspace()

    # ------------------------------------------------------------------ #
    # Predictions mode (mars-inference): model-seeded session + explicit save
    # ------------------------------------------------------------------ #

    @staticmethod
    def _model_signature(model_path: Path) -> str:
        """Cheap fingerprint of a checkpoint file, to detect a changed model on reload."""
        stat = model_path.stat()
        return f"{stat.st_size}_{int(stat.st_mtime)}"

    def load_for_inference(self, jp2_path: Path, model_path: Path) -> None:
        """Open jp2_path in predictions mode: reuse cached predictions for this exact
        model if present, otherwise run inference (with a progress dialog) and seed
        them into a fresh session. Requires predictions_mode=True.
        """
        if not self.predictions_mode:
            raise RuntimeError("load_for_inference requires predictions_mode=True")

        model_id = model_path.stem
        model_sig = self._model_signature(model_path)
        previous_session = self.session
        previous_settings = self.model_path, self.labels_dir, self.config.labeler
        # The loader needs the requested model for its GSD check and cache folder,
        # but a cancelled open must retain the previous session's save destination.
        self.model_path = model_path
        self.labels_dir = Path(self.config.paths.predictions_dir) / model_id
        self.config.labeler = f"model:{model_id}"
        obs_id = jp2_path.stem
        if not self._load_observation(jp2_path):
            if self.session is previous_session:
                self.model_path, self.labels_dir, self.config.labeler = previous_settings
            return  # failure/cancellation has already been reported

        self.setWindowTitle(f"Mars Obs Labeler — Predictions [{model_id}] — {jp2_path.name}")
        self.npca_gallery = self._try_load_npca_gallery(model_path)
        self.local_npca_gallery = {}
        self._model_sig = None

        # Inspect the cache only after loading has handled unreadable saved labels.
        # Recovery uses a fresh folder, and --fresh must never read the old cache.
        saved = (
            Session.read_saved_metadata(self.labels_dir, obs_id)
            if not self.ignore_cached_predictions else None
        )
        cache_valid = (
            saved is not None
            and saved.get("model_sig") == model_sig
            and not self.ignore_cached_predictions
        )
        if saved is not None and not cache_valid:
            QMessageBox.information(
                self,
                "Cached predictions are stale",
                f"Predictions saved in\n{self.labels_dir}\n"
                "were made with a different version of this checkpoint file "
                "(size/modified time differ). Re-running inference.",
            )

        self.save_predictions_button.setEnabled(True)
        self.uncertainty_button.setEnabled(True)

        self.rerun_button.setEnabled(True)
        if cache_valid:
            self._model_sig = model_sig
            note = (
                "Pixel-wise view, per-block confidence and this observation's "
                "Neural-PCA gallery are rebuilt by the inference pass and are not "
                "stored in the prediction cache -- use \u21BB Re-run inference "
                "for them."
            )
            self.render_mode_button.setToolTip(note)
            self.uncertainty_button.setToolTip(note)
            self.status_label.setText(f"Loaded cached predictions ({model_id}) for {obs_id}")
            return

        self._run_prediction(model_path, model_sig)

    def _resolve_gsd_scaling(self, raster) -> float:
        """Decide the observation/training GSD ratio to run at, asking if needed.

        A segmentation model learns terrain at a fixed pixels-per-metre, so
        imagery of a different GSD is a scale domain shift: the same landform
        covers a different number of pixels than in any training crop. Measured on
        labelled NOAH-H ground at a 2.07x mismatch, correcting the scale is worth
        about +40% mIoU for ~4x the compute, so it is offered rather than assumed.

        Returns the ratio to hand `build_inference_plan`; 1.0 means "read native
        pixels", i.e. no correction.
        """
        import math

        from marslabeler.inference.modelio import raster_gsd_metres, training_gsd_metres

        dataset = getattr(raster, "_dataset", None)
        observed = raster_gsd_metres(dataset) if dataset is not None else None
        if not observed:
            return 1.0

        reference = None
        source = ""
        if self.model_path is not None:
            reference = training_gsd_metres(
                self.model_path, ai4exomars_path=self.config.inference.ai4exomars_path
            )
            if reference:
                source = f"the imagery {self.model_path.stem} was trained on"
        if not reference:
            reference = float(self.config.inference.expected_gsd_m)
            source = "the expected NOAH-H product scale (inference.expected_gsd_m)"
        if reference <= 0:
            return 1.0

        ratio = observed / reference
        if abs(math.log2(ratio)) <= float(self.config.inference.gsd_log2_tolerance):
            return 1.0

        direction = "coarser" if ratio > 1 else "finer"
        detail = (
            f"This observation is {observed:.4g} m/pixel, but {source} is "
            f"{reference:.4g} m/pixel -- {ratio:.2f}x {direction}.\n\n"
            "Matching the training scale resamples each window so a model pixel "
            f"spans {reference:.4g} m again. On labelled NOAH-H ground at this "
            "mismatch that was worth about +40% mIoU, but it needs roughly "
            f"{ratio ** 2:.0f}x the inference time.\n\n"
            "Running natively is faster, but predictions are not comparable to "
            "the model's validation scores."
        )

        if self.match_training_gsd is not None:
            # Decided on the command line -- report, don't ask.
            choice = bool(self.match_training_gsd)
            self.status_label.setText(
                f"{observed:.4g} m/px vs {reference:.4g} m/px training scale; "
                f"{'matching' if choice else 'running native'}"
            )
        else:
            answer = QMessageBox.question(
                self, "Resolution mismatch",
                detail + "\n\nMatch the training scale?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            choice = answer == QMessageBox.StandardButton.Yes

        if not choice:
            self.status_label.setText(
                f"Warning: running at {observed:.4g} m/px, {ratio:.2f}x {direction} "
                f"than the {reference:.4g} m/px training scale"
            )
            return 1.0
        print(f"[gsd] matching training scale: {observed:.4g} -> {reference:.4g} m/px "
              f"(ratio {ratio:.3f})")
        return ratio

    def _retire_high_nodata_blocks(self) -> int:
        """Mark blocks above inference.nodata_skip_threshold nodata as nodata.

        Uses the per-block nodata_fraction already computed by the preprocessing
        pass (self.skip_decisions) -- no extra raster reads. Distinct from
        skip.nodata_skip_threshold (mars-label's "don't bother a human with this
        block" cutoff): a majority-nodata block is still meaningfully labelable
        by a human, but a model prediction on one is closer to noise, so this
        defaults stricter and is checked separately right before inference.
        """
        threshold = self.config.inference.nodata_skip_threshold
        ids = [
            b.block_id
            for b in self.session.grid.iter_blocks()
            if self.session.labels.get_record(b.block_id).status != "nodata"
            and self.skip_decisions.get(b.block_id, {}).get("nodata_fraction", 0.0) > threshold
        ]
        if ids:
            self.session.labels.set_nodata_bulk(ids)
        return len(ids)

    def _rerun_inference(self) -> None:
        """Re-run inference over the open observation, replacing cached predictions.

        Needed because the prediction cache stores one class id per block; the
        per-pixel maps, per-block confidence and the per-observation Neural-PCA
        gallery live only in memory for the lifetime of a prediction run.
        """
        if not self.session or not self.model_path:
            return
        if not self.predictions_mode:
            return
        confirm = QMessageBox.question(
            self,
            "Re-run inference?",
            "Re-run the model over every non-nodata block of this observation?\n\n"
            "This replaces the current predictions and restores the pixel-wise "
            "view, per-block confidence and this observation's Neural-PCA gallery.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._run_prediction(self.model_path, self._model_signature(self.model_path))

    def _run_prediction(self, model_path: Path, model_sig: str) -> None:
        """Run the model over every non-nodata block and seed the results as labels."""
        from marslabeler.ui.predictdialog import PredictDialog

        retired = self._retire_high_nodata_blocks()
        if retired:
            self.status_label.setText(
                f"Skipping {retired} blocks >{self.config.inference.nodata_skip_threshold:.0%} "
                "nodata (off-swath) -- running inference on the rest..."
            )

        blocks = [
            b
            for b in self.session.grid.iter_blocks()
            if self.session.labels.get_record(b.block_id).status != "nodata"
        ]

        dialog = PredictDialog(
            self.session.raster,
            blocks,
            self.session.grid.block_size,
            model_path,
            self.classes_scheme,
            {
                "device": self.config.inference.device,
                "ai4exomars_path": self.config.inference.ai4exomars_path,
                "batch_size": self.config.inference.batch_size,
                "context_multiplier": self.config.inference.context_multiplier,
                "gsd_ratio": self.gsd_ratio,
            },
        )
        dialog.start()

        if dialog.exec() != QDialog.DialogCode.Accepted:
            msg = f"Inference failed: {dialog.error}" if dialog.error else "Inference cancelled"
            self.status_label.setText(msg)
            if dialog.error:
                QMessageBox.critical(self, "Inference failed", dialog.error)
            return

        predictions = dialog.get_predictions()
        class_names = {cid: c.name for cid, c in self.classes_scheme.classes.items()}
        self.session.labels.seed_bulk(predictions, class_names)
        self.block_confidence = dialog.get_confidence()
        self.block_pixel_predictions = dialog.get_pixel_predictions()
        self.local_npca_gallery = dialog.get_local_npca()
        self.render_mode_button.setEnabled(bool(self.block_pixel_predictions))
        self._model_sig = model_sig

        self._refresh_view()
        self._refresh_history()
        self.status_label.setText(
            f"Predicted {len(predictions)} blocks with {model_path.stem} -- "
            "review, then press Save Predictions"
        )

    def _save_predictions(self) -> None:
        """Persist the current (possibly reviewed) predictions to the predictions cache."""
        if not self.session or not self.predictions_mode:
            return
        try:
            extra_meta = {
                "model_sig": self._model_sig,
                "model_stem": self.model_path.stem if self.model_path else None,
            }
            self.session.save_session(self.labels_dir, extra_meta=extra_meta)
            self.status_label.setText(f"Predictions saved to {self.labels_dir}")
        except Exception as e:
            self.status_label.setText(f"Save error: {str(e)}")

    def _try_load_npca_gallery(self, model_path: Path) -> Optional[dict]:
        """Load the model's neural-PCA gallery sidecar, if one has been fitted yet.

        Returns None (rather than raising) whenever it isn't available -- missing
        artifact, vision_backend not installed, corrupt file -- since this is a
        purely additive analysis layer that shouldn't block predictions from
        opening. See AI4ExoMars/vision_backend/pc_align/fit_neural_pca.py.
        """
        try:
            from marslabeler.inference.modelio import sidecar_path
            from marslabeler.inference.npca_gallery import load_npca_gallery

            return load_npca_gallery(
                sidecar_path(model_path, "npca.pt"),
                ai4exomars_path=self.config.inference.ai4exomars_path,
            )
        except FileNotFoundError:
            return None
        except Exception:
            return None

    def _toggle_render_mode(self) -> None:
        """Pixel-wise/Block-wise view button: switch how the "classes" display
        layer renders, without touching what's stored (LabelStore keeps the
        block-level class either way; blockwise reads it directly, pixelwise
        renders the cached per-pixel crops instead -- see block_pixel_predictions)."""
        if self.render_mode_button.isChecked():
            self.prediction_render_mode = "blockwise"
            self.render_mode_button.setText("Block-wise view")
        else:
            self.prediction_render_mode = "pixelwise"
            self.render_mode_button.setText("Pixel-wise view")
        self._refresh_label_overlay()

    def _toggle_uncertainty_layer(self) -> None:
        """Uncertainty Heatmap button: swap the class-color overlay for the
        Mahalanobis epistemic-uncertainty heatmap (computing it on first use)."""
        if not self.session:
            self.uncertainty_button.setChecked(False)
            return

        if not self.uncertainty_button.isChecked():
            self.display_layer = "classes"
            self._refresh_view()
            return

        if not self.block_uncertainty:
            self._compute_uncertainty_layer()

        if self.block_uncertainty:
            self.display_layer = "uncertainty"
        else:
            self.uncertainty_button.setChecked(False)
            self.display_layer = "classes"
        self._refresh_view()

    def _compute_uncertainty_layer(self) -> None:
        """Run the Mahalanobis uncertainty scorer over every non-nodata block."""
        from marslabeler.ui.uncertaintydialog import UncertaintyDialog

        # Idempotent: a no-op if _run_prediction already retired these blocks (the
        # normal case); catches the cache-hit path, where it never ran.
        self._retire_high_nodata_blocks()

        blocks = [
            b
            for b in self.session.grid.iter_blocks()
            if self.session.labels.get_record(b.block_id).status != "nodata"
        ]

        dialog = UncertaintyDialog(
            self.session.raster,
            blocks,
            self.session.grid.block_size,
            self.model_path,
            {
                "device": self.config.inference.device,
                "ai4exomars_path": self.config.inference.ai4exomars_path,
                "batch_size": self.config.inference.batch_size,
                "context_multiplier": self.config.inference.context_multiplier,
                "gsd_ratio": self.gsd_ratio,
            },
        )
        dialog.start()

        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.status_label.setText("Uncertainty heatmap unavailable")
            if dialog.error:
                QMessageBox.information(self, "Uncertainty heatmap unavailable", dialog.error)
            return

        self.block_uncertainty = dialog.get_scores()
        self.status_label.setText(f"Computed uncertainty for {len(self.block_uncertainty)} blocks")

    def _on_legend_class_clicked(self, class_id: int) -> None:
        """Legend row clicked: label the current block with that class.

        Deliberately routed through the same controller entry point the hotkeys
        use, so a click and a key press are the same operation as far as undo,
        auto-advance and panel-change are concerned. Focus is returned to the
        canvas afterwards so the keyboard keeps working without a manual click
        back -- otherwise the legend would steal focus on first use and the
        hotkeys would appear to stop responding.
        """
        if not self.session or not self.controller:
            return
        if self.view_mode == "overview":
            self.status_label.setText(
                "Overview is navigation-only -- open a panel to label."
            )
            return
        if self._showing_summary:
            return

        # A marquee selection is an explicit "label THESE blocks" gesture, so a
        # class click fills it rather than labelling only the cursor's block.
        if self.selection_rect is not None:
            self._fill_selection(class_id)
            self.canvas.setFocus()
            return

        self.controller.label_class(class_id)
        self.canvas.setFocus()

    def _show_class_summary(self) -> None:
        """Legend panel's Summary button: swap the centre view to the class summary.

        Shown in-window (replacing the canvas in the centre column) rather than as a separate
        dialog: on macOS a non-modal child window of a fullscreen/maximised parent
        opens on another Space, so the old dialog reported itself open while
        nothing became visible. Clicking Summary again returns to the map.
        """
        if self._showing_summary:
            self._close_class_summary()
            return
        if not self.session:
            self.status_label.setText("Summary needs an observation loaded first.")
            return
        if not self.classes_scheme:
            self.status_label.setText(
                "Summary unavailable: the class scheme failed to load "
                f"({self.config.paths.classes_file})."
            )
            return

        from marslabeler.ui.summarydialog import ClassSummaryView

        try:
            view = ClassSummaryView(
                self.classes_scheme,
                self.session,
                npca_gallery=self.npca_gallery,
                local_npca_gallery=getattr(self, "local_npca_gallery", {}) or {},
                on_local_npca_clicked=self._on_local_npca_clicked,
                block_confidence=self.block_confidence,
                block_uncertainty=self.block_uncertainty,
            )
        except Exception as exc:
            traceback.print_exc()
            self.status_label.setText(
                f"Could not open Class Summary: {type(exc).__name__}: {exc}"
            )
            QMessageBox.warning(
                self, "Class Summary failed to open",
                f"{type(exc).__name__}: {exc}\n\n"
                "The full traceback was printed to the terminal.",
            )
            return

        view.thumbnail_clicked.connect(self._on_npca_thumbnail_clicked)
        view.close_requested.connect(self._close_class_summary)
        # Take the canvas's width demands so the splitter keeps the user's column
        # widths across the swap; the view's own content scrolls inside it.
        view.setSizePolicy(self.canvas.sizePolicy())
        view.setMinimumWidth(self.canvas.minimumSizeHint().width())

        # Swap the centre column: insert at the canvas's index, detach the canvas
        # (kept alive on self.canvas), and restore the splitter's column widths.
        index = self.main_layout.indexOf(self.canvas)
        sizes = self.main_layout.sizes()
        self.main_layout.insertWidget(index, view)
        self.canvas.setParent(None)
        self.main_layout.setSizes(sizes)
        self._summary_page = view
        self._showing_summary = True
        self.view_label.setText("CLASS SUMMARY   ·   COVERAGE & MODEL ANALYSIS")
        self.status_label.setText(
            f"Class Summary ({view.npca_source} Neural-PCA examples) "
            "-- press Summary again or Esc to return to the map"
        )

    def _close_class_summary(self) -> None:
        """Put the panel canvas back in the centre column."""
        if not self._showing_summary or self._summary_page is None:
            return
        index = self.main_layout.indexOf(self._summary_page)
        sizes = self.main_layout.sizes()
        self.main_layout.insertWidget(index, self.canvas)
        self._summary_page.setParent(None)
        self._summary_page.deleteLater()
        self._summary_page = None
        self.main_layout.setSizes(sizes)
        self._showing_summary = False
        # The canvas had no size while hidden, so re-apply the fit/zoom transform.
        if self.session:
            self.canvas.apply_zoom_view()
        self._sync_workspace()
        self.status_label.setText("Ready")

    def _on_local_npca_clicked(self, block_id: str) -> None:
        """Per-observation Neural-PCA thumbnail clicked: move the cursor to that block.

        Unlike `_on_npca_thumbnail_clicked`, no image switch is involved -- these
        exemplars were ranked over the blocks of the observation already open, so
        this is ordinary in-image navigation.
        """
        if not self.session:
            return
        grid = self.session.grid
        target = None
        for index, block in enumerate(grid.iter_blocks()):
            if block.block_id == block_id:
                target = index
                break
        if target is None:
            QMessageBox.information(
                self, "Can't locate this block",
                f"Block {block_id!r} is not part of the observation currently open.",
            )
            return

        self._close_class_summary()  # so the jump is actually visible
        self.session.move_to_block(target)
        self.current_panel_idx = self.session.current_block().panel_idx
        self._set_view("panel")
        self._refresh_history()
        block = grid.get_block(target)
        self.status_label.setText(
            f"Jumped to {block_id} (panel {block.panel_idx}, "
            f"block {block.block_row},{block.block_col})"
        )

    def _on_npca_thumbnail_clicked(self, source_id: str) -> None:
        """Neural-PCA gallery thumbnail (Class Summary window) clicked: jump
        this window to that exact block, opening its source training mosaic
        first (in plain view -- no fresh inference run) if a different image
        is currently loaded. The gallery's thumbnails are top-activating crops
        from AI4ExoMars's training run, not from whatever observation happens
        to be open here, so this is a genuine "switch images" navigation, not
        just scrolling the current one -- see resolve_training_imagery_path's
        docstring for why.
        """
        from marslabeler.inference.npca_gallery import parse_npca_source_id

        parsed = parse_npca_source_id(source_id)
        if parsed is None:
            QMessageBox.information(
                self, "Can't locate this thumbnail",
                f"Couldn't parse a pixel location out of {source_id!r}.",
            )
            return
        mosaic_stem, col, row = parsed

        if not self.model_path:
            QMessageBox.information(
                self, "Can't locate this thumbnail",
                "No model checkpoint is loaded in this window, so its "
                "training imagery can't be resolved.",
            )
            return

        from marslabeler.inference.modelio import resolve_training_imagery_path

        self.status_label.setText(f"Resolving source image for {source_id!r}...")
        QApplication.processEvents()
        imagery_path = resolve_training_imagery_path(
            self.model_path, ai4exomars_path=self.config.inference.ai4exomars_path
        )
        if imagery_path is None or imagery_path.stem != mosaic_stem:
            QMessageBox.information(
                self, "Can't locate this thumbnail",
                f"Couldn't find the source image ({mosaic_stem!r}) for this "
                "thumbnail on disk -- it may have moved, or come from a "
                "different checkpoint's training run than the one loaded here.",
            )
            self.status_label.setText("Ready")
            return

        already_open = (
            self.session is not None
            and self.session.raster.path is not None
            and Path(self.session.raster.path).resolve() == imagery_path.resolve()
        )
        if not already_open:
            self.status_label.setText(
                f"Opening {imagery_path.name} (this may take a while for a large mosaic)..."
            )
            QApplication.processEvents()
            if not self._load_observation(imagery_path):
                return  # load failed/cancelled; _load_observation already reported it

        block_idx = self.session.grid.block_index_at_pixel(col, row)
        self.session.move_to_block(block_idx)
        self._set_view("panel")
        self._refresh_history()
        self._set_zoom(4)
        self.status_label.setText(f"Jumped to {source_id}")

    def _update_history_panel(self):
        """Swap the history placeholder for the real panel (by reference)."""
        history = HistoryPanel(
            self.session.grid, self.session.labels, hidden_panels=self.empty_panels
        )
        history.on_panel_selected = self._on_panel_selected
        history.setMaximumWidth(200)
        # QSplitter (main_layout) has no replaceWidget() (that's QLayout-only):
        # insert the new widget at the old one's index, then detach the old one
        # synchronously so the splitter never briefly shows both.
        old_widget = self.history_panel
        old_index = self.main_layout.indexOf(old_widget)
        self.main_layout.insertWidget(old_index, history)
        old_widget.setParent(None)
        old_widget.deleteLater()
        self.history_panel = history

    def _update_legend_panel(self):
        """Swap the legend placeholder for the real legend (by reference)."""
        legend = LegendPanel(self.classes_scheme)
        legend.on_summary_clicked = self._show_class_summary
        legend.on_class_clicked = self._on_legend_class_clicked
        # The legend is its own QSplitter column now, and QSplitter has no
        # replaceWidget() (that's QLayout-only) -- same insert-then-detach
        # pattern as _update_history_panel, and it must preserve the column's
        # current width so a user-dragged legend width survives the swap.
        old_widget = self.legend_panel
        old_index = self.main_layout.indexOf(old_widget)
        sizes = self.main_layout.sizes()
        self.main_layout.insertWidget(old_index, legend)
        old_widget.setParent(None)
        old_widget.deleteLater()
        self.main_layout.setSizes(sizes)
        self.legend_panel = legend
        self._sync_legend_selection()

    def _sync_legend_selection(self) -> None:
        """Use the selected block's record, never the last-used paint class."""
        if not isinstance(self.legend_panel, LegendPanel):
            return
        class_id = None
        if self.session:
            record = self.session.labels.get_record(self.session.current_block().block_id)
            if record.status in ("labeled", "abstain"):
                class_id = record.class_id
        self.legend_panel.set_current_class(class_id)

    def _load_current_panel(self):
        """Load and display the panel the cursor is in (synchronous, decimated read)."""
        if not self.session:
            return

        # Keep the displayed panel in sync with the cursor; drop stale selection
        self.current_panel_idx = self.session.current_block().panel_idx
        self._clear_selection()
        self.status_label.setText(f"Loading panel {self.current_panel_idx}...")

        # Synchronous decimated read at the current zoom resolution (fast via GDAL).
        # Read the FULL panel_size extent (black-padded past the image edge), not the
        # clipped w/h from get_panel_coords: the grid overlay and click mapping both
        # assume the canvas spans exactly panel_size x panel_size. Feeding a clipped
        # edge panel here would stretch it to fill the square canvas, so block
        # boundaries would no longer land on the drawn grid lines -- the border
        # panels' clicks would select a different block than the one previewed.
        # This matches what _render_overview / _render_multi already do.
        grid = self.session.grid
        x, y, _w, _h = grid.get_panel_coords(self.current_panel_idx)
        out = self.canvas.canvas_width  # 1600 at zoom 1, larger when zoomed in
        panel_data = self.session.raster.read_window_padded(
            x, y, grid.panel_size, grid.panel_size, out, out
        )
        self._render_panel(panel_data)

    def _render_panel(self, panel_data: np.ndarray):
        """Render the given panel image and its overlays."""
        if not self.session:
            return

        self._sync_workspace()

        # Display panel image
        self.canvas.set_panel_image(panel_data, stretch_percentiles=(1, 99))

        # Set grid
        grid = self.session.grid
        self.canvas.set_grid(grid.blocks_per_panel_row, grid.blocks_per_panel_col)

        # Build the colored label overlay
        self._refresh_label_overlay()

        # Set current block highlight
        current_block = self.session.current_block()
        self.canvas.set_current_block_highlight(current_block.block_row, current_block.block_col)

        # Apply fit (zoom 1) or magnified view centered on the current block (zoom > 1)
        self.canvas.apply_zoom_view()

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

    def _on_block_clicked(self, block_row: int, block_col: int):
        """Click handling depends on the active view mode."""
        if not self.session:
            return

        if self.view_mode == "overview":
            # Cell = panel → open it (skip hidden empty panels)
            panel_idx = block_row * self.session.grid.panels_across + block_col
            if panel_idx < self.session.grid.num_panels and panel_idx not in self.empty_panels:
                self._enter_panel(panel_idx)
            return

        if self.view_mode == "multi":
            # Cell = block in the region → move the cursor there (stay zoomed out)
            grid = self.session.grid
            pr0, pc0 = self._multi_origin
            idx = self._global_block_to_idx(
                pr0 * grid.blocks_per_panel_row + block_row,
                pc0 * grid.blocks_per_panel_col + block_col,
            )
            if idx is not None:
                self.session.move_to_block(idx)
                self.current_panel_idx = self.session.current_block().panel_idx
                self.canvas.set_current_block_highlight(block_row, block_col)
                self._update_preview()
            return

        # Single-panel mode: navigate to the clicked block
        self._clear_selection()  # a plain click cancels any pending marquee selection
        grid = self.session.grid
        panel_blocks = grid.get_panel_blocks(self.current_panel_idx)
        local_idx = block_row * grid.blocks_per_panel_col + block_col
        if local_idx < len(panel_blocks):
            block_idx = self.current_panel_idx * grid.blocks_per_panel + local_idx
            self.session.move_to_block(block_idx)
            # Same panel image — just move highlight + preview (no raster reload)
            self._on_cursor_changed()

    def _on_panel_selected(self, panel_idx: int):
        """Handle history panel click: move the cursor into that panel and show it.

        This is a review/jump action (e.g. going back to fix a panel) — it does
        NOT NA-fill; only the explicit Next Panel action does that.
        """
        if not self.session:
            return
        # Clicking a panel in the history always opens it in single-panel mode
        self._enter_panel(panel_idx)

    # ------------------------------------------------------------------ #
    # Region labeling: drag-paint brush + Shift+click rectangle
    # ------------------------------------------------------------------ #

    def _class_colors(self) -> dict[int, str]:
        """class_id → hex color, including abstain/nodata."""
        colors = {
            cid: c.color for cid, c in self.classes_scheme.classes.items()
        }
        colors[self.classes_scheme.abstain.id] = self.classes_scheme.abstain.color
        colors[self.classes_scheme.nodata.id] = self.classes_scheme.nodata.color
        return colors

    def _refresh_label_overlay(self) -> None:
        """Rebuild only the colored/heatmap block overlay (no raster re-read)."""
        self._sync_legend_selection()
        if not self.session:
            return
        grid = self.session.grid

        if self.display_layer == "uncertainty":
            values = np.full(
                (grid.blocks_per_panel_row, grid.blocks_per_panel_col), np.nan, dtype=np.float32
            )
            for block in grid.get_panel_blocks(self.current_panel_idx):
                score = self.block_uncertainty.get(block.block_id)
                if score is not None:
                    values[block.block_row, block.block_col] = score
            self.canvas.set_scalar_overlay(values)
            return

        panel_blocks = grid.get_panel_blocks(self.current_panel_idx)
        if self.prediction_render_mode == "pixelwise" and any(
            b.block_id in self.block_pixel_predictions for b in panel_blocks
        ):
            pixel_ids = self._build_pixel_class_array(panel_blocks)
            self.canvas.set_pixel_class_overlay(pixel_ids, self._class_colors())
            return

        block_data = np.full(
            (grid.blocks_per_panel_row, grid.blocks_per_panel_col), -3, dtype=np.int16
        )
        for block in panel_blocks:
            record = self.session.labels.get_record(block.block_id)
            block_data[block.block_row, block.block_col] = record.class_id
        self.canvas.set_label_overlay(block_data, self._class_colors())

    def _build_pixel_class_array(self, panel_blocks: list) -> np.ndarray:
        """Native-resolution (panel h x panel w) array of per-pixel predicted
        class ids for the current panel, stitched from block_pixel_predictions'
        cached per-block crops. A block with no cached crop (not yet predicted,
        or skipped as nodata) is left at -1 (transparent when rendered) --
        callers only take this path once at least one block in the panel has
        one; a fully-missing panel falls back to the blockwise render instead.
        """
        grid = self.session.grid
        _, _, panel_w, panel_h = grid.get_panel_coords(self.current_panel_idx)
        pixel_ids = np.full((panel_h, panel_w), -1, dtype=np.int16)
        for block in panel_blocks:
            crop = self.block_pixel_predictions.get(block.block_id)
            if crop is None:
                continue
            y0 = block.block_row * grid.block_size
            x0 = block.block_col * grid.block_size
            y1 = min(y0 + crop.shape[0], panel_h)
            x1 = min(x0 + crop.shape[1], panel_w)
            pixel_ids[y0:y1, x0:x1] = crop[: y1 - y0, : x1 - x0]
        return pixel_ids

    def _panel_block_map(self) -> dict:
        """(block_row, block_col) → BlockInfo for the current panel."""
        return {
            (b.block_row, b.block_col): b
            for b in self.session.grid.get_panel_blocks(self.current_panel_idx)
        }

    def _hotkey_class_id(self, event: QKeyEvent) -> Optional[int]:
        """Resolve a key event to any class id (user classes or abstain), or None."""
        if not self.classes_scheme:
            return None
        text = event.text()
        if event.key() == Qt.Key.Key_Space or text == " ":
            text = "space"
        return self.classes_scheme.hotkey_to_id.get(text)

    def _class_id_for_event(self, event: QKeyEvent) -> Optional[int]:
        """Resolve a key event to a user class id (>=0), or None (for the paint brush)."""
        cid = self._hotkey_class_id(event)
        return cid if (cid is not None and cid >= 0) else None

    def _on_block_paint(self, block_row: int, block_col: int, is_start: bool) -> None:
        """Drag-paint: label the block under the cursor with the held class brush."""
        if self.held_class_id is None or not self.session or self.view_mode != "panel":
            return
        block = self._panel_block_map().get((block_row, block_col))
        if block is None or block.w_px <= 0 or block.h_px <= 0:
            return
        # snapshot only on the first cell so the whole stroke is one undo step
        self.session.labels.assign(
            block.block_id,
            self.held_class_id,
            self.classes_scheme.get_name(self.held_class_id),
            snapshot=is_start,
        )
        self.session.mark_changed()
        self._refresh_label_overlay()

    def _on_block_paint_end(self) -> None:
        """End of a drag-paint stroke: refresh history and autosave if completed."""
        if self.held_class_id is None or not self.session:
            return
        self._refresh_history()
        self._maybe_save_completed_panel(self.current_panel_idx)
        self._maybe_autosave()

    def _on_selection_made(self, r0: int, c0: int, r1: int, c1: int) -> None:
        """A Shift+drag marquee was completed — remember it and await a class key."""
        if self.view_mode != "panel":
            self.canvas.clear_selection()
            return
        self.selection_rect = (r0, c0, r1, c1)
        n = (r1 - r0 + 1) * (c1 - c0 + 1)
        self.status_label.setText(
            f"Selected {n} blocks — press a class key (q/w/e) to fill, or Esc to cancel"
        )

    def _clear_selection(self) -> None:
        """Drop the marquee selection and remove its highlight."""
        if self.selection_rect is not None:
            self.selection_rect = None
            self.canvas.clear_selection()

    def _fill_selection(self, class_id: int) -> None:
        """Fill every block in the marquee selection with the given class."""
        if self.selection_rect is None or not self.session:
            return
        r0, c0, r1, c1 = self.selection_rect
        bmap = self._panel_block_map()
        ids = [
            bmap[(r, c)].block_id
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if (r, c) in bmap and bmap[(r, c)].w_px > 0 and bmap[(r, c)].h_px > 0
        ]
        name = self.classes_scheme.get_name(class_id)
        if ids:
            count = self.session.labels.bulk_assign(ids, class_id, name)
            self.session.mark_changed(count)
        if class_id >= 0:
            self.last_class_id = class_id
        self._clear_selection()
        self._refresh_label_overlay()
        self._refresh_history()
        self._maybe_save_completed_panel(self.current_panel_idx)
        self._maybe_autosave()
        self.status_label.setText(f"Filled {len(ids)} blocks with {name}")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle keyboard input."""
        # Ignore input while the loading overlay is up (except letting Esc through)
        if self.loading_overlay is not None:
            super().keyPressEvent(event)
            return

        # Esc returns from the class summary to the map before anything else
        # gets a chance at the key (Esc is also the overview's "go back").
        if self._showing_summary:
            if event.key() == Qt.Key.Key_Escape:
                self._close_class_summary()
            return

        if not event.isAutoRepeat():
            # Zoom ladder: in-panel magnify ↔ multi-panel zoom-out ↔ overview
            if event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
                self._zoom_in()
                return
            if event.key() in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
                self._zoom_out()
                return
            # 'O' toggles the whole-image overview
            if event.key() == Qt.Key.Key_O:
                self._toggle_overview()
                return
            # Esc leaves a zoomed-out view back to the single panel
            if event.key() == Qt.Key.Key_Escape and self.view_mode != "panel":
                self._set_view("panel")
                return

        # Overview is navigation-only: ignore labeling/navigation keys
        if self.view_mode == "overview":
            super().keyPressEvent(event)
            return

        # If a marquee selection is active, a class key fills it (Esc cancels it)
        if self.selection_rect is not None and not event.isAutoRepeat():
            if event.key() == Qt.Key.Key_Escape:
                self._clear_selection()
                self.status_label.setText("Selection cancelled")
                return
            cid = self._hotkey_class_id(event)
            if cid is not None:
                self._fill_selection(cid)
                return

        # Track the held class key so it can act as a drag-paint brush
        if not event.isAutoRepeat():
            cid = self._class_id_for_event(event)
            if cid is not None:
                self.held_class_id = cid
                self.last_class_id = cid

        if self.controller and self.controller.handle_key_press(event):
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        """Release the drag-paint brush when its class key is let go."""
        if not event.isAutoRepeat():
            cid = self._class_id_for_event(event)
            if cid is not None and cid == self.held_class_id:
                self.held_class_id = None
        super().keyReleaseEvent(event)

    def resizeEvent(self, event) -> None:
        """Keep the loading overlay covering the whole window on resize."""
        super().resizeEvent(event)
        if self.loading_overlay is not None:
            self.loading_overlay.setGeometry(self.rect())

    def _on_labels_changed(self) -> None:
        """Callback: labels have changed, refresh the active view."""
        prev_panel = self.current_panel_idx  # panel the just-labeled block was in
        self._refresh_view()  # panel mode re-syncs current_panel_idx to the cursor
        self._refresh_history()
        # If that panel is now fully done (e.g. last block labeled), autosave it
        self._maybe_save_completed_panel(prev_panel)
        self._maybe_autosave()

    def _on_panel_changed_kb(self) -> None:
        """Callback: cursor rolled into another panel (auto-advance / PageUp). Just redraw."""
        self._refresh_view()
        self._refresh_history()

    def _on_cursor_changed(self) -> None:
        """Callback: cursor moved (no label change)."""
        if not self.session:
            return

        # In multi mode the highlight is region-relative and the view follows the cursor
        if self.view_mode == "multi":
            self._render_multi(self.multi_span)
            return
        if self.view_mode == "overview":
            return

        current_block = self.session.current_block()
        if current_block.panel_idx != self.current_panel_idx:
            self._refresh_view()
            self._refresh_history()
            return
        self.canvas.set_current_block_highlight(current_block.block_row, current_block.block_col)
        # Keep the current block in view when zoomed in
        self.canvas.recenter_current()
        self._update_preview()

    # Detail buffer cap when zoomed in (px). ~near-native for a 4096 panel,
    # ~10 MB — stays constant no matter how far you zoom (magnification is view-only).
    DETAIL_BUFFER = 3200

    def _set_zoom(self, zoom: int) -> None:
        """Set panel zoom (1,2,4,8,16,32). Buffer is capped, so memory stays constant."""
        zoom = max(1, min(32, zoom))
        if not self.session or zoom == self.canvas.zoom:
            return

        old_buffer = self.canvas.canvas_width
        # Zoom 1 uses the fast 1600 buffer; any zoom-in uses one capped detail buffer
        new_buffer = (
            self.canvas.base_size
            if zoom == 1
            else min(self.session.grid.panel_size, self.DETAIL_BUFFER)
        )

        self.canvas.zoom = zoom
        if new_buffer != old_buffer:
            # Buffer resolution changed (crossing 1 ↔ zoomed) — re-read the panel once
            self.canvas.canvas_width = new_buffer
            self.canvas.canvas_height = new_buffer
            self._load_current_panel()
        else:
            # Same buffer (e.g. 2→4→8…) — just re-scale the view, no re-read
            self.canvas.apply_zoom_view()

        self.status_label.setText(f"Zoom {zoom}×")

    # ------------------------------------------------------------------ #
    # View modes: single panel / multi-panel zoom-out / whole-image overview
    # ------------------------------------------------------------------ #

    DONE_PANEL_TINT = -100  # pseudo class id used to tint completed panels in overview

    def _refresh_view(self) -> None:
        """Re-render whatever view is active (panel / multi / overview)."""
        if not self.session:
            return
        if self.view_mode == "overview":
            self._render_overview()
        elif self.view_mode == "multi":
            self._render_multi(self.multi_span)
        else:
            self._load_current_panel()

    def _set_view(self, mode: str, span: int = 2) -> None:
        """Switch view mode and re-render. Resets zoom + selection."""
        if not self.session:
            return
        self.view_mode = mode
        self.canvas.zoom = 1
        self._clear_selection()
        self.overview_button.setChecked(mode == "overview")
        if mode == "panel":
            self.canvas.canvas_width = self.canvas.base_size
            self.canvas.canvas_height = self.canvas.base_size
        elif mode == "multi":
            self.multi_span = span
        self._refresh_view()

    def _toggle_overview(self) -> None:
        """Overview button / 'O': toggle whole-image overview vs single panel."""
        self._set_view("panel" if self.view_mode == "overview" else "overview")

    def _enter_panel(self, panel_idx: int) -> None:
        """Jump into a panel in single-panel mode (from overview/multi click)."""
        self.session.move_to_panel(panel_idx)
        self._set_view("panel")
        self._refresh_history()

    def _zoom_in(self) -> None:
        """'+': magnify within a panel, or step the zoom-out ladder back in."""
        if self.view_mode == "overview":
            self._set_view("multi", span=4)
        elif self.view_mode == "multi":
            self._set_view("multi", span=2) if self.multi_span > 2 else self._set_view("panel")
        else:
            self._set_zoom(self.canvas.zoom * 2)

    def _zoom_out(self) -> None:
        """'-': zoom out within a panel, then out to multi-panel, then overview."""
        if self.view_mode == "panel":
            if self.canvas.zoom > 1:
                self._set_zoom(self.canvas.zoom // 2)
            else:
                self._set_view("multi", span=2)
        elif self.view_mode == "multi":
            self._set_view("multi", span=4) if self.multi_span < 4 else self._set_view("overview")
        # overview is the maximum zoom-out

    def _render_overview(self) -> None:
        """Whole-image overview: panels as cells, current panel highlighted.

        Labeling mode: panels tinted green once fully labeled -- a progress
        indicator, meaningful because panels start empty and fill in over a
        session. Predictions mode has no such progress (every panel is fully
        predicted the instant inference finishes, so a "done" tint would just
        paint the whole overview one color) -- it shows each panel's majority
        class instead (or mean uncertainty, matching whichever the single-panel
        view is currently showing).
        """
        grid = self.session.grid
        self._sync_workspace()
        self.view_label.setText(f"OBSERVATION OVERVIEW   ·   {grid.num_panels} PANELS")
        pa, pd = grid.panels_across, grid.panels_down
        cell = 200
        self.canvas.zoom = 1
        self.canvas.canvas_width = pa * cell
        self.canvas.canvas_height = pd * cell

        img = self.session.raster.read_window_padded(
            0, 0, pa * grid.panel_size, pd * grid.panel_size,
            self.canvas.canvas_width, self.canvas.canvas_height,
        )
        self.canvas.set_panel_image(img, stretch_percentiles=(1, 99))
        self.canvas.set_grid(pa, pd)

        if self.predictions_mode:
            self._render_overview_predictions(pa, pd)
        else:
            cell_data = np.full((pd, pa), -3, dtype=np.int16)
            for p in range(grid.num_panels):
                pr, pc = divmod(p, pa)
                if self._panel_complete(p):
                    cell_data[pr, pc] = self.DONE_PANEL_TINT
            self.canvas.set_label_overlay(cell_data, {self.DONE_PANEL_TINT: "#4CAF50"})

        cur_pr, cur_pc = divmod(self.current_panel_idx, pa)
        self.canvas.set_current_block_highlight(cur_pr, cur_pc)
        self.canvas.apply_zoom_view()
        self.status_label.setText("Overview — click a panel to open it, O or Esc to go back")

    def _render_overview_predictions(self, panels_across: int, panels_down: int) -> None:
        """Predictions-mode overview overlay: per-panel majority class, or per-panel
        mean uncertainty when the Uncertainty Heatmap layer is toggled on."""
        grid = self.session.grid

        if self.display_layer == "uncertainty":
            values = np.full((panels_down, panels_across), np.nan, dtype=np.float32)
            for p in range(grid.num_panels):
                pr, pc = divmod(p, panels_across)
                scores = [
                    self.block_uncertainty[b.block_id]
                    for b in grid.get_panel_blocks(p)
                    if b.block_id in self.block_uncertainty
                ]
                if scores:
                    values[pr, pc] = sum(scores) / len(scores)
            self.canvas.set_scalar_overlay(values)
            return

        cell_data = np.full((panels_down, panels_across), -3, dtype=np.int16)
        for p in range(grid.num_panels):
            pr, pc = divmod(p, panels_across)
            class_ids = [
                self.session.labels.get_record(b.block_id).class_id
                for b in grid.get_panel_blocks(p)
                if self.session.labels.get_record(b.block_id).status not in ("unlabeled", "nodata")
            ]
            if class_ids:
                cell_data[pr, pc] = Counter(class_ids).most_common(1)[0][0]
        self.canvas.set_label_overlay(cell_data, self._class_colors())

    def _render_multi(self, span: int) -> None:
        """Zoom-out showing span×span panels around the current one (black-padded)."""
        grid = self.session.grid
        pa, pd = grid.panels_across, grid.panels_down
        bppr, bppc = grid.blocks_per_panel_row, grid.blocks_per_panel_col
        self.current_panel_idx = self.session.current_block().panel_idx
        self._sync_workspace()
        self.view_label.setText(f"REGIONAL VIEW   ·   {span} × {span} PANELS")
        cur_pr, cur_pc = divmod(self.current_panel_idx, pa)

        # Span-aligned region origin (in panels) containing the current panel
        pr0 = (cur_pr // span) * span
        pc0 = (cur_pc // span) * span
        self._multi_origin = (pr0, pc0)

        buf = self.canvas.base_size
        self.canvas.zoom = 1
        self.canvas.canvas_width = buf
        self.canvas.canvas_height = buf
        img = self.session.raster.read_window_padded(
            pc0 * grid.panel_size, pr0 * grid.panel_size,
            span * grid.panel_size, span * grid.panel_size, buf, buf,
        )
        self.canvas.set_panel_image(img, stretch_percentiles=(1, 99))

        rows, cols = span * bppr, span * bppc
        self.canvas.set_grid(cols, rows)

        cell_data = np.full((rows, cols), -3, dtype=np.int16)
        for r in range(rows):
            for c in range(cols):
                idx = self._global_block_to_idx(pr0 * bppr + r, pc0 * bppc + c)
                if idx is not None:
                    block = grid.get_block(idx)
                    cell_data[r, c] = self.session.labels.get_record(block.block_id).class_id
        self.canvas.set_label_overlay(cell_data, self._class_colors())

        cb = self.session.current_block()
        self.canvas.set_current_block_highlight(
            cur_pr * bppr + cb.block_row - pr0 * bppr,
            cur_pc * bppc + cb.block_col - pc0 * bppc,
        )
        self.canvas.apply_zoom_view()
        self._update_preview()
        self.status_label.setText(
            f"Zoom out {span}×{span} panels — click a block, label with q/w/e, +/- to zoom"
        )

    def _global_block_to_idx(self, global_block_row: int, global_block_col: int):
        """Map a global (block_row, block_col) across the whole image to a block index."""
        grid = self.session.grid
        pa, pd = grid.panels_across, grid.panels_down
        bppr, bppc = grid.blocks_per_panel_row, grid.blocks_per_panel_col
        if global_block_row < 0 or global_block_col < 0:
            return None
        if global_block_row >= pd * bppr or global_block_col >= pa * bppc:
            return None
        panel_row, lbr = divmod(global_block_row, bppr)
        panel_col, lbc = divmod(global_block_col, bppc)
        panel_idx = panel_row * pa + panel_col
        if panel_idx >= grid.num_panels:
            return None
        idx = panel_idx * grid.blocks_per_panel + lbr * bppc + lbc
        return idx if idx < grid.num_blocks() else None

    def _update_preview(self) -> None:
        """Refresh the side preview from the current block."""
        self._sync_legend_selection()
        if not self.session:
            return
        cb = self.session.current_block()
        data = self.session.raster.read_window(
            cb.x_px, cb.y_px, cb.w_px, cb.h_px, cb.w_px, cb.h_px
        )
        self.preview.set_block_image(data, cb.block_id)

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
        if self.predictions_mode or not self.session or not self.controller:
            return

        if self.controller.should_autosave():
            self._do_autosave()

    def _do_autosave(self) -> None:
        """Perform autosave."""
        if self.predictions_mode or not self.session:
            return

        try:
            self.session.save_session(self.labels_dir)
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

    def _autosave_session(self, note: str = "") -> bool:
        """Save the session parquet + cursor JSON."""
        if not self.session or self.predictions_mode:
            return True
        try:
            self.session.save_session(self.labels_dir)
            if self.controller:
                self.controller.reset_autosave()
            msg = "Saved" if not note else f"Saved ({note})"
            self.status_label.setText(msg)
            return True
        except Exception as e:
            self.status_label.setText(f"Save error: {str(e)}")
            return False

    def closeEvent(self, event) -> None:
        """Persist the last edits/cursor before releasing the observation."""
        if not self._autosave_session(note="on close"):
            QMessageBox.critical(self, "Could not save labels", self.status_label.text())
            event.ignore()
            return
        if self.autosave_timer:
            self.autosave_timer.stop()
        if self.session:
            self.session.raster.close()
        event.accept()

    def _go_to_next_panel(self) -> None:
        """Button/handler: finalize current panel (NA-fill + save), then advance.

        Skips over fully off-swath (empty) panels.
        """
        if not self.session:
            return

        leaving = self.current_panel_idx
        self._finalize_panel(leaving)

        nxt = self._next_nonempty_panel(leaving)
        if nxt is None:
            self.status_label.setText("No more panels with image content")
            return

        self.session.move_to_panel(nxt)
        self.current_panel_idx = nxt
        self._load_current_panel()
        self._refresh_history()

    # ---- empty (off-swath) panel handling -----------------------------

    def _compute_empty_panels(self) -> set[int]:
        """Panels where every block is essentially all no-data (off-swath)."""
        empty: set[int] = set()
        if not self.skip_decisions:
            return empty
        for panel_idx in range(self.session.grid.num_panels):
            blocks = self.session.grid.get_panel_blocks(panel_idx)
            if blocks and all(
                self.skip_decisions.get(b.block_id, {}).get("nodata_fraction", 0.0) >= 0.95
                for b in blocks
            ):
                empty.add(panel_idx)
        return empty

    def _retire_empty_panels(self) -> None:
        """Mark every block in empty panels as nodata so they're done + skippable."""
        if not self.empty_panels:
            return
        ids = [
            b.block_id
            for panel_idx in self.empty_panels
            for b in self.session.grid.get_panel_blocks(panel_idx)
        ]
        if ids:
            self.session.labels.set_nodata_bulk(ids)
            self.saved_complete_panels.update(self.empty_panels)

    def _first_nonempty_panel(self) -> Optional[int]:
        for p in range(self.session.grid.num_panels):
            if p not in self.empty_panels:
                return p
        return None

    def _next_nonempty_panel(self, after: int) -> Optional[int]:
        p = after + 1
        n = self.session.grid.num_panels
        while p < n and p in self.empty_panels:
            p += 1
        return p if p < n else None

    def _refresh_history(self) -> None:
        """Refresh the history panel's progress bars, done markers, and current-panel highlight."""
        if isinstance(self.history_panel, HistoryPanel):
            self.history_panel.set_current_panel(self.current_panel_idx)
            self.history_panel.refresh()

    def _export(self) -> None:
        """Export labels to exports/<obs_id>/ (coarse GeoTIFF + parquet + classes)."""
        if not self.session:
            return
        try:
            obs_id = self.session.grid.obs_id
            # The exported Parquet must carry the same source identity as saves.
            self.session.raster.validate_fingerprint(self.session.labels.metadata.get("source"))
            self.session.labels.metadata["source"] = self.session.raster.fingerprint()
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
