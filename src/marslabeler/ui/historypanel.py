"""History panel: list of panels with completion progress and done markers."""

from typing import Optional, Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QScrollArea,
    QFrame,
    QGridLayout,
    QProgressBar,
)

from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import LabelStore

_DONE_STATUSES = ("labeled", "abstain", "nodata")


class HistoryPanel(QWidget):
    """Shows all panels with completion progress and a done/saved marker."""

    def __init__(self, grid: Grid, label_store: LabelStore, parent=None, hidden_panels=None):
        super().__init__(parent)
        self.grid = grid
        self.label_store = label_store
        self.hidden_panels = set(hidden_panels or ())  # fully empty panels to omit
        self.on_panel_selected: Optional[Callable[[int], None]] = None
        self.current_panel_idx: Optional[int] = None  # panel open in the canvas right now

        # Stored references so we can refresh in place (no rebuild on every label)
        self.panel_frames: dict[int, QFrame] = {}
        self.panel_bars: dict[int, QProgressBar] = {}
        self.panel_labels: dict[int, QLabel] = {}

        self.setMaximumWidth(200)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QLabel("Panels")
        title.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(title)

        # Scrollable panel list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #444; }")

        list_widget = QWidget()
        list_layout = QVBoxLayout()
        list_layout.setSpacing(2)
        list_layout.setContentsMargins(4, 4, 4, 4)
        list_widget.setLayout(list_layout)

        # Add panel items (skip fully-empty/no-data panels)
        for panel_idx in range(grid.num_panels):
            if panel_idx in self.hidden_panels:
                continue
            item = self._create_panel_item(panel_idx)
            list_layout.addWidget(item)

        list_layout.addStretch()
        scroll.setWidget(list_widget)
        layout.addWidget(scroll)

    def _is_complete(self, panel_idx: int) -> bool:
        """A panel is complete when no block is still unlabeled."""
        for block in self.grid.get_panel_blocks(panel_idx):
            if self.label_store.get_record(block.block_id).status == "unlabeled":
                return False
        return True

    def _labeled_count(self, panel_idx: int) -> int:
        return sum(
            1
            for block in self.grid.get_panel_blocks(panel_idx)
            if self.label_store.get_record(block.block_id).status in _DONE_STATUSES
        )

    def _create_panel_item(self, panel_idx: int) -> QWidget:
        """Create a visual item for a panel."""
        frame = QFrame()
        frame.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QGridLayout()
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)
        frame.setLayout(layout)

        # Panel label
        panel_row, panel_col = divmod(panel_idx, self.grid.panels_across)
        label = QLabel(f"Panel ({panel_row}, {panel_col})")
        label.setStyleSheet("font-weight: bold;")
        # Let clicks pass through to the frame so the whole card is clickable
        label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(label, 0, 0, 1, 2)

        # Progress bar
        total_blocks = self.grid.blocks_per_panel
        progress = QProgressBar()
        progress.setMaximum(total_blocks)
        progress.setTextVisible(True)
        progress.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(progress, 1, 0, 1, 2)

        # Store refs
        self.panel_frames[panel_idx] = frame
        self.panel_bars[panel_idx] = progress
        self.panel_labels[panel_idx] = label

        # Click handler
        def on_click():
            if self.on_panel_selected:
                self.on_panel_selected(panel_idx)

        frame.mousePressEvent = lambda event: on_click()

        self._apply_item_state(panel_idx)
        return frame

    def _apply_item_state(self, panel_idx: int) -> None:
        """Update progress value, label text, and done/current styling for one panel."""
        total_blocks = self.grid.blocks_per_panel
        labeled = self._labeled_count(panel_idx)
        complete = self._is_complete(panel_idx)
        is_current = panel_idx == self.current_panel_idx

        bar = self.panel_bars[panel_idx]
        bar.setMaximum(total_blocks)
        bar.setValue(labeled)
        bar.setFormat(f"{labeled}/{total_blocks}")

        panel_row, panel_col = divmod(panel_idx, self.grid.panels_across)
        label = self.panel_labels[panel_idx]
        frame = self.panel_frames[panel_idx]
        name = f"Panel ({panel_row}, {panel_col})"

        if complete:
            label.setText(f"✓ {name}")
            label.setStyleSheet("font-weight: bold; color: #7ED957;")
        else:
            label.setText(name)
            label.setStyleSheet("font-weight: bold;")

        if is_current:
            # Border color always wins over the done/not-done fill so the active
            # panel stays findable in the list regardless of its completion state.
            border = "3px solid #FFEB3B"
            bg = "#1f3d1f" if complete else "#3d3a1f"
        elif complete:
            border = "1px solid #4CAF50"
            bg = "#1f3d1f"
        else:
            border = "1px solid #444"
            bg = "#2a2a2a"
        frame.setStyleSheet(
            f"QFrame {{ background-color: {bg}; border: {border}; "
            "border-radius: 2px; padding: 4px; }"
        )

        if is_current and complete:
            frame.setToolTip("Currently viewing (done and saved)")
        elif complete:
            frame.setToolTip("Done and saved")
        elif is_current:
            frame.setToolTip("Currently viewing")
        else:
            frame.setToolTip("")

    def set_current_panel(self, panel_idx: Optional[int]) -> None:
        """Mark panel_idx as the one open in the canvas right now (yellow border)."""
        if panel_idx == self.current_panel_idx:
            return
        previous = self.current_panel_idx
        self.current_panel_idx = panel_idx
        if previous is not None and previous in self.panel_frames:
            self._apply_item_state(previous)
        if panel_idx is not None and panel_idx in self.panel_frames:
            self._apply_item_state(panel_idx)

    def refresh(self) -> None:
        """Recompute progress and done state for all panels (in place)."""
        for panel_idx in self.panel_frames:
            self._apply_item_state(panel_idx)
