"""History panel: list of panels with completion progress and done markers."""

from typing import Optional, Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
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
        self.setObjectName("historyPanel")
        self.grid = grid
        self.label_store = label_store
        self.hidden_panels = set(hidden_panels or ())  # fully empty panels to omit
        self.on_panel_selected: Optional[Callable[[int], None]] = None
        self.current_panel_idx: Optional[int] = None  # panel open in the canvas right now

        # Stored references so we can refresh in place (no rebuild on every label)
        self.panel_frames: dict[int, QFrame] = {}
        self.panel_bars: dict[int, QProgressBar] = {}
        self.panel_labels: dict[int, QLabel] = {}
        self.panel_counts: dict[int, QLabel] = {}
        self.panel_statuses: dict[int, QLabel] = {}

        self.setMinimumWidth(140)
        self.setMaximumWidth(220)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(10)
        self.setLayout(layout)

        header = QHBoxLayout()
        title = QLabel("PANELS")
        title.setStyleSheet("color: #91a0b2; font-size: 10px; font-weight: 600;")
        header.addWidget(title)
        header.addStretch()
        count = QLabel(str(sum(i not in self.hidden_panels for i in range(grid.num_panels))))
        count.setStyleSheet("color: #91a0b2; font-size: 10px;")
        header.addWidget(count)
        layout.addLayout(header)

        # Scrollable panel list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        list_widget = QWidget()
        list_layout = QVBoxLayout()
        list_layout.setSpacing(5)
        list_layout.setContentsMargins(0, 0, 0, 0)
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
        frame.setObjectName("panelNavigationRow")
        frame.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QGridLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(9, 9, 9, 9)
        frame.setLayout(layout)

        # Panel label
        panel_row, panel_col = divmod(panel_idx, self.grid.panels_across)
        label = QLabel(f"Panel ({panel_row}, {panel_col})")
        # Let clicks pass through to the frame so the whole card is clickable
        label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(label, 0, 0, 1, 2)

        count = QLabel()
        count.setStyleSheet("color: #91a0b2; font-size: 10px; background: transparent;")
        count.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(count, 1, 0)

        status = QLabel()
        status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        status.setStyleSheet("color: #91a0b2; font-size: 9px; background: transparent;")
        status.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(status, 1, 1)

        # Progress bar
        total_blocks = self.grid.blocks_per_panel
        progress = QProgressBar()
        progress.setMaximum(total_blocks)
        progress.setTextVisible(False)
        progress.setFixedHeight(3)
        progress.setStyleSheet(
            "QProgressBar { background: #2c3947; border: none; border-radius: 1px; }"
            "QProgressBar::chunk { background: #b65d32; border-radius: 1px; }"
        )
        progress.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(progress, 2, 0, 1, 2)

        # Store refs
        self.panel_frames[panel_idx] = frame
        self.panel_bars[panel_idx] = progress
        self.panel_labels[panel_idx] = label
        self.panel_counts[panel_idx] = count
        self.panel_statuses[panel_idx] = status

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
        self.panel_counts[panel_idx].setText(f"{labeled} / {total_blocks}")
        self.panel_statuses[panel_idx].setText(
            "ACTIVE" if is_current else "DONE" if complete else ""
        )

        panel_row, panel_col = divmod(panel_idx, self.grid.panels_across)
        label = self.panel_labels[panel_idx]
        frame = self.panel_frames[panel_idx]
        name = f"Panel ({panel_row}, {panel_col})"

        if complete:
            label.setText(f"✓ {name}")
        else:
            label.setText(name)
        label.setStyleSheet(
            "font-size: 11px; font-weight: 500; background: transparent; border: none; "
            f"color: {'#e6a079' if is_current or complete else '#e6edf3'};"
        )
        frame.setAccessibleName(f"{name}, {labeled} of {total_blocks} blocks complete")

        if is_current:
            # Border color always wins over the done/not-done fill so the active
            # panel stays findable in the list regardless of its completion state.
            border = "1px solid #b65d32"
            bg = "#38281f"
        else:
            border = "1px solid #2c3947"
            bg = "#19222c"
        frame.setStyleSheet(
            f"QFrame#panelNavigationRow {{ background-color: {bg}; border: {border}; "
            "border-radius: 4px; }"
            "QFrame#panelNavigationRow:hover { border-color: #b65d32; }"
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
        """Mark panel_idx as the one open in the canvas right now."""
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
