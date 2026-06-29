"""History panel: list of panels with completion progress."""

from typing import Optional, Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QScrollArea,
    QFrame,
    QGridLayout,
)
from PySide6.QtGui import QProgressBar

from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import LabelStore


class HistoryPanel(QWidget):
    """Shows all panels with completion progress."""

    def __init__(self, grid: Grid, label_store: LabelStore, parent=None):
        super().__init__(parent)
        self.grid = grid
        self.label_store = label_store
        self.on_panel_selected: Optional[Callable[[int], None]] = None

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

        # Add panel items
        for panel_idx in range(grid.num_panels):
            item = self._create_panel_item(panel_idx)
            list_layout.addWidget(item)

        list_layout.addStretch()
        scroll.setWidget(list_widget)
        layout.addWidget(scroll)

    def _create_panel_item(self, panel_idx: int) -> QWidget:
        """Create a visual item for a panel."""
        frame = QFrame()
        frame.setCursor(Qt.CursorShape.PointingHandCursor)
        frame.setStyleSheet(
            "QFrame { background-color: #2a2a2a; border: 1px solid #444; border-radius: 2px; padding: 4px; }"
        )
        layout = QGridLayout()
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)
        frame.setLayout(layout)

        # Panel label
        panel_row, panel_col = divmod(panel_idx, self.grid.panels_across)
        label = QLabel(f"Panel ({panel_row}, {panel_col})")
        label.setStyleSheet("font-weight: bold;")
        layout.addWidget(label, 0, 0, 1, 2)

        # Progress bar
        total_blocks = self.grid.blocks_per_panel
        labeled_count = sum(
            1 for block in self.grid.get_panel_blocks(panel_idx)
            if self.label_store.get_record(block.block_id).status in ("labeled", "abstain", "nodata")
        )

        progress = QProgressBar()
        progress.setMaximum(total_blocks)
        progress.setValue(labeled_count)
        progress.setTextVisible(True)
        progress.setFormat(f"{labeled_count}/{total_blocks}")
        layout.addWidget(progress, 1, 0, 1, 2)

        # Click handler
        def on_click():
            if self.on_panel_selected:
                self.on_panel_selected(panel_idx)

        frame.mousePressEvent = lambda event: on_click()

        return frame
