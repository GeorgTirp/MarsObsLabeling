"""Terrain legend with restrained color swatches and keyboard shortcuts.

Lives in its own full-height column of MainWindow's splitter (between the
canvas and the right-hand preview/actions column), so the whole class list is
readable at a glance instead of being squeezed into a short scroll box.
"""

from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QFrame,
    QPushButton,
    QSizePolicy,
)

from marslabeler.classes import ClassScheme

# Minimum height of one class block. Enough for a two-line wrapped name (the
# longest NOAH-H DC names run ~50 chars) without the row collapsing to text
# height at narrow column widths.
CLASS_ROW_MIN_HEIGHT = 34


def contrast_text_color(hex_color: str) -> str:
    """Black or white -- whichever has the higher WCAG contrast ratio against
    `hex_color`. Useful when rendering class labels directly over imagery.
    """
    color = QColor(hex_color)

    def _linearize(channel_8bit: int) -> float:
        channel = channel_8bit / 255.0
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    luminance = (
        0.2126 * _linearize(color.red())
        + 0.7152 * _linearize(color.green())
        + 0.0722 * _linearize(color.blue())
    )
    contrast_with_black = (luminance + 0.05) / 0.05
    contrast_with_white = 1.05 / (luminance + 0.05)
    return "#000000" if contrast_with_black >= contrast_with_white else "#ffffff"


class ClassRow(QFrame):
    """One legend entry, clickable to assign that class to the current block.

    Labelling was keyboard-only, which forced a new user to memorise 18 hotkeys
    before they could label anything. Clicking a row now does exactly what its
    hotkey does -- the click is routed through the same controller entry point,
    so undo, auto-advance and panel-change behave identically either way.
    """

    clicked = Signal(int)  # class id

    def __init__(self, class_id: int, parent=None):
        super().__init__(parent)
        self._class_id = class_id
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._class_id)
        super().mousePressEvent(event)


class LegendPanel(QWidget):
    """Displays terrain classes with a swatch, name and hotkey."""

    def __init__(self, classes_scheme: ClassScheme, parent=None):
        super().__init__(parent)
        self.classes_scheme = classes_scheme
        self.current_class_id: Optional[int] = None
        self._class_rows: dict[int, ClassRow] = {}
        self.setObjectName("legendPanel")
        # Wide enough for the longest class name to wrap to ~2 lines; the
        # column is user-resizable (MainWindow's splitter) so this is a floor,
        # not a cap.
        self.setMinimumWidth(230)

        # Set by MainWindow after construction; opens the per-class Summary window
        # (coverage + confidence/uncertainty stats + neural-PCA gallery).
        self.on_summary_clicked: Optional[Callable[[], None]] = None
        # Set by MainWindow; receives the class id of a clicked legend row.
        self.on_class_clicked: Optional[Callable[[int], None]] = None

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(8)
        self.setLayout(layout)

        header = QHBoxLayout()
        title = QLabel("TERRAIN CLASSES")
        title.setStyleSheet("color: #91a0b2; font-size: 10px; font-weight: 600;")
        header.addWidget(title)
        header.addStretch()
        count = QLabel(str(len(self.classes_scheme.classes)))
        count.setStyleSheet("color: #91a0b2; font-size: 10px;")
        header.addWidget(count)
        layout.addLayout(header)

        # Scroll area is an overflow guard for short windows only -- in its own
        # full-height column the whole list normally fits with no scrolling.
        scroll = QScrollArea()
        self.scroll_area = scroll
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        list_widget = QWidget()
        list_layout = QVBoxLayout()
        list_layout.setSpacing(2)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_widget.setLayout(list_layout)

        for class_obj in sorted(self.classes_scheme.classes.values(), key=lambda x: x.id):
            list_layout.addWidget(self._create_class_row(class_obj))
        list_layout.addWidget(self._create_class_row(self.classes_scheme.abstain))
        list_layout.addStretch()

        scroll.setWidget(list_widget)
        layout.addWidget(scroll, 1)

        self.summary_button = QPushButton("Class summary")
        self.summary_button.setMinimumHeight(32)
        self.summary_button.setToolTip("Inspect class coverage, confidence and representative blocks")
        self.summary_button.clicked.connect(self._on_summary_clicked)
        layout.addWidget(self.summary_button)

    def _on_summary_clicked(self) -> None:
        if self.on_summary_clicked:
            self.on_summary_clicked()

    def _emit_class_clicked(self, class_id: int) -> None:
        if self.on_class_clicked:
            self.on_class_clicked(class_id)

    def set_current_class(self, class_id: Optional[int]) -> None:
        """Highlight the current block's assigned class, independent of the brush."""
        if class_id not in self._class_rows:
            class_id = None
        if class_id != self.current_class_id:
            for cid in (self.current_class_id, class_id):
                if cid is None:
                    continue
                row = self._class_rows[cid]
                current = cid == class_id
                row.setProperty("currentClass", current)
                row.setAccessibleDescription("Assigned to the current block" if current else "")
                row.findChild(QLabel, "currentClassMarker").setText("✓" if current else "")
                for widget in (row, *row.findChildren(QLabel)):
                    widget.style().unpolish(widget)
                    widget.style().polish(widget)
                    widget.update()
            self.current_class_id = class_id
        if class_id is not None:
            self.scroll_area.ensureWidgetVisible(self._class_rows[class_id], 0, 8)

    def _create_class_row(self, class_obj) -> QWidget:
        """A neutral row keeps the terrain color distinct from interface state."""
        frame = ClassRow(class_obj.id)
        self._class_rows[class_obj.id] = frame
        frame.setObjectName("terrainClassRow")
        frame.setProperty("currentClass", False)
        frame.setAccessibleName(f"Assign {class_obj.name}")
        frame.clicked.connect(self._emit_class_clicked)
        frame.setMinimumHeight(CLASS_ROW_MIN_HEIGHT)
        frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        frame.setStyleSheet(
            "QFrame#terrainClassRow { background-color: #19222c; "
            "border: 1px solid transparent; border-radius: 4px; }"
            "QFrame#terrainClassRow:hover { background-color: #24333f; "
            "border-color: #b65d32; }"
            'QFrame#terrainClassRow[currentClass="true"] { background-color: #423024; '
            "border-color: #e6a079; }"
            "QLabel#terrainClassName { color: #e6edf3; font-size: 11px; font-weight: 400; "
            "background: transparent; border: none; }"
            'QFrame#terrainClassRow[currentClass="true"] QLabel#terrainClassName { '
            "color: #fff4eb; font-weight: 600; }"
        )
        frame.setToolTip(
            f"Click to assign '{class_obj.name}'"
            + (f" (or press {class_obj.hotkey})" if class_obj.hotkey else "")
        )

        row = QHBoxLayout()
        row.setContentsMargins(7, 5, 6, 5)
        row.setSpacing(8)
        frame.setLayout(row)

        swatch = QFrame()
        swatch.setFixedWidth(5)
        swatch.setMinimumHeight(18)
        swatch.setStyleSheet(
            f"background-color: {class_obj.color}; border: none; border-radius: 2px;"
        )
        swatch.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        row.addWidget(swatch)

        name = QLabel(class_obj.name)
        name.setObjectName("terrainClassName")
        name.setWordWrap(True)
        # Children must not eat the click, or only the row's padding would be
        # clickable -- which reads as "clicking sometimes works".
        name.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        row.addWidget(name, 1)

        marker = QLabel("")
        marker.setObjectName("currentClassMarker")
        marker.setFixedWidth(12)
        marker.setAlignment(Qt.AlignmentFlag.AlignCenter)
        marker.setStyleSheet(
            "color: #f0b993; font-size: 13px; font-weight: 600; "
            "background: transparent; border: none;"
        )
        marker.setToolTip("Assigned to the current block")
        marker.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        row.addWidget(marker)

        if class_obj.hotkey:
            keycap = QLabel(class_obj.hotkey.upper())
            keycap.setAlignment(Qt.AlignmentFlag.AlignCenter)
            keycap.setMinimumWidth(18)
            keycap.setStyleSheet(
                "color: #91a0b2; background-color: #141b22; "
                "border: 1px solid #2c3947; border-radius: 3px; "
                "font-family: monospace; font-size: 9px; font-weight: 400; "
                "padding: 2px 3px;"
            )
            keycap.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            keycap.setToolTip(
                f"Click this row, or press {class_obj.hotkey}, "
                f"to assign '{class_obj.name}'"
            )
            row.addWidget(keycap, 0, Qt.AlignmentFlag.AlignVCenter)

        return frame
