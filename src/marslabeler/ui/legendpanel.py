"""Legend panel: class definitions as full-width color blocks with the class
name written directly on the swatch, plus its hotkey.

Lives in its own full-height column of MainWindow's splitter (between the
canvas and the right-hand preview/actions column), so the whole class list is
readable at a glance instead of being squeezed into a short scroll box.
"""

from typing import Callable, Optional

from PySide6.QtCore import Qt
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
CLASS_ROW_MIN_HEIGHT = 38


def contrast_text_color(hex_color: str) -> str:
    """Black or white -- whichever has the higher WCAG contrast ratio against
    `hex_color`. Needed because the class name is drawn *on* the swatch, and
    the 14-class palette spans a wide lightness range (a fixed white would be
    unreadable on the light olives/cyans, a fixed black on the dark blues).
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


class LegendPanel(QWidget):
    """Displays terrain classes as color blocks labeled with name + hotkey."""

    def __init__(self, classes_scheme: ClassScheme, parent=None):
        super().__init__(parent)
        self.classes_scheme = classes_scheme
        # Wide enough for the longest class name to wrap to ~2 lines; the
        # column is user-resizable (MainWindow's splitter) so this is a floor,
        # not a cap.
        self.setMinimumWidth(210)

        # Set by MainWindow after construction; opens the per-class Summary window
        # (coverage + confidence/uncertainty stats + neural-PCA gallery).
        self.on_summary_clicked: Optional[Callable[[], None]] = None

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self.setLayout(layout)

        title = QLabel("Classes")
        title.setStyleSheet("font-weight: bold; padding: 2px;")
        layout.addWidget(title)

        # Scroll area is an overflow guard for short windows only -- in its own
        # full-height column the whole list normally fits with no scrolling.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        list_widget = QWidget()
        list_layout = QVBoxLayout()
        list_layout.setSpacing(3)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_widget.setLayout(list_layout)

        for class_obj in sorted(self.classes_scheme.classes.values(), key=lambda x: x.id):
            list_layout.addWidget(self._create_class_row(class_obj))
        list_layout.addWidget(self._create_class_row(self.classes_scheme.abstain))
        list_layout.addStretch()

        scroll.setWidget(list_widget)
        layout.addWidget(scroll, 1)

        self.summary_button = QPushButton("\U0001F4CA Summary")
        self.summary_button.clicked.connect(self._on_summary_clicked)
        layout.addWidget(self.summary_button)

    def _on_summary_clicked(self) -> None:
        if self.on_summary_clicked:
            self.on_summary_clicked()

    def _create_class_row(self, class_obj) -> QWidget:
        """One class as a color block with its name written on it + hotkey badge."""
        text_color = contrast_text_color(class_obj.color)
        # Translucent shades of the text color, so the keycap reads as an inset
        # on any swatch without introducing a third hue.
        keycap_fill = "rgba(255, 255, 255, 0.25)" if text_color == "#ffffff" else "rgba(0, 0, 0, 0.16)"
        keycap_edge = "rgba(255, 255, 255, 0.65)" if text_color == "#ffffff" else "rgba(0, 0, 0, 0.45)"

        frame = QFrame()
        frame.setMinimumHeight(CLASS_ROW_MIN_HEIGHT)
        frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        frame.setStyleSheet(
            f"QFrame {{ background-color: {class_obj.color}; "
            "border: 1px solid rgba(0, 0, 0, 0.35); border-radius: 4px; }"
        )

        row = QHBoxLayout()
        row.setContentsMargins(8, 5, 6, 5)
        row.setSpacing(6)
        frame.setLayout(row)

        name = QLabel(class_obj.name)
        name.setWordWrap(True)
        name.setStyleSheet(
            f"color: {text_color}; font-size: 11px; font-weight: bold; "
            "background: transparent; border: none;"
        )
        row.addWidget(name, 1)

        if class_obj.hotkey:
            keycap = QLabel(class_obj.hotkey.upper())
            keycap.setAlignment(Qt.AlignmentFlag.AlignCenter)
            keycap.setStyleSheet(
                f"color: {text_color}; background-color: {keycap_fill}; "
                f"border: 1px solid {keycap_edge}; border-radius: 4px; "
                "font-family: monospace; font-size: 11px; font-weight: bold; "
                "padding: 2px 6px;"
            )
            keycap.setToolTip(f"Press {class_obj.hotkey} to assign '{class_obj.name}'")
            row.addWidget(keycap, 0, Qt.AlignmentFlag.AlignVCenter)

        return frame
