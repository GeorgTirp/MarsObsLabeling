"""Legend panel: class definitions with colors and hotkeys."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap, QIcon
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QScrollArea,
    QGridLayout,
    QFrame,
)

from marslabeler.classes import ClassScheme


class LegendPanel(QWidget):
    """Displays terrain classes with colors, names, and hotkeys."""

    def __init__(self, classes_scheme: ClassScheme, parent=None):
        super().__init__(parent)
        self.classes_scheme = classes_scheme
        self.setMaximumWidth(250)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QLabel("Classes")
        title.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(title)

        # Scrollable class list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #444; }")

        list_widget = QWidget()
        list_layout = QVBoxLayout()
        list_layout.setSpacing(2)
        list_layout.setContentsMargins(4, 4, 4, 4)
        list_widget.setLayout(list_layout)

        # Add user classes
        for class_obj in sorted(self.classes_scheme.classes.values(), key=lambda x: x.id):
            item = self._create_class_item(class_obj)
            list_layout.addWidget(item)

        # Add abstain
        item = self._create_class_item(self.classes_scheme.abstain)
        list_layout.addWidget(item)

        list_layout.addStretch()
        scroll.setWidget(list_widget)
        layout.addWidget(scroll)

    def _create_class_item(self, class_obj) -> QWidget:
        """Create a visual item for a class."""
        frame = QFrame()
        frame.setStyleSheet("QFrame { background-color: #2a2a2a; border: 1px solid #444; border-radius: 2px; }")
        layout = QGridLayout()
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)
        frame.setLayout(layout)

        # Color swatch
        swatch = QLabel()
        pixmap = QPixmap(24, 24)
        pixmap.fill(QColor(class_obj.color))
        swatch.setPixmap(pixmap)
        layout.addWidget(swatch, 0, 0, 2, 1)

        # Class name
        name_label = QLabel(class_obj.name)
        name_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(name_label, 0, 1)

        # Hotkey
        if class_obj.hotkey:
            key_label = QLabel(f"[{class_obj.hotkey}]")
            key_label.setStyleSheet("font-size: 10px; color: #aaa;")
            layout.addWidget(key_label, 1, 1)

        return frame
