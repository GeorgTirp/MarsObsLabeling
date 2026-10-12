"""Help dialog: keyboard shortcuts and usage guide."""

from html import escape

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QScrollArea, QWidget, QDialogButtonBox
from PySide6.QtGui import QFont

from marslabeler.classes import ClassScheme


class HelpDialog(QDialog):
    """Modal dialog showing keyboard shortcuts and usage."""

    def __init__(self, classes_scheme: ClassScheme, parent=None, on_customize=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts & Help")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(600)

        layout = QVBoxLayout()

        # Title
        title = QLabel("Mars Obs Labeler — Keyboard Guide")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Scrollable help text
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        help_widget = QWidget()
        help_layout = QVBoxLayout()

        # Build help text
        help_text = "<b>LABELING</b><br>"
        for cls_obj in [*classes_scheme.classes.values(), classes_scheme.abstain]:
            help_text += f"  <b>{escape(cls_obj.hotkey or 'Legend only')}</b> → {escape(cls_obj.name)}<br>"

        help_text += (
            f"  <b>Backspace / Delete</b> → Clear current block or selected region<br>"
            f"  <b>Ctrl+Z</b> → Undo<br>"
            f"  <b>Shift+Ctrl+Z</b> → Redo<br><br>"
            f"<b>NAVIGATION</b><br>"
            f"  <b>← / →</b> → Move left/right<br>"
            f"  <b>↑ / ↓</b> → Move up/down<br>"
            f"  <b>Page Up/Down</b> → Previous/next panel<br>"
            f"  <b>Home</b> → First block in panel<br>"
            f"  <b>+ / −</b> → Zoom in/out: 32×…1× → 2×2 panels → 4×4 → whole image<br>"
            f"  <b>O</b> → Toggle whole-image overview (click a panel to open it)<br>"
            f"  <b>F11 / Focus block</b> → Enlarge the block preview with class keys still active;<br>"
            f"    press again to restore the map and panels<br>"
            f"  <b>Esc</b> → Leave a zoomed-out view back to the single panel<br><br>"
            f"<b>AUTO-ADVANCE</b><br>"
            f"  After labeling, cursor moves to the next unlabeled block automatically.<br>"
            f"  Opening a panel starts at its first unlabeled block.<br><br>"
            f"<b>LABEL REGIONS (mouse)</b><br>"
            f"  <b>Click</b> a block → select it<br>"
            f"  <b>Hold a class key and select blocks</b> → click or drag with the left mouse button<br>"
            f"  <b>Erase</b> toolbar button → drag over blocks to remove labels<br>"
            f"    (Ctrl+Z undoes the whole stroke)<br>"
            f"  <b>Shift + drag</b> left mouse → draw a yellow box over an area,<br>"
            f"    then press a class key to fill, or Backspace / Delete to erase<br>"
            f"    (Esc cancels the selection)<br><br>"
            f"<b>TIPS</b><br>"
            f"  • All unseen blocks have unlabeled status<br>"
            f"  • Completed panels show in the history sidebar<br>"
            f"  • Observation progress counts terrain labels; skipped blocks have no label<br>"
            f"  • <b>Edit → Customize class keys</b> saves your preferred keys locally<br>"
            f"  • Session auto-saves at the configured label/time intervals<br>"
            f"  • Press <b>?</b> anytime to show this help<br>"
        )

        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_layout.addWidget(help_label)
        help_layout.addStretch()

        help_widget.setLayout(help_layout)
        scroll.setWidget(help_widget)
        layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        if on_customize is not None:
            customize = buttons.addButton("Customize class keys…", QDialogButtonBox.ButtonRole.ActionRole)
            def open_customization():
                self.accept()
                on_customize()
            customize.clicked.connect(open_customization)
        layout.addWidget(buttons)

        self.setLayout(layout)
