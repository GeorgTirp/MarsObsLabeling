"""Help dialog: keyboard shortcuts and usage guide."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QScrollArea, QWidget
from PySide6.QtGui import QFont

from marslabeler.classes import ClassScheme


class HelpDialog(QDialog):
    """Modal dialog showing keyboard shortcuts and usage."""

    def __init__(self, classes_scheme: ClassScheme):
        super().__init__()
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
        for cls_id, cls_obj in classes_scheme.classes.items():
            help_text += f"  <b>{cls_obj.hotkey}</b> → {cls_obj.name}<br>"

        help_text += (
            f"  <b>{classes_scheme.abstain.hotkey}</b> → Abstain (skip this block)<br>"
            f"  <b>Backspace</b> → Clear current block<br>"
            f"  <b>Ctrl+Z</b> → Undo<br>"
            f"  <b>Shift+Ctrl+Z</b> → Redo<br><br>"
            f"<b>NAVIGATION</b><br>"
            f"  <b>← / →</b> → Move left/right<br>"
            f"  <b>↑ / ↓</b> → Move up/down<br>"
            f"  <b>Page Up/Down</b> → Previous/next panel<br>"
            f"  <b>Home</b> → First block in panel<br><br>"
            f"<b>AUTO-ADVANCE</b><br>"
            f"  After labeling, cursor moves to the next unlabeled block automatically.<br><br>"
            f"<b>TIPS</b><br>"
            f"  • All unseen blocks have unlabeled status<br>"
            f"  • Completed panels show in the history sidebar<br>"
            f"  • Session auto-saves every 25 labels<br>"
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

        # Close button (just click anywhere or press Escape)
        close_hint = QLabel("<i>Press Escape or click outside to close</i>")
        close_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(close_hint)

        self.setLayout(layout)
