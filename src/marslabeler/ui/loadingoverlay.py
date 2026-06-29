"""Loading overlay: blocks interaction during initialization."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PySide6.QtGui import QFont, QColor


class LoadingOverlay(QWidget):
    """Full-window overlay shown during loading to prevent interaction."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0.7);")
        self.setCursor(Qt.CursorShape.WaitCursor)

        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Title
        title = QLabel("Loading Observation")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: white;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Status text
        self.status_label = QLabel("Preparing...")
        self.status_label.setStyleSheet("color: #cccccc; font-size: 12px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                background-color: #222;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #4C72B0;
                border-radius: 3px;
            }
            """
        )
        self.progress_bar.setMaximumWidth(300)
        layout.addWidget(self.progress_bar, alignment=Qt.AlignmentFlag.AlignCenter)

        container = QWidget()
        container.setLayout(layout)
        container.setStyleSheet("background-color: transparent;")

        main_layout = QVBoxLayout()
        main_layout.addStretch()
        main_layout.addWidget(container, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addStretch()
        self.setLayout(main_layout)

    def set_progress(self, value: int) -> None:
        """Update progress bar (0-100)."""
        self.progress_bar.setValue(value)

    def set_status(self, text: str) -> None:
        """Update status text."""
        self.status_label.setText(text)

    def mousePressEvent(self, event):
        """Block all mouse clicks."""
        event.ignore()

    def keyPressEvent(self, event):
        """Block all key presses except Escape."""
        if event.key() == Qt.Key.Key_Escape:
            super().keyPressEvent(event)
        else:
            event.ignore()
