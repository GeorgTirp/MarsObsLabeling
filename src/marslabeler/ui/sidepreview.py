"""Side preview: native-resolution view of the current block."""

import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from marslabeler.ui.render import numpy_to_qimage, apply_display_stretch


class SidePreview(QWidget):
    """Displays the current block at native 1:1 resolution."""

    PREVIEW_SIZE = 480  # square preview box, matches square blocks (e.g. 512×512)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumWidth(550)
        self.setMinimumWidth(self.PREVIEW_SIZE + 30)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QLabel("Block Preview (1:1)")
        title.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(title)

        # Image display — fixed square box so the crop reads as a true square
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(self.PREVIEW_SIZE, self.PREVIEW_SIZE)
        self.image_label.setStyleSheet("background-color: #1a1a1a; border: 1px solid #444;")
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Info label
        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-size: 10px; color: #aaa; padding: 4px;")
        layout.addWidget(self.info_label)

        layout.addStretch()

    def set_block_image(self, block_data: np.ndarray, block_id: str = "") -> None:
        """
        Display a block at native resolution.

        Args:
            block_data: Grayscale image array (block at native resolution)
            block_id: Optional block identifier for display
        """
        if block_data.size == 0:
            self.image_label.clear()
            self.image_label.setText("(empty)")
            self.info_label.setText("(empty)")
            return

        # Apply display stretch
        stretched = apply_display_stretch(block_data, (1, 99))

        # Convert to QImage
        qimage = numpy_to_qimage(stretched)

        # Scale to fit the square preview box, keeping aspect ratio.
        # Full blocks (512×512) fill the box; partial edge blocks letterbox cleanly.
        qimage = qimage.scaled(
            self.PREVIEW_SIZE,
            self.PREVIEW_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )

        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)

        # Display info
        info_text = f"{block_id} | {block_data.shape[0]}×{block_data.shape[1]} px"
        self.info_label.setText(info_text)
