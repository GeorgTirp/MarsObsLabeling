"""Side preview: native-resolution view of the current block."""

import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QLabel
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel as QLabelWidget

from marslabeler.ui.render import numpy_to_qimage, apply_display_stretch


class SidePreview(QWidget):
    """Displays the current block at native 1:1 resolution."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumWidth(550)
        self.setMinimumWidth(300)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QLabelWidget("Block Preview (1:1)")
        title.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(title)

        # Image display (scrollable if large)
        self.image_label = QLabelWidget()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1a1a1a; border: 1px solid #444;")
        layout.addWidget(self.image_label)

        # Info label
        self.info_label = QLabelWidget()
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
            self.info_label.setText("(empty)")
            return

        # Apply display stretch
        stretched = apply_display_stretch(block_data, (1, 99))

        # Convert to QImage
        qimage = numpy_to_qimage(stretched)

        # Scale to fit (max 500px, maintain aspect)
        max_size = 480
        if qimage.width() > max_size or qimage.height() > max_size:
            qimage = qimage.scaledToWidth(max_size, Qt.TransformationMode.FastTransformation)

        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)

        # Display info
        info_text = f"{block_id} | {block_data.shape[0]}×{block_data.shape[1]} px"
        self.info_label.setText(info_text)
