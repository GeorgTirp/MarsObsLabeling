"""Responsive inspector for the current block's native image crop."""

import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy

from marslabeler.ui.render import numpy_to_qimage, apply_display_stretch


class SidePreview(QWidget):
    """Displays a native crop scaled to the available inspector width."""

    PREVIEW_SIZE = 480

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("blockInspector")
        self.setMinimumWidth(240)
        self.setMaximumWidth(self.PREVIEW_SIZE)
        self.setStyleSheet("QWidget#blockInspector { background-color: #10161d; }")
        self._source_pixmap: QPixmap | None = None

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(8)
        self.setLayout(layout)

        title = QLabel("BLOCK INSPECTOR")
        title.setStyleSheet(
            "color: #e6edf3; font-size: 11px; font-weight: 600; background: transparent;"
        )
        layout.addWidget(title)

        subtitle = QLabel("Native crop · scaled to fit")
        subtitle.setStyleSheet(
            "color: #91a0b2; font-size: 11px; background: transparent;"
        )
        layout.addWidget(subtitle)

        # Ignore the pixmap's size hint horizontally, so a large crop never
        # prevents the splitter from making this inspector narrower again.
        self.image_label = QLabel("Select a block to inspect")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.image_label.setStyleSheet(
            "background-color: #0c1117; border: 1px solid #2c3947; "
            "border-radius: 4px; color: #91a0b2; font-size: 11px;"
        )
        layout.addWidget(self.image_label)

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setTextFormat(Qt.TextFormat.PlainText)
        self.info_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.info_label.setStyleSheet(
            "font-size: 11px; color: #91a0b2; background: transparent; padding-top: 2px;"
        )
        layout.addWidget(self.info_label)

        layout.addStretch()
        self.resize(self.sizeHint())
        self._resize_preview()

    def sizeHint(self) -> QSize:
        return QSize(288, 410)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._resize_preview()

    def _resize_preview(self) -> None:
        """Fit the cached image without touching the raster or display stretch."""
        margins = self.layout().contentsMargins()
        side = max(1, min(self.PREVIEW_SIZE, self.width() - margins.left() - margins.right()))
        self.image_label.setFixedHeight(side)
        if self._source_pixmap is not None:
            # Leave the image frame visible, and letterbox partial edge crops.
            image_side = max(1, side - 2)
            self.image_label.setPixmap(
                self._source_pixmap.scaled(
                    image_side,
                    image_side,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
            )

    def set_block_image(self, block_data: np.ndarray, block_id: str = "") -> None:
        """
        Display a native block crop, preserving its aspect ratio when scaled.

        Args:
            block_data: Grayscale image array (block at native resolution)
            block_id: Optional block identifier for display
        """
        if block_data.size == 0:
            self._source_pixmap = None
            self.image_label.clear()
            self.image_label.setText("(empty)")
            self.info_label.setText("(empty)")
            return

        # Apply display stretch
        stretched = apply_display_stretch(block_data, (1, 99))

        # Retain the full image so resizing never rescales an already-scaled crop.
        qimage = numpy_to_qimage(stretched)
        self._source_pixmap = QPixmap.fromImage(qimage)
        self._resize_preview()

        # Display info
        dimensions = f"{block_data.shape[1]} × {block_data.shape[0]} px"
        info_text = f"{block_id}\n{dimensions}" if block_id else dimensions
        self.info_label.setText(info_text)
