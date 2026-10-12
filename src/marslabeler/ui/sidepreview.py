"""Responsive inspector for the current block's native image crop."""

import numpy as np
from PySide6.QtCore import Qt, QSize, QEvent
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy

from marslabeler.ui.render import numpy_to_qimage, apply_display_stretch


class SidePreview(QWidget):
    """Displays the full native crop, fitted to the available preview area."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("blockInspector")
        self.setMinimumWidth(240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("QWidget#blockInspector { background-color: #10161d; }")
        self._source_pixmap: QPixmap | None = None

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(8)
        self.setLayout(layout)

        title = QLabel("BLOCK PREVIEW")
        title.setStyleSheet(
            "color: #e6edf3; font-size: 11px; font-weight: 600; background: transparent;"
        )
        layout.addWidget(title)

        subtitle = QLabel("Current block · full native crop")
        subtitle.setStyleSheet(
            "color: #91a0b2; font-size: 11px; background: transparent;"
        )
        layout.addWidget(subtitle)

        # The crop fits the available area in both directions, so enlarging
        # the preview never pushes part of the block off screen. Ignore the
        # pixmap's size hint to keep the splitter freely resizable afterwards.
        self.image_label = QLabel("Select a block to inspect")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_label.setMinimumHeight(160)
        self.image_label.installEventFilter(self)
        self.image_label.setStyleSheet(
            "background-color: #0c1117; border: 1px solid #2c3947; "
            "border-radius: 4px; color: #91a0b2; font-size: 11px;"
        )
        layout.addWidget(self.image_label, 1)

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setTextFormat(Qt.TextFormat.PlainText)
        self.info_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.info_label.setStyleSheet(
            "font-size: 11px; color: #91a0b2; background: transparent; padding-top: 2px;"
        )
        layout.addWidget(self.info_label)

        self.resize(self.sizeHint())
        self._resize_preview()

    def sizeHint(self) -> QSize:
        return QSize(528, 620)

    def eventFilter(self, watched, event):
        if watched is self.image_label and event.type() == QEvent.Type.Resize:
            self._resize_preview()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._resize_preview()

    def _resize_preview(self) -> None:
        """Fit the cached image without touching the raster or display stretch."""
        if self._source_pixmap is not None:
            # Leave the image frame visible, and letterbox partial edge crops.
            self.image_label.setPixmap(
                self._source_pixmap.scaled(
                    max(1, self.image_label.width() - 2),
                    max(1, self.image_label.height() - 2),
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
