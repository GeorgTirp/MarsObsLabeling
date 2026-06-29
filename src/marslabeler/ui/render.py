"""Rendering utilities: numpy→QImage, display stretch, overlay composition."""

from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QColor, QPainter, QPen, QBrush

from marslabeler.io.preprocess import apply_display_stretch_with_mask, compute_invalid_mask


def numpy_to_qimage(data: np.ndarray) -> QImage:
    """
    Convert 8-bit grayscale numpy array to QImage.

    Args:
        data: np.ndarray of shape (height, width), dtype uint8

    Returns:
        QImage in Grayscale8 format
    """
    if data.dtype != np.uint8:
        raise ValueError(f"Expected uint8, got {data.dtype}")

    height, width = data.shape
    bytes_per_line = width
    return QImage(data.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)


def apply_display_stretch(
    data: np.ndarray,
    percentiles: Tuple[int, int] = (1, 99),
) -> np.ndarray:
    """
    Apply robust percentile stretch to enhance contrast (viewing only).

    This is for display purposes only and never affects labels.
    Automatically detects and ignores invalid (nodata/black) pixels.

    Args:
        data: Input array (any dtype)
        percentiles: (low, high) percentiles for stretch

    Returns:
        uint8 array stretched to [0, 255], with invalid pixels set to 0
    """
    if data.size == 0:
        return np.zeros((data.shape), dtype=np.uint8)

    # Detect invalid pixels (dtype-aware: 0 and 255 for uint8, etc.)
    invalid_mask = compute_invalid_mask(data)

    # Use preprocessing function that handles invalid pixels
    return apply_display_stretch_with_mask(data, percentiles, invalid_mask)


def create_grid_overlay(
    width: int,
    height: int,
    grid_spacing: int,
    color: QColor = QColor(0, 255, 0),
    line_width: int = 1,
) -> QPixmap:
    """
    Create a grid overlay image.

    Args:
        width, height: Canvas dimensions
        grid_spacing: Pixels between grid lines
        color: Grid line color
        line_width: Line width in pixels

    Returns:
        QPixmap with transparent background and grid
    """
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    pen = QPen(color, line_width, Qt.PenStyle.SolidLine)
    painter.setPen(pen)

    # Vertical lines
    for x in range(0, width + 1, grid_spacing):
        painter.drawLine(x, 0, x, height)

    # Horizontal lines
    for y in range(0, height + 1, grid_spacing):
        painter.drawLine(0, y, width, y)

    painter.end()
    return pixmap


def create_block_overlay(
    width: int,
    height: int,
    block_width: int,
    block_height: int,
    block_data: np.ndarray,  # 2D array of class IDs
    class_colors: dict[int, str],  # class_id -> hex color
    alpha: float = 0.4,
) -> QPixmap:
    """
    Create an overlay with colored blocks for labeled classes.

    Args:
        width, height: Total canvas size
        block_width, block_height: Size of each block in pixels
        block_data: 2D array where each element is a class_id
        class_colors: Mapping from class_id to hex color (e.g., "#FF0000")
        alpha: Alpha blend factor (0-1)

    Returns:
        QPixmap with transparent background and colored blocks
    """
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setOpacity(alpha)

    for row in range(block_data.shape[0]):
        for col in range(block_data.shape[1]):
            class_id = block_data[row, col]
            if class_id not in class_colors or class_id == -3:  # -3 = unlabeled
                continue

            color = QColor(class_colors[class_id])
            brush = QBrush(color)
            x = col * block_width
            y = row * block_height
            painter.fillRect(x, y, block_width, block_height, brush)

    painter.end()
    return pixmap


def create_current_block_highlight(
    width: int,
    height: int,
    block_width: int,
    block_height: int,
    block_row: int,
    block_col: int,
    color: QColor = QColor(255, 255, 0),
    line_width: int = 3,
) -> QPixmap:
    """
    Create a highlight box around the current block.

    Args:
        width, height: Canvas size
        block_width, block_height: Block dimensions
        block_row, block_col: Current block position
        color: Highlight color
        line_width: Border thickness

    Returns:
        QPixmap with transparent background and highlight rect
    """
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    pen = QPen(color, line_width, Qt.PenStyle.SolidLine)
    painter.setPen(pen)

    x = block_col * block_width
    y = block_row * block_height
    painter.drawRect(x, y, block_width, block_height)

    painter.end()
    return pixmap
