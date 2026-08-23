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


def value_to_heatmap_color(value: float) -> QColor:
    """Map a [0,1] scalar to blue(low) -> yellow(mid) -> red(high)."""
    v = max(0.0, min(1.0, value))
    if v < 0.5:
        t = v / 0.5
        r, g, b = int(255 * t), int(255 * t), int(255 * (1 - t))
    else:
        t = (v - 0.5) / 0.5
        r, g, b = 255, int(255 * (1 - t)), 0
    return QColor(r, g, b)


def create_heatmap_overlay(
    width: int,
    height: int,
    block_width: int,
    block_height: int,
    block_values: np.ndarray,  # 2D float array in [0,1]; NaN = no value (left uncolored)
    alpha: float = 0.5,
) -> QPixmap:
    """
    Create a continuous-value heatmap overlay (e.g. uncertainty), one flat color per block.

    Args:
        width, height: Total canvas size
        block_width, block_height: Size of each block in pixels
        block_values: 2D array of scores in [0, 1]; NaN cells are left transparent
        alpha: Alpha blend factor (0-1)

    Returns:
        QPixmap with transparent background and colored blocks
    """
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setOpacity(alpha)

    for row in range(block_values.shape[0]):
        for col in range(block_values.shape[1]):
            value = block_values[row, col]
            if np.isnan(value):
                continue

            brush = QBrush(value_to_heatmap_color(float(value)))
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


def _nearest_neighbor_resize(data: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    """Resize a 2D array by nearest-neighbor index selection (no averaging).

    Discrete class ids must never be interpolated -- averaging class 2 and
    class 5 does not mean class 3.5 -- so this is used instead of QImage's own
    (bilinear/smooth) scaling for per-pixel class overlays. Downsampling only
    in practice (a native panel crop is >= the canvas it's decimated to), but
    correct for upsampling too.
    """
    in_height, in_width = data.shape
    if in_height == out_height and in_width == out_width:
        return data
    row_idx = (np.arange(out_height) * in_height // max(out_height, 1)).clip(0, in_height - 1)
    col_idx = (np.arange(out_width) * in_width // max(out_width, 1)).clip(0, in_width - 1)
    return data[row_idx[:, None], col_idx]


def create_pixel_class_overlay(
    class_ids: np.ndarray,
    class_colors: dict[int, str],
    out_width: int,
    out_height: int,
    alpha: float = 0.5,
) -> QPixmap:
    """Per-pixel colored overlay for a full semantic-segmentation prediction
    map -- the pixel-wise counterpart to create_block_overlay's per-block one.

    Args:
        class_ids: 2D array (native resolution, any size) of predicted class
            ids; values with no entry in class_colors (including any sentinel
            used for "no prediction here", e.g. -1) render fully transparent.
        class_colors: class_id -> hex color.
        out_width, out_height: canvas size to decimate to (nearest-neighbor).
        alpha: alpha blend factor (0-1).

    Returns:
        QPixmap (out_width x out_height) with transparent background and
        colored pixels.
    """
    resized = _nearest_neighbor_resize(class_ids, out_height, out_width).astype(np.int32)

    # Size the table to cover every id actually present, not just the known
    # ones -- an id with no entry in class_colors (unknown/out-of-range, not
    # just the -1 "no prediction" sentinel) must land on the table's
    # zero-initialized default (transparent), never get folded into some
    # unrelated known class by a clip.
    known_ids = list(class_colors.keys())
    max_known = max(known_ids) if known_ids else -1
    max_seen = int(resized.max()) if resized.size else -1
    max_id = max(max_known, max_seen)

    # +1 offset: table index 0 is reserved for "no known color" (unmapped ids,
    # including the -1 nodata/no-prediction sentinel), fully transparent.
    lut = np.zeros((max_id + 2, 4), dtype=np.uint8)
    for class_id, hex_color in class_colors.items():
        if class_id < 0:
            continue
        color = QColor(hex_color)
        lut[class_id + 1] = (color.red(), color.green(), color.blue(), int(round(255 * alpha)))

    # max_id already covers resized's true max, so this only guards against a
    # sentinel more negative than -1, which shouldn't occur in practice.
    safe_ids = np.clip(resized, -1, max_id)
    rgba = np.ascontiguousarray(lut[safe_ids + 1])  # (out_height, out_width, 4) uint8

    qimage = QImage(rgba.data, out_width, out_height, out_width * 4, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimage)
