"""Tests for rendering utilities."""

import numpy as np
import pytest
from PySide6.QtGui import QImage, QColor, QPixmap

from marslabeler.ui.render import (
    numpy_to_qimage,
    apply_display_stretch,
    create_grid_overlay,
    create_block_overlay,
    create_current_block_highlight,
)


def test_numpy_to_qimage():
    """Test converting numpy array to QImage."""
    data = np.ones((100, 100), dtype=np.uint8) * 128
    qimage = numpy_to_qimage(data)

    assert qimage.width() == 100
    assert qimage.height() == 100
    assert qimage.format() == QImage.Format.Format_Grayscale8


def test_numpy_to_qimage_wrong_dtype():
    """Test error on non-uint8 input."""
    data = np.ones((100, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        numpy_to_qimage(data)


def test_apply_display_stretch():
    """Test display stretch."""
    data = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    stretched = apply_display_stretch(data, (0, 100))

    assert stretched.dtype == np.uint8
    assert stretched.shape == (2, 2)
    assert stretched.min() >= 0
    assert stretched.max() <= 255


def test_apply_display_stretch_uniform():
    """Test stretch on uniform data."""
    data = np.ones((10, 10), dtype=np.uint8) * 100
    stretched = apply_display_stretch(data, (1, 99))

    # Uniform data should map to middle gray
    assert np.all(stretched == 128)


def test_apply_display_stretch_empty():
    """Test stretch on empty array."""
    data = np.array([], dtype=np.uint8).reshape(0, 0)
    stretched = apply_display_stretch(data, (1, 99))

    assert stretched.shape == (0, 0)


def test_create_grid_overlay():
    """Test creating a grid overlay."""
    pixmap = create_grid_overlay(400, 400, 100, QColor(0, 255, 0), 1)

    assert pixmap.width() == 400
    assert pixmap.height() == 400


def test_create_block_overlay():
    """Test creating a block overlay."""
    block_data = np.array([[0, 1], [2, -3]], dtype=np.int16)
    class_colors = {0: "#FF0000", 1: "#00FF00", 2: "#0000FF"}

    pixmap = create_block_overlay(400, 400, 200, 200, block_data, class_colors, alpha=0.4)

    assert pixmap.width() == 400
    assert pixmap.height() == 400


def test_create_current_block_highlight():
    """Test creating a highlight box."""
    pixmap = create_current_block_highlight(400, 400, 100, 100, 1, 2, QColor(255, 255, 0), 3)

    assert pixmap.width() == 400
    assert pixmap.height() == 400
