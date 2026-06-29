"""Tests for raster reading."""

import numpy as np
import pytest

from marslabeler.io.raster import RasterSource


def test_raster_source_open_close(synthetic_geotiff):
    """Test opening and closing a raster."""
    src = RasterSource(synthetic_geotiff)
    src.open()

    assert src.width == 4096
    assert src.height == 4096
    assert src.nodata == 0

    src.close()


def test_raster_source_context_manager(synthetic_geotiff):
    """Test using RasterSource as a context manager."""
    with RasterSource(synthetic_geotiff) as src:
        assert src.width == 4096
        assert src.height == 4096


def test_read_window_full_resolution(synthetic_geotiff):
    """Test reading a window at full resolution."""
    with RasterSource(synthetic_geotiff) as src:
        # Read 256x256 window at (0, 0) at native size
        data = src.read_window(0, 0, 256, 256, 256, 256)

        assert data.shape == (256, 256)
        # Top-left quadrant is nodata (0)
        assert np.all(data == 0)


def test_read_window_decimated(synthetic_geotiff):
    """Test reading a window with decimation."""
    with RasterSource(synthetic_geotiff) as src:
        # Read 512x512 window, output as 64x64 (8x decimation)
        data = src.read_window(0, 0, 512, 512, 64, 64)

        assert data.shape == (64, 64)


def test_read_window_at_image_edge(synthetic_geotiff):
    """Test reading a window that extends beyond image bounds."""
    with RasterSource(synthetic_geotiff) as src:
        # Read a window that partially goes off the edge
        data = src.read_window(4000, 4000, 256, 256, 128, 128)

        # Should return the clamped size (256 in this case)
        assert data.shape == (128, 128)


def test_nodata_fraction_in_nodata_region(synthetic_geotiff):
    """Test nodata_fraction in a region with nodata."""
    with RasterSource(synthetic_geotiff) as src:
        # Top-left quadrant is all nodata (0)
        frac = src.nodata_fraction(0, 0, 512, 512)

        assert frac > 0.9  # Should be nearly 100%


def test_nodata_fraction_in_data_region(synthetic_geotiff):
    """Test nodata_fraction in a region with valid data."""
    with RasterSource(synthetic_geotiff) as src:
        # Bottom-right quadrant should have valid data
        frac = src.nodata_fraction(2048, 2048, 512, 512)

        assert frac < 0.1  # Should be nearly 0%


def test_variance_low_variance_region(synthetic_geotiff):
    """Test variance in a uniform region."""
    with RasterSource(synthetic_geotiff) as src:
        # Top-right quadrant is constant (100), so variance should be low
        var = src.variance(2048, 0, 512, 512)

        assert var < 10  # Very low variance (constant values)


def test_variance_high_variance_region(synthetic_geotiff):
    """Test variance in a random region."""
    with RasterSource(synthetic_geotiff) as src:
        # Bottom-right quadrant is random noise, high variance
        var = src.variance(2048, 2048, 512, 512)

        assert var > 1000  # High variance (random data)


def test_gsd_and_transform(synthetic_geotiff):
    """Test ground sample distance and affine transform."""
    with RasterSource(synthetic_geotiff) as src:
        gsd = src.gsd
        assert gsd == 1.0  # 1 meter/pixel from identity transform

        transform = src.transform
        # Identity transform has a=1, e=-1 (row-major with y going down)
        assert transform.a == 1.0


def test_raster_not_open_raises(synthetic_geotiff):
    """Test that accessing properties without open() raises."""
    src = RasterSource(synthetic_geotiff)
    with pytest.raises(RuntimeError, match="not open"):
        _ = src.width
