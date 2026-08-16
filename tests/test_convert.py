"""Tests for io/convert.py's raster -> tiled GeoTIFF conversion."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from marslabeler.io.convert import convert_to_tiled_geotiff


@pytest.fixture
def small_raster(tmp_path):
    """A small, non-trivial single-band raster to convert (deliberately not a
    multiple of chunk_size, to exercise the edge-chunk path)."""
    path = tmp_path / "source.tif"
    width, height = 1000, 700
    rng = np.random.default_rng(0)
    data = rng.integers(0, 65535, (height, width), dtype=np.uint16)
    with rasterio.open(
        str(path), "w", driver="GTiff", height=height, width=width, count=1,
        dtype=np.uint16, crs="EPSG:4326", transform=Affine.identity(), nodata=0,
    ) as dst:
        dst.write(data, 1)
    return path, data


def test_convert_preserves_pixel_values(small_raster, tmp_path):
    src_path, original = small_raster
    dst_path = tmp_path / "out.tif"

    convert_to_tiled_geotiff(src_path, dst_path, block_size=256, chunk_size=512)

    with rasterio.open(str(dst_path)) as ds:
        converted = ds.read(1)
    assert np.array_equal(converted, original)


def test_convert_output_is_tiled_at_block_size(small_raster, tmp_path):
    src_path, _ = small_raster
    dst_path = tmp_path / "out.tif"

    convert_to_tiled_geotiff(src_path, dst_path, block_size=256, chunk_size=512)

    with rasterio.open(str(dst_path)) as ds:
        assert ds.profile["tiled"] is True
        assert ds.block_shapes == [(256, 256)]


def test_convert_preserves_georeferencing_and_dtype(small_raster, tmp_path):
    src_path, _ = small_raster
    dst_path = tmp_path / "out.tif"

    convert_to_tiled_geotiff(src_path, dst_path, block_size=256, chunk_size=512)

    with rasterio.open(str(src_path)) as src, rasterio.open(str(dst_path)) as dst:
        assert dst.crs == src.crs
        assert dst.transform == src.transform
        assert dst.dtypes[0] == src.dtypes[0]
        assert dst.nodata == src.nodata
        assert (dst.width, dst.height) == (src.width, src.height)


def test_convert_rejects_chunk_size_not_multiple_of_block_size(small_raster, tmp_path):
    src_path, _ = small_raster
    dst_path = tmp_path / "out.tif"

    with pytest.raises(ValueError, match="multiple"):
        convert_to_tiled_geotiff(src_path, dst_path, block_size=300, chunk_size=512)


def test_convert_progress_callback_reaches_total(small_raster, tmp_path):
    src_path, _ = small_raster
    dst_path = tmp_path / "out.tif"
    calls = []

    convert_to_tiled_geotiff(
        src_path, dst_path, block_size=256, chunk_size=512,
        progress_cb=lambda done, total: calls.append((done, total)),
    )

    assert calls
    assert calls[-1][0] == calls[-1][1]  # final call reports done == total
    assert all(done <= total for done, total in calls)


def test_convert_creates_output_parent_dirs(small_raster, tmp_path):
    src_path, _ = small_raster
    dst_path = tmp_path / "nested" / "dir" / "out.tif"

    convert_to_tiled_geotiff(src_path, dst_path, block_size=256, chunk_size=512)

    assert dst_path.exists()
