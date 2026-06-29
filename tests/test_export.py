"""Tests for label export."""

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from marslabeler.classes import load_classes
from marslabeler.model.export import export_coarse_geotiff, export_class_metadata
from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import LabelStore


@pytest.fixture
def test_grid():
    """Create a test grid."""
    return Grid(4096, 4096, 4096, 512, "TEST_OBS", Affine.identity())


@pytest.fixture
def test_store_with_labels(test_grid):
    """Create a label store with some labels."""
    store = LabelStore(test_grid, "test_user")
    blocks = list(store.records.keys())

    # Label some blocks
    store.assign(blocks[0], 0, "Class A")
    store.assign(blocks[1], 1, "Class B")
    store.assign(blocks[2], -1, "Abstain")
    store.set_nodata(blocks[3])
    # blocks[4:] remain unlabeled

    return store


def test_export_coarse_geotiff_shape(test_store_with_labels, test_grid, tmp_path):
    """Test that exported GeoTIFF has correct shape."""
    output_path = tmp_path / "labels.tif"
    export_coarse_geotiff(test_store_with_labels, test_grid, output_path)

    with rasterio.open(str(output_path)) as src:
        # One pixel per block: 8x8 = 64 blocks per panel, 1 panel
        assert src.width == 8
        assert src.height == 8
        assert src.count == 1
        # dtypes returns tuple of strings, e.g. ('uint8',)
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 255


def test_export_coarse_geotiff_values(test_store_with_labels, test_grid, tmp_path):
    """Test that exported GeoTIFF has correct class values."""
    output_path = tmp_path / "labels.tif"
    export_coarse_geotiff(test_store_with_labels, test_grid, output_path)

    with rasterio.open(str(output_path)) as src:
        data = src.read(1)

        # Block 0 at (0,0) → labeled as class 0
        assert data[0, 0] == 0

        # Block 1 at (0,1) → labeled as class 1
        assert data[0, 1] == 1

        # Block 2 at (0,2) → abstain (253)
        assert data[0, 2] == 253

        # Block 3 at (0,3) → nodata (254)
        assert data[0, 3] == 254

        # Remaining blocks → unlabeled (255 = nodata)
        assert data[0, 4] == 255


def test_export_geotiff_geotransform(test_store_with_labels, test_grid, tmp_path):
    """Test that geotransform is scaled correctly."""
    output_path = tmp_path / "labels.tif"
    export_coarse_geotiff(test_store_with_labels, test_grid, output_path)

    with rasterio.open(str(output_path)) as src:
        # Affine should be scaled by block_size (512)
        transform = src.transform
        # With identity source and block_size=512:
        # coarse_a = 1.0 * 512 = 512
        assert transform.a == 512.0
        # Rasterio's identity transform may differ; just check it's scaled
        # The key point is that the scale factor was applied
        assert abs(transform.e) == 512.0 or transform.e == 512.0


def test_export_geotiff_crs(test_store_with_labels, test_grid, tmp_path):
    """Test that CRS is preserved in export."""
    # Create grid with EPSG:4326
    grid = Grid(4096, 4096, 4096, 512, "TEST_OBS", Affine.identity())
    grid.crs = "EPSG:4326"

    store = LabelStore(grid, "test_user")
    output_path = tmp_path / "labels.tif"
    export_coarse_geotiff(store, grid, output_path)

    with rasterio.open(str(output_path)) as src:
        # CRS should be set
        assert src.crs is not None


def test_export_multi_panel_layout(tmp_path):
    """Test export with multiple panels."""
    # 8192x8192 with 4096 panels = 2x2 = 4 panels
    grid = Grid(8192, 8192, 4096, 512, "TEST_OBS", Affine.identity())
    store = LabelStore(grid, "test_user")

    blocks = list(store.records.keys())
    # Label first block of each panel
    store.assign(blocks[0], 0, "Class A")
    store.assign(blocks[64], 1, "Class B")  # First block of panel 1
    store.assign(blocks[128], 2, "Class C")  # First block of panel 2
    store.assign(blocks[192], 3, "Class D")  # First block of panel 3

    output_path = tmp_path / "labels_multi.tif"
    export_coarse_geotiff(store, grid, output_path)

    with rasterio.open(str(output_path)) as src:
        # 2x2 panels * 8x8 blocks per panel = 16x16 output
        assert src.width == 16
        assert src.height == 16

        data = src.read(1)
        # Panel 0, block 0 at (0, 0)
        assert data[0, 0] == 0
        # Panel 1, block 0 at (0, 8) [right panel]
        assert data[0, 8] == 1
        # Panel 2, block 0 at (8, 0) [bottom panel]
        assert data[8, 0] == 2
        # Panel 3, block 0 at (8, 8) [bottom-right panel]
        assert data[8, 8] == 3


def test_export_class_metadata(tmp_config_dir, tmp_path):
    """Test exporting class metadata."""
    classes_path = tmp_config_dir / "classes.yaml"
    classes_scheme = load_classes(classes_path)

    output_path = tmp_path / "classes.json"
    export_class_metadata(None, classes_scheme, output_path)

    assert output_path.exists()

    with open(output_path) as f:
        data = json.load(f)

    assert "classes" in data
    assert len(data["classes"]) > 0
    # Should have user classes + abstain + nodata
    assert len(data["classes"]) >= 3

    # Check first class
    first_class = data["classes"][0]
    assert "id" in first_class
    assert "name" in first_class
    assert "color" in first_class
