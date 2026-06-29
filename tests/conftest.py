"""Test fixtures."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine, from_bounds


@pytest.fixture
def tmp_config_dir():
    """Create a temporary directory with test config files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a minimal app.yaml
        app_yaml = tmpdir / "app.yaml"
        app_yaml.write_text("""
paths:
  classes_file: classes.yaml
  labels_dir: ./labels
geometry:
  panel_size: 4096
  block_size: 512
navigation:
  advance_mode: next_unlabeled
  advance_on_edit: false
display:
  max_canvas_px: 1600
  stretch_percentiles: [1, 99]
skip:
  nodata_skip_threshold: 0.5
  variance_skip_threshold: 0.0
  skip_low_variance: false
autosave:
  every_n_labels: 25
  every_seconds: 30
export:
  full_res: false
labeler: null
""")

        # Create a minimal classes.yaml
        classes_yaml = tmpdir / "classes.yaml"
        classes_yaml.write_text("""
classes:
  - { id: 0,  name: "Class A", color: "#4C72B0", hotkey: "q" }
  - { id: 1,  name: "Class B", color: "#DD8452", hotkey: "w" }
abstain:
  id: -1
  name: "Abstain"
  color: "#000000"
  hotkey: "space"
nodata:
  id: -2
  name: "No data"
  color: "#222222"
""")

        yield tmpdir


@pytest.fixture
def synthetic_geotiff(tmp_path) -> Path:
    """Create a synthetic GeoTIFF for testing."""
    # Create a 4096x4096 image with a known nodata quadrant
    width, height = 4096, 4096
    data = np.random.randint(50, 200, (height, width), dtype=np.uint8)

    # Top-left quadrant: nodata (0)
    data[0 : height // 2, 0 : width // 2] = 0

    # Top-right quadrant: constant (uniform/low-variance)
    data[0 : height // 2, width // 2 : width] = 100

    # Geotransform: pixel at (0, 0) = map coord (0, 0), 1 meter/pixel
    transform = Affine.identity()

    output_path = tmp_path / "test.tif"
    with rasterio.open(
        str(output_path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs="EPSG:4326",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data, 1)

    return output_path


@pytest.fixture
def synthetic_geotiff_with_overviews(synthetic_geotiff) -> Path:
    """Build overviews on the synthetic GeoTIFF."""
    # For testing, we'll just use the base file (internal overview building
    # would require gdaladdo or rasterio.shutil.copy with copy_src_overviews)
    return synthetic_geotiff
