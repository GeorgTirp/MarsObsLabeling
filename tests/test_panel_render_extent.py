"""Single-panel rendering must cover a full panel_size square.

Regression guard for the border-panel block mismatch: the canvas draws a uniform
blocks_per_panel_row x blocks_per_panel_col grid and maps clicks through it, so
the rendered image has to span exactly panel_size on each side. Rendering the
clipped extent returned by Grid.get_panel_coords() stretches the partial edge
panels to fill the square canvas, which desynchronises the click mapping from the
block actually shown in the side preview.

Uses an image whose dimensions are not a whole number of panels, so the right-hand
column and bottom row of panels are genuinely partial (the realistic HiRISE case).
"""

import numpy as np
import pytest
import rasterio
from pathlib import Path
from rasterio.transform import Affine

from PySide6.QtWidgets import QApplication

from marslabeler.classes import load_classes
from marslabeler.io.raster import RasterSource
from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import LabelStore
from marslabeler.model.session import Session
from marslabeler.ui.mainwindow import MainWindow

IMG_W, IMG_H = 2500, 2100
PANEL_SIZE, BLOCK_SIZE = 1024, 128


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def ragged_geotiff(tmp_path) -> Path:
    """A raster whose size is not a multiple of panel_size or block_size."""
    data = np.random.randint(50, 200, (IMG_H, IMG_W), dtype=np.uint8)
    out = tmp_path / "ragged.tif"
    with rasterio.open(
        str(out), "w", driver="GTiff", height=IMG_H, width=IMG_W, count=1,
        dtype=np.uint8, crs="EPSG:4326", transform=Affine.identity(),
    ) as dst:
        dst.write(data, 1)
    return out


@pytest.fixture
def window(qapp, ragged_geotiff):
    win = MainWindow(Path("configs/app.yaml"))
    # _load_observation() normally does this; these tests construct the session directly
    win.classes_scheme = load_classes(win.config.paths.classes_file)
    raster = RasterSource(ragged_geotiff)
    raster.open()
    grid = Grid(IMG_W, IMG_H, PANEL_SIZE, BLOCK_SIZE, "RAGGED_OBS", Affine.identity())
    labels = LabelStore(grid, "test_user")
    win.session = Session(raster, grid, labels, win.config.to_dict())
    yield win
    raster.close()


def _spy_reads(window):
    """Record the (width, height) of every panel-extent read the renderer issues."""
    raster = window.session.raster
    calls = {"padded": [], "plain": []}
    real_padded = raster.read_window_padded
    real_plain = raster.read_window

    def padded(x, y, w, h, ow, oh):
        calls["padded"].append((w, h))
        return real_padded(x, y, w, h, ow, oh)

    def plain(x, y, w, h, ow, oh):
        calls["plain"].append((w, h))
        return real_plain(x, y, w, h, ow, oh)

    raster.read_window_padded = padded
    raster.read_window = plain
    return calls


def test_edge_panels_render_full_panel_extent(window):
    """Every panel -- including the clipped right/bottom ones -- renders panel_size square."""
    grid = window.session.grid
    assert grid.panels_across * PANEL_SIZE > IMG_W, "fixture must have a partial last column"
    assert grid.panels_down * PANEL_SIZE > IMG_H, "fixture must have a partial last row"

    for panel_idx in range(grid.num_panels):
        window.session.move_to_panel(panel_idx)
        calls = _spy_reads(window)
        window._load_current_panel()

        assert calls["padded"], f"panel {panel_idx} issued no padded read"
        w, h = calls["padded"][0]
        assert (w, h) == (PANEL_SIZE, PANEL_SIZE), (
            f"panel {panel_idx} rendered a {w}x{h} extent; the canvas grid assumes "
            f"{PANEL_SIZE}x{PANEL_SIZE}, so blocks would not line up with the drawn grid"
        )


def test_edge_panel_is_actually_clipped(window):
    """Guards the fixture: get_panel_coords really does return a short extent here."""
    grid = window.session.grid
    last = grid.num_panels - 1
    _x, _y, w, h = grid.get_panel_coords(last)
    assert w < PANEL_SIZE and h < PANEL_SIZE, (
        "bottom-right panel is not partial -- this fixture no longer exercises the bug"
    )
