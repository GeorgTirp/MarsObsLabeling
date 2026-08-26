"""Coverage preprocessing must not scale with the labeling tile size.

Deciding "is this block off-swath" once cost one windowed raster decode per
block, so halving the tile size quadrupled the wait before labeling could start
-- minutes on a HiRISE strip, for a decision that needs no precision at all.
It is now derived from a single decimated validity mask of the whole raster.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from marslabeler.io.raster import RasterSource
from marslabeler.model.grid import Grid
from marslabeler.ui.preprocessdialog import PreprocessWorker

IMG_W, IMG_H = 2048, 1536

CONFIG = {
    "skip": {
        "nodata_skip_threshold": 0.99,
        "skip_low_variance": False,
        "variance_skip_threshold": 0.0,
    }
}


@pytest.fixture
def half_empty(tmp_path):
    """Left half real data, right half nodata (0) -- a synthetic swath edge."""
    data = np.full((IMG_H, IMG_W), 120, dtype=np.uint8)
    data[:, IMG_W // 2:] = 0
    path = tmp_path / "swath.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=IMG_H, width=IMG_W, count=1,
        dtype=np.uint8, crs="EPSG:4326", transform=Affine.identity(),
    ) as dst:
        dst.write(data, 1)
    raster = RasterSource(path)
    raster.open()
    yield raster
    raster.close()


def _decisions(raster, block_size):
    grid = Grid(IMG_W, IMG_H, 1024, block_size, "OBS", Affine.identity())
    worker = PreprocessWorker(raster, grid, CONFIG)
    worker.run()
    return grid, worker.skip_decisions


def test_validity_mask_is_one_read_and_respects_a_pixel_budget(half_empty):
    mask, decimation = half_empty.validity_mask(max_pixels=10_000)
    assert mask.size <= 10_000
    assert decimation > 1
    assert mask.dtype == np.bool_
    # left half valid, right half not
    assert mask[:, : mask.shape[1] // 2 - 1].mean() > 0.95
    assert mask[:, mask.shape[1] // 2 + 1:].mean() < 0.05


def test_nodata_fractions_track_the_swath_edge(half_empty):
    grid, decisions = _decisions(half_empty, 256)
    for block in grid.iter_blocks():
        if block.w_px <= 0 or block.h_px <= 0:
            continue
        frac = decisions[block.block_id]["nodata_fraction"]
        if block.x_px + block.w_px <= IMG_W // 2:
            assert frac < 0.05, f"{block.block_id} is inside the swath"
        elif block.x_px >= IMG_W // 2:
            assert frac > 0.95, f"{block.block_id} is off-swath"


def test_every_block_gets_a_decision_at_any_tile_size(half_empty):
    for block_size in (512, 256, 128):
        grid, decisions = _decisions(half_empty, block_size)
        assert len(decisions) == grid.num_blocks()
        assert all(0.0 <= d["nodata_fraction"] <= 1.0 for d in decisions.values())


def test_zero_extent_blocks_count_as_fully_nodata(half_empty):
    """A block whose origin is past the image edge has no data by definition."""
    # 1536px tall vs 2 panels of 1024: the bottom panel row runs past the image,
    # so its lower blocks have no pixels of their own.
    grid, decisions = _decisions(half_empty, 256)
    ghosts = [b for b in grid.iter_blocks() if b.w_px == 0 or b.h_px == 0]
    assert ghosts, "fixture no longer produces out-of-image blocks"
    for block in ghosts:
        assert decisions[block.block_id]["nodata_fraction"] == 1.0
        assert decisions[block.block_id]["should_skip"] is True


def test_partially_empty_blocks_are_kept_for_the_human(half_empty):
    """At the 0.99 labeling threshold a half-empty block is still labelable."""
    grid, decisions = _decisions(half_empty, 256)
    straddling = [
        b for b in grid.iter_blocks()
        if b.w_px > 0 and b.x_px < IMG_W // 2 < b.x_px + b.w_px
    ]
    for block in straddling:
        assert decisions[block.block_id]["should_skip"] is False


def test_session_reuses_precomputed_fractions_instead_of_reading(half_empty, tmp_path):
    """Navigation must not re-decode a window per block."""
    from marslabeler.model.labelstore import LabelStore
    from marslabeler.model.session import Session

    grid, decisions = _decisions(half_empty, 256)
    session = Session(half_empty, grid, LabelStore(grid, "t"), CONFIG)
    session.skip_decisions = decisions

    def _boom(*args, **kwargs):
        raise AssertionError("nodata_fraction was read during navigation")

    half_empty.nodata_fraction = _boom
    block = next(b for b in grid.iter_blocks() if b.w_px > 0)
    session._should_skip_block(block)  # must not raise
