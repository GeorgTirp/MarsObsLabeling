"""Predictions must land on exactly the block they were computed for.

Two ways that mapping can silently break, both regression-guarded here:

1. Block ids are ``{obs_id}_{x_px}_{y_px}``, so a labels file written at one tile
   size shares ids with a different tile size wherever origins coincide (every
   512px origin is also a 256px origin). Grafting those on would attach a label
   covering 4x the area to a smaller block -- predictions visibly displaced
   against the imagery.
2. A block whose origin lies past the image edge has no pixels of its own. Voting
   a class for it paints the overlay out over the black margin beyond the swath.
"""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import Affine

from marslabeler.inference.engine import NO_PREDICTION, _majority_class
from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import LabelStore

OBS = "TEST_OBS"
# Deliberately not a whole number of tiles, so edge blocks are clipped.
IMG_W, IMG_H = 2500, 2100
PANEL = 1024


def _grid(block_size: int) -> Grid:
    return Grid(IMG_W, IMG_H, PANEL, block_size, OBS, Affine.identity())


def _save_labelled(tmp_path, grid: Grid, class_id: int = 3):
    """Write a labels file with every in-image block assigned `class_id`."""
    store = LabelStore(grid, "tester")
    for block in grid.iter_blocks():
        if block.w_px > 0 and block.h_px > 0:
            store.assign(block.block_id, class_id, "Bedrock: Smooth", snapshot=False)
    path = tmp_path / f"{OBS}.parquet"
    store.save_parquet(path)
    return path


def test_same_geometry_round_trips(tmp_path):
    grid = _grid(256)
    path = _save_labelled(tmp_path, grid)
    reloaded = LabelStore.load_parquet(path, grid, "tester")
    for block in grid.iter_blocks():
        if block.w_px > 0 and block.h_px > 0:
            assert reloaded.get_record(block.block_id).class_id == 3


def test_no_record_keeps_a_mismatched_extent_across_tile_sizes(tmp_path):
    """The core guarantee: nothing is grafted onto a block it doesn't describe."""
    path = _save_labelled(tmp_path, _grid(512))
    finer = _grid(256)
    reloaded = LabelStore.load_parquet(path, finer, "tester")

    for block in finer.iter_blocks():
        record = reloaded.get_record(block.block_id)
        if record.class_id == 3:  # inherited from the file
            assert (record.x_px, record.y_px, record.w_px, record.h_px) == (
                block.x_px, block.y_px, block.w_px, block.h_px
            ), f"{block.block_id} inherited a label describing a different extent"


def test_coinciding_ids_exist_so_the_guard_is_load_bearing(tmp_path):
    """Guards the fixture: the two grids really do share block ids."""
    coarse = {b.block_id for b in _grid(512).iter_blocks()}
    fine = {b.block_id for b in _grid(256).iter_blocks()}
    assert coarse & fine, "no shared ids -- this test would pass vacuously"


def test_labels_from_another_observation_are_rejected(tmp_path):
    grid = _grid(256)
    path = _save_labelled(tmp_path, grid)
    other = Grid(IMG_W, IMG_H, PANEL, 256, "DIFFERENT_OBS", Affine.identity())
    reloaded = LabelStore.load_parquet(path, other, "tester")
    assert all(r.class_id != 3 for r in reloaded.records.values())


def test_empty_crop_votes_no_prediction_rather_than_class_zero():
    assert _majority_class(np.zeros((0, 0), dtype=np.int64)) == NO_PREDICTION
    assert _majority_class(np.zeros((256, 0), dtype=np.int64)) == NO_PREDICTION
    # a real crop still votes normally
    crop = np.full((4, 4), 7, dtype=np.int64)
    crop[0, 0] = 2
    assert _majority_class(crop) == 7


def test_zero_extent_blocks_exist_at_the_edges():
    """Guards the premise: this geometry really does produce pixel-less blocks."""
    ghosts = [b for b in _grid(256).iter_blocks() if b.w_px == 0 or b.h_px == 0]
    assert ghosts, "fixture no longer exercises out-of-image blocks"
    for block in ghosts:
        assert block.x_px >= IMG_W or block.y_px >= IMG_H
