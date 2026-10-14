"""Cached summaries must stay tied to their observation and calibration."""

import json

from rasterio.transform import Affine

from marslabeler.inference.local_npca import LocalGalleryItem
from marslabeler.inference.summarycache import encode_summary, decode_summary
from marslabeler.model.grid import Grid


def test_summary_round_trip_and_calibration_change_invalidates(tmp_path):
    checkpoint = tmp_path / "best.pt"
    grid = Grid(10, 10, 16, 8, "OBS", Affine.identity())
    block = next(grid.iter_blocks())
    item = LocalGalleryItem(1, 1.2, block.block_id, block.block_row, block.block_col,
                            block.panel_idx, block.x_px, block.y_px, block.w_px, block.h_px)
    gallery = {2: {0: [item]}}
    encoded = encode_summary({block.block_id: .8}, gallery, block_classifier=True,
                             num_pixel_classes=14, npca_error="", checkpoint=checkpoint)
    raw = json.loads(json.dumps(encoded))
    decoded = decode_summary(raw, checkpoint=checkpoint, grid=grid)
    assert decoded["gallery"] == gallery
    assert decoded["confidence"] == {block.block_id: .8}
    assert decoded["block_classifier"]
    # Installing a newly fitted basis must rebuild activations, not reuse old rankings.
    (tmp_path / "best.npca.pt").write_bytes(b"new basis")
    assert decode_summary(raw, checkpoint=checkpoint, grid=grid) is None


def test_corrupt_or_misaligned_summary_requests_rebuild(tmp_path):
    checkpoint = tmp_path / "best.pt"
    grid = Grid(10, 10, 16, 8, "OBS", Affine.identity())
    for bad in (None, {}, {"version": 99}):
        assert decode_summary(bad, checkpoint=checkpoint, grid=grid) is None
    encoded = encode_summary({"wrong-observation": .8}, {}, block_classifier=False,
                             num_pixel_classes=14, npca_error="", checkpoint=checkpoint)
    assert decode_summary(encoded, checkpoint=checkpoint, grid=grid) is None
