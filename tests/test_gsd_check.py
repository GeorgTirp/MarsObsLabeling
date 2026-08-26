"""Scale (GSD) agreement between an observation and the model's training imagery.

A segmentation model learns terrain at a fixed pixels-per-metre. Run on imagery
of a different GSD, the same landform covers a different number of pixels than
it did in any training crop -- predictions still come out, but they are not
comparable to the model's validation scores. That degrades silently, so it must
be surfaced.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from marslabeler.inference.modelio import MARS_RADIUS_M, raster_gsd_metres

NOAH_GSD = 0.2414  # what the AI4ExoMars stage-3 checkpoints are trained on


def _raster(tmp_path, name, transform, crs):
    path = tmp_path / name
    with rasterio.open(
        path, "w", driver="GTiff", height=64, width=64, count=1,
        dtype=np.uint8, crs=crs, transform=transform,
    ) as dst:
        dst.write(np.zeros((64, 64), np.uint8), 1)
    return path


def test_projected_crs_reports_metres_directly(tmp_path):
    path = _raster(tmp_path, "p.tif", Affine.scale(0.25, -0.25), CRS.from_epsg(3857))
    with rasterio.open(path) as d:
        assert raster_gsd_metres(d) == pytest.approx(0.25)


def test_geographic_crs_is_converted_from_degrees(tmp_path):
    """NOAH-H's own source tiles are geographic; degrees must become metres."""
    degrees = 4.2395954e-06  # a real oxia_drg_v17 tile's pixel size
    path = _raster(tmp_path, "g.tif", Affine.scale(degrees, -degrees),
                   CRS.from_epsg(4326))
    with rasterio.open(path) as d:
        got = raster_gsd_metres(d)
    expected = degrees * (math.pi / 180.0) * MARS_RADIUS_M
    assert got == pytest.approx(expected)
    # that real tile is a ~0.25 m/px product
    assert got == pytest.approx(0.25, abs=0.01)


def test_degrees_are_not_mistaken_for_metres(tmp_path):
    """Guards the bug this conversion exists to prevent: 4.2e-06 'metres' would
    look absurdly fine-grained and never trip a coarser/finer check."""
    degrees = 4.2395954e-06
    path = _raster(tmp_path, "g2.tif", Affine.scale(degrees, -degrees),
                   CRS.from_epsg(4326))
    with rasterio.open(path) as d:
        assert raster_gsd_metres(d) > 0.1  # not 4.2e-06


@pytest.mark.parametrize(
    "observed,should_warn",
    [
        (0.2414, False),  # exactly the training scale
        (0.25, False),    # NOAH-H's own product spread -- within tolerance
        (0.2513, False),  # the geographic source tiles
        (0.5, True),      # HiRISE RDR at 2x coarser -- the real mismatch
        (0.125, True),    # 2x finer
        (1.0, True),
    ],
)
def test_tolerance_admits_product_spread_but_catches_a_factor_of_two(
    observed, should_warn
):
    tolerance = 0.2  # inference.gsd_log2_tolerance default
    warned = abs(math.log2(observed / NOAH_GSD)) > tolerance
    assert warned is should_warn


def test_invalid_transforms_are_ignored(tmp_path):
    path = _raster(tmp_path, "z.tif", Affine.scale(0.25, -0.25), CRS.from_epsg(3857))
    with rasterio.open(path) as d:
        assert raster_gsd_metres(d) is not None

    class _NoTransform:
        crs = None

        @property
        def transform(self):
            raise RuntimeError("no transform")

    assert raster_gsd_metres(_NoTransform()) is None


# --------------------------------------------------------------------------- #
# Scale-matched inference: reading a smaller native window and magnifying it
# --------------------------------------------------------------------------- #

from dataclasses import dataclass

from marslabeler.inference.engine import (
    InferencePlan,
    _block_crop,
    _to_native_resolution,
)
from marslabeler.inference.modelio import native_block_size_for_gsd

RATIO = 0.5 / 0.24138461  # the real ESP_016142_1625 vs NOAH-H mismatch


@dataclass
class _Blk:
    block_id: str
    block_row: int
    block_col: int
    panel_idx: int
    x_px: int
    y_px: int
    w_px: int
    h_px: int


def _blk(w, h):
    return _Blk("b", 0, 0, 0, 0, 0, w, h)


def test_no_correction_is_a_bit_identical_no_op():
    plan = InferencePlan(pad_size=512)
    assert plan.read_window_px == 512
    assert plan.model_px_per_native_px == 1.0
    out = np.arange(512 * 512).reshape(512, 512) % 14
    block = _blk(512, 512)
    crop = _block_crop(out, block, plan)
    assert crop.shape == (512, 512)
    assert np.array_equal(_to_native_resolution(crop, block), crop)


def test_scale_matched_plan_reads_a_smaller_native_window():
    native = round(512 / RATIO)
    plan = InferencePlan(pad_size=512, native_window=native)
    assert plan.read_window_px == native == 247
    # each native pixel becomes ~2.07 model pixels
    assert plan.model_px_per_native_px == pytest.approx(RATIO, rel=1e-2)


def test_a_full_block_maps_to_the_whole_model_window():
    native = round(512 / RATIO)
    plan = InferencePlan(pad_size=512, native_window=native)
    out = np.zeros((512, 512), dtype=np.int64)
    assert _block_crop(out, _blk(native, native), plan).shape == (512, 512)


def test_a_clipped_edge_block_maps_proportionally_and_round_trips():
    native = round(512 / RATIO)
    plan = InferencePlan(pad_size=512, native_window=native)
    out = np.zeros((512, 512), dtype=np.int64)
    block = _blk(100, 60)  # a partial block at the image edge
    crop = _block_crop(out, block, plan)
    assert crop.shape == (
        round(60 * plan.model_px_per_native_px),
        round(100 * plan.model_px_per_native_px),
    )
    # pixel maps are stored against the observation's own pixels
    assert _to_native_resolution(crop, block).shape == (60, 100)


def test_zero_extent_block_yields_an_empty_crop():
    plan = InferencePlan(pad_size=512, native_window=round(512 / RATIO))
    out = np.zeros((512, 512), dtype=np.int64)
    assert _block_crop(out, _blk(0, 0), plan).size == 0


def test_crop_never_exceeds_the_model_window():
    """A block larger than the window must be clamped, not index out of bounds."""
    plan = InferencePlan(pad_size=512, native_window=round(512 / RATIO))
    out = np.zeros((512, 512), dtype=np.int64)
    crop = _block_crop(out, _blk(5000, 5000), plan)
    assert crop.shape == (512, 512)


def test_native_block_size_shrinks_to_one_model_window():
    """The tile must cover exactly what one corrected window sees, or a block
    would extend past the window that predicted it."""
    assert native_block_size_for_gsd(512, 256, RATIO) == 247
    assert native_block_size_for_gsd(512, 256, 1.0) == 512  # unchanged when off


def test_class_ids_are_never_interpolated_on_the_way_back():
    """Downsampling a class map must stay on real class ids."""
    plan = InferencePlan(pad_size=512, native_window=round(512 / RATIO))
    out = np.random.default_rng(0).integers(0, 14, (512, 512))
    block = _blk(247, 247)
    back = _to_native_resolution(_block_crop(out, block, plan), block)
    assert set(np.unique(back)).issubset(set(np.unique(out)))
