"""Tests for the (torch-free) block-level inference engine."""

import numpy as np
import pytest

from marslabeler.inference.engine import (
    InferencePlan,
    compute_global_quantization,
    quantize_to_uint8,
    run_block_inference,
)
from marslabeler.model.grid import BlockInfo


def make_block(block_id: str, x: int, y: int, w: int = 8, h: int = 8) -> BlockInfo:
    return BlockInfo(
        block_id=block_id,
        panel_idx=0,
        panel_row=0,
        panel_col=0,
        block_row=y // h,
        block_col=x // w,
        x_px=x,
        y_px=y,
        w_px=w,
        h_px=h,
    )


class FakeRaster:
    """Deterministic stand-in for RasterSource: fills each window with a value
    derived from its (x, y) origin, so tests can assert exactly what the engine read.
    """

    def __init__(self, value_fn):
        self.value_fn = value_fn
        self.calls = []

    def read_window_padded(self, x, y, width, height, out_width, out_height):
        self.calls.append((x, y, width, height, out_width, out_height))
        value = self.value_fn(x, y)
        return np.full((out_height, out_width), value, dtype=np.uint8)


def majority_vote_predict_fn(local_batch, context_batch):
    """Fake model: 'predicts' the constant value baked into each fake window."""
    return local_batch.astype(np.int64)


def test_run_block_inference_basic_majority_vote():
    blocks = [make_block("b0", 0, 0), make_block("b1", 8, 0)]
    raster = FakeRaster(lambda x, y: 3 if x == 0 else 7)
    plan = InferencePlan(pad_size=8, batch_size=4)

    results = run_block_inference(raster, blocks, plan, majority_vote_predict_fn)

    assert results == {"b0": 3, "b1": 7}


def test_run_block_inference_batches_correctly():
    blocks = [make_block(f"b{i}", i * 8, 0) for i in range(5)]
    raster = FakeRaster(lambda x, y: (x // 8) % 5)
    plan = InferencePlan(pad_size=8, batch_size=2)

    results = run_block_inference(raster, blocks, plan, majority_vote_predict_fn)

    assert results == {f"b{i}": i for i in range(5)}


def test_run_block_inference_crops_padding_before_voting():
    """pad_size > block size: only the block's own w_px x h_px corner should count."""
    block = make_block("b0", 0, 0, w=4, h=4)
    raster = FakeRaster(lambda x, y: 0)  # ignored; predict_fn below overrides

    def predict_fn(local_batch, context_batch):
        # Whole padded window predicts class 9, except we can't see that from
        # FakeRaster alone -- build the array directly to prove cropping happens.
        out = np.full(local_batch.shape, 9, dtype=np.int64)
        out[:, :4, :4] = 2  # only the true block region votes class 2
        return out

    plan = InferencePlan(pad_size=16, batch_size=4)
    results = run_block_inference(raster, [block], plan, predict_fn)

    assert results == {"b0": 2}


def test_run_block_inference_progress_callback():
    blocks = [make_block(f"b{i}", i * 8, 0) for i in range(3)]
    raster = FakeRaster(lambda x, y: 1)
    plan = InferencePlan(pad_size=8, batch_size=2)

    calls = []
    run_block_inference(
        raster, blocks, plan, majority_vote_predict_fn,
        progress_cb=lambda done, total: calls.append((done, total)),
    )

    assert calls == [(2, 3), (3, 3)]


def test_run_block_inference_cancellation_stops_early():
    blocks = [make_block(f"b{i}", i * 8, 0) for i in range(4)]
    raster = FakeRaster(lambda x, y: 1)
    plan = InferencePlan(pad_size=8, batch_size=1)

    seen = []

    def predict_fn(local_batch, context_batch):
        seen.append(1)
        return local_batch.astype(np.int64)

    results = run_block_inference(
        raster, blocks, plan, predict_fn,
        should_cancel=lambda: len(seen) >= 2,
    )

    assert len(seen) == 2
    assert len(results) == 2


def test_run_block_inference_requests_context_window_when_needed():
    block = make_block("b0", 100, 100, w=8, h=8)
    raster = FakeRaster(lambda x, y: 1)
    plan = InferencePlan(pad_size=8, needs_context=True, context_px=32, batch_size=4)

    seen_context_shapes = []

    def predict_fn(local_batch, context_batch):
        seen_context_shapes.append(None if context_batch is None else context_batch.shape)
        return local_batch.astype(np.int64)

    run_block_inference(raster, [block], plan, predict_fn)

    assert seen_context_shapes == [(1, 8, 8)]
    # Context window should be centered on the block and sized context_px, decimated
    # down to pad_size -- i.e. one of the raster calls used width=height=32.
    assert any(call[2:4] == (32, 32) for call in raster.calls)


def test_run_block_inference_empty_blocks_returns_empty_dict():
    raster = FakeRaster(lambda x, y: 1)
    plan = InferencePlan(pad_size=8)
    assert run_block_inference(raster, [], plan, majority_vote_predict_fn) == {}


def test_predict_fn_batch_size_mismatch_raises():
    blocks = [make_block("b0", 0, 0)]
    raster = FakeRaster(lambda x, y: 1)
    plan = InferencePlan(pad_size=8, batch_size=4)

    def bad_predict_fn(local_batch, context_batch):
        return np.zeros((0, 8, 8), dtype=np.int64)  # wrong count

    with pytest.raises(ValueError, match="predict_fn returned"):
        run_block_inference(raster, blocks, plan, bad_predict_fn)


class FakeWholeImageRaster:
    """Stand-in for RasterSource that returns a fixed decimated array regardless
    of the requested window -- enough for compute_global_quantization, which only
    ever asks for one whole-image read."""

    def __init__(self, decimated: np.ndarray):
        self.decimated = decimated

    def read_window_padded(self, x, y, width, height, out_width, out_height):
        return self.decimated


def test_compute_global_quantization_returns_percentile_bounds():
    rng = np.random.default_rng(0)
    decimated = rng.integers(200, 800, size=(500, 500)).astype(np.uint16)
    raster = FakeWholeImageRaster(decimated)

    p_low, p_high = compute_global_quantization(raster, 25924, 37205, sample_size=500)

    expected_low, expected_high = np.percentile(decimated, [1, 99])
    assert p_low == pytest.approx(expected_low, abs=1.0)
    assert p_high == pytest.approx(expected_high, abs=1.0)


def test_compute_global_quantization_ignores_nodata_pixels():
    """uint16 convention: 0 = invalid (see io.preprocess.compute_invalid_mask)."""
    decimated = np.full((100, 100), 500, dtype=np.uint16)
    decimated[:50, :] = 0  # half the image is nodata

    raster = FakeWholeImageRaster(decimated)
    p_low, p_high = compute_global_quantization(raster, 1000, 1000, sample_size=100)

    # Only the valid (500-valued) half should count -- bounds collapse around 500,
    # not dragged toward 0 by the nodata half.
    assert p_low == pytest.approx(500.0, abs=1.0)
    assert p_high == pytest.approx(500.0, abs=1.0)


def test_compute_global_quantization_all_invalid_returns_safe_fallback():
    decimated = np.zeros((50, 50), dtype=np.uint16)
    raster = FakeWholeImageRaster(decimated)

    bounds = compute_global_quantization(raster, 1000, 1000, sample_size=50)

    assert bounds == (0.0, 1.0)  # degenerate but doesn't crash


def test_quantize_to_uint8_linear_mapping_within_bounds():
    batch = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    result = quantize_to_uint8(batch, bounds=(100.0, 400.0))

    assert result.dtype == np.uint8
    assert result[0, 0] == 0  # p_low -> 0
    assert result[1, 1] == 255  # p_high -> 255
    assert 0 < result[0, 1] < result[1, 0] < 255  # monotonic in between


def test_quantize_to_uint8_clips_out_of_range_values():
    batch = np.array([[-50, 1000]], dtype=np.int32)
    result = quantize_to_uint8(batch, bounds=(0.0, 500.0))

    assert result[0, 0] == 0
    assert result[0, 1] == 255
