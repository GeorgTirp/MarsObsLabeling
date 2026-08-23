"""Tests for predictdialog.py's model-channel -> classes.yaml id remap, both
for the majority-voted scalar (existing, exercised indirectly via PredictWorker
in acceptance tests) and the new per-pixel crop remap used for the pixel-wise
prediction view."""

import numpy as np
import pytest

from marslabeler.ui.predictdialog import _remap_pixel_maps


def test_remap_pixel_maps_empty_input_returns_empty_dict():
    assert _remap_pixel_maps({}, {0: 1, 1: 2}) == {}


def test_remap_pixel_maps_applies_mapping_per_pixel():
    maps = {"b0": np.array([[0, 1], [1, 2]], dtype=np.uint8)}
    out = _remap_pixel_maps(maps, {0: 5, 1: 6, 2: 7})
    assert out.keys() == maps.keys()
    assert np.array_equal(out["b0"], np.array([[5, 6], [6, 7]], dtype=np.uint8))
    assert out["b0"].dtype == np.uint8


def test_remap_pixel_maps_preserves_multiple_blocks():
    maps = {
        "b0": np.full((2, 2), 0, dtype=np.uint8),
        "b1": np.full((2, 2), 2, dtype=np.uint8),
    }
    out = _remap_pixel_maps(maps, {0: 10, 2: 12})
    assert np.all(out["b0"] == 10)
    assert np.all(out["b1"] == 12)


def test_remap_pixel_maps_raises_on_unmapped_channel_index():
    maps = {"b0": np.array([[0, 3]], dtype=np.uint8)}  # channel 3 has no mapping
    with pytest.raises(ValueError, match="channel index 3"):
        _remap_pixel_maps(maps, {0: 1})


def test_remap_pixel_maps_shape_unchanged():
    maps = {"b0": np.zeros((7, 5), dtype=np.uint8)}
    out = _remap_pixel_maps(maps, {0: 4})
    assert out["b0"].shape == (7, 5)
