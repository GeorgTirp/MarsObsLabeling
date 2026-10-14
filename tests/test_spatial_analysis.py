"""Spatial analysis must preserve coordinates, missing data and lazy computation."""

from pathlib import Path

import numpy as np
import pytest

from marslabeler.inference.engine import InferencePlan, run_block_scores_with_pixel_maps
from marslabeler.inference.pixelmaps import PixelMapStore
from marslabeler.ui.render import create_pixel_heatmap_overlay
from tests.test_inference_engine import FakeRaster, make_block
from tests.test_mainwindow_inference import qapp, window_with_session


def test_score_maps_crop_partial_blocks_and_mask_nodata():
    class Raster:
        def read_window_padded(self, x, y, w, h, ow, oh):
            pixels = np.full((oh, ow), 120, dtype=np.uint8)
            pixels[1, 1] = 0
            return pixels

    block = make_block("edge", 8, 0, w=3, h=2)
    plan = InferencePlan(pad_size=8)
    scores = np.arange(64, dtype=np.float32).reshape(1, 8, 8)
    means, maps = run_block_scores_with_pixel_maps(Raster(), [block], plan, lambda *args: scores)
    expected = np.array([[0, 1, 2], [8, np.nan, 10]], dtype=np.float32)
    np.testing.assert_allclose(maps["edge"], expected, equal_nan=True)
    assert means["edge"] == pytest.approx(4.2)
    assert np.unique(maps["edge"][np.isfinite(maps["edge"])]).size == 5


def test_score_maps_scaled_to_native_grid_and_disk_round_trip():
    plan = InferencePlan(pad_size=8, native_window=4, batch_size=1)
    blocks = [make_block("left", 0, 0, 4, 4), make_block("right", 4, 0, 2, 4)]
    raster = FakeRaster(lambda x, y: 120)
    scored = np.broadcast_to(np.arange(8, dtype=np.float32)[None, None, :], (1, 8, 8))
    store = PixelMapStore()
    means, maps = run_block_scores_with_pixel_maps(
        raster, blocks, plan, lambda *args: scored, map_store=store,
    )
    assert maps is store
    np.testing.assert_array_equal(maps["left"][0], [0, 2, 5, 7])
    np.testing.assert_array_equal(maps["right"][0], [0, 3])
    assert means == {"left": 3.5, "right": 1.5}
    # Reading/updating must release file handles, including on Windows.
    before = maps["left"]
    maps["left"] = before + 1
    np.testing.assert_array_equal(maps["left"], before + 1)
    del maps["left"]
    assert list(maps) == ["right"]


def test_block_classifier_input_excludes_neighboring_blocks():
    raster = FakeRaster(lambda x, y: 120)
    block = make_block("small", 0, 0, 3, 2)
    received = []

    def score(local, context):
        received.append(local.copy())
        return np.ones_like(local, dtype=np.float32)

    run_block_scores_with_pixel_maps(
        raster, [block], InferencePlan(pad_size=8, clip_to_block=True), score,
    )
    assert np.all(received[0][0, :2, :3] == 120)
    assert np.count_nonzero(received[0]) == 6


def test_pixel_overlay_has_fixed_scale_and_transparent_missing_pixels(qapp):
    values = np.array([[0.0, 5.0, 10.0, np.nan]], dtype=np.float32)
    image = create_pixel_heatmap_overlay(values, 4, 1, maximum=10).toImage()
    colors = [image.pixelColor(x, 0) for x in range(4)]
    assert colors[0].blue() == 255
    assert colors[1].red() > 240 and colors[1].green() > 240
    assert colors[2].red() == 255 and colors[2].blue() == 0
    assert colors[3].alpha() == 0
    second = create_pixel_heatmap_overlay(np.array([[5.0]]), 1, 1, maximum=10).toImage()
    assert second.pixelColor(0, 0) == colors[1]


def test_layers_compute_only_on_first_click_and_independently(window_with_session, monkeypatch):
    window = window_with_session
    window.model_path = Path("model.pt")
    window.uncertainty_button.setEnabled(True)
    window.laplace_button.setEnabled(True)
    calls = []
    monkeypatch.setattr(window, "_refresh_view", lambda: None)

    def compute(method):
        calls.append(method)
        target = window.block_laplace if method == "laplace" else window.block_uncertainty
        target["TEST_OBS_0_0"] = 0.2

    monkeypatch.setattr(window, "_compute_uncertainty_layer", compute)
    assert not calls
    window.uncertainty_button.click()
    assert calls == ["mahalanobis"]
    window.uncertainty_button.click()
    window.uncertainty_button.click()
    assert calls == ["mahalanobis"]
    window.laplace_button.click()
    assert calls == ["mahalanobis", "laplace"]
    assert not window.uncertainty_button.isChecked()
    assert window.display_layer == "laplace"
    window.laplace_button.click()
    assert window.display_layer == "classes"


def test_cancelled_layer_keeps_previous_map(window_with_session, monkeypatch):
    window = window_with_session
    window.model_path = Path("model.pt")
    window.display_layer = "uncertainty"
    window.uncertainty_button.setChecked(True)
    window.laplace_button.setEnabled(True)
    monkeypatch.setattr(window, "_compute_uncertainty_layer", lambda method: None)
    window.laplace_button.click()
    assert window.display_layer == "uncertainty"
    assert window.uncertainty_button.isChecked()
    assert not window.laplace_button.isChecked()


@pytest.mark.parametrize("layer", ["uncertainty", "laplace"])
def test_visible_map_sampling_preserves_pixel_coordinates(window_with_session, layer):
    window = window_with_session
    window.display_layer = layer
    window.canvas.canvas_width, window.canvas.canvas_height = 8, 4
    left, right = list(window.session.grid.iter_blocks())[:2]
    maps = window.pixel_uncertainty if layer == "uncertainty" else window.pixel_laplace
    maps[left.block_id] = np.tile(np.arange(512), (512, 1))
    maps[right.block_id] = np.tile(np.arange(512) + 1000, (512, 1))
    captured = {}
    window.canvas.set_pixel_scalar_overlay = lambda values, **kw: captured.update(values=values, **kw)
    assert window._render_pixel_analysis(256, 0, 1024, 512)
    np.testing.assert_allclose(captured["values"][0], [256, 384, 1000, 1128, 1256, 1384, np.nan, np.nan])


def test_all_nodata_scores_are_not_false_certainty():
    means, maps = run_block_scores_with_pixel_maps(
        FakeRaster(lambda x, y: 0), [make_block("empty", 0, 0)],
        InferencePlan(pad_size=8), lambda local, ctx: np.zeros_like(local, dtype=np.float32),
    )
    assert means == {}
    assert np.isnan(maps["empty"]).all()
