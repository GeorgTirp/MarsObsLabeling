"""Tests for MainWindow's pixel-wise vs. block-wise prediction rendering toggle.

Bypasses _load_observation() (modal dialog + QThread) the same way
test_mainwindow_inference.py does -- constructs Session directly and assigns
it onto a bare MainWindow.
"""

import numpy as np
import pytest
from pathlib import Path
from rasterio.transform import Affine

from PySide6.QtWidgets import QApplication

from marslabeler.classes import load_classes
from marslabeler.io.raster import RasterSource
from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import LabelStore
from marslabeler.model.session import Session
from marslabeler.ui.mainwindow import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def window_with_session(qapp, tmp_config_dir, synthetic_geotiff):
    """Predictions-mode MainWindow with a small 2x2-blocks-per-panel session,
    small enough to hand-build exact expected stitched pixel arrays."""
    window = MainWindow(tmp_config_dir / "app.yaml", predictions_mode=True)
    window.classes_scheme = load_classes(tmp_config_dir / "classes.yaml")

    raster = RasterSource(synthetic_geotiff)
    raster.open()
    # panel_size=16, block_size=8 -> 2x2 blocks/panel, purely for cheap grid math;
    # unrelated to the raster's own real content (_build_pixel_class_array never
    # reads pixels through it).
    grid = Grid(4096, 4096, 16, 8, "TEST_OBS", Affine.identity())
    labels = LabelStore(grid, "test_user")
    window.session = Session(raster, grid, labels, window.config.to_dict())
    window.current_panel_idx = 0

    yield window
    raster.close()


def _panel0_blocks(window):
    return window.session.grid.get_panel_blocks(0)


# --------------------------------------------------------------------------
# Default state
# --------------------------------------------------------------------------
def test_default_render_mode_is_pixelwise():
    """The whole point of the feature: don't double inference cost by
    defaulting to the mode that needs a second pass -- pixelwise is free
    (already computed), so it's the default."""
    window = MainWindow(Path("configs/app.yaml"), predictions_mode=True)
    assert window.prediction_render_mode == "pixelwise"
    assert window.block_pixel_predictions == {}


def test_render_mode_button_hidden_outside_predictions_mode():
    window = MainWindow(Path("configs/app.yaml"), predictions_mode=False)
    assert window.render_mode_button.isVisible() is False


def test_render_mode_button_disabled_until_pixel_data_exists():
    window = MainWindow(Path("configs/app.yaml"), predictions_mode=True)
    assert window.render_mode_button.isEnabled() is False


# --------------------------------------------------------------------------
# _build_pixel_class_array: stitching
# --------------------------------------------------------------------------
def test_build_pixel_class_array_stitches_blocks_at_correct_offsets(window_with_session):
    window = window_with_session
    blocks = _panel0_blocks(window)
    assert len(blocks) == 4  # 2x2 blocks/panel

    for block in blocks:
        # Fill each block's crop with a distinct constant so placement is checkable.
        fill_value = block.block_row * 2 + block.block_col
        window.block_pixel_predictions[block.block_id] = np.full((8, 8), fill_value, dtype=np.uint8)

    stitched = window._build_pixel_class_array(blocks)
    assert stitched.shape == (16, 16)
    assert np.all(stitched[0:8, 0:8] == 0)  # block_row=0, block_col=0
    assert np.all(stitched[0:8, 8:16] == 1)  # block_row=0, block_col=1
    assert np.all(stitched[8:16, 0:8] == 2)  # block_row=1, block_col=0
    assert np.all(stitched[8:16, 8:16] == 3)  # block_row=1, block_col=1


def test_build_pixel_class_array_missing_block_left_as_sentinel(window_with_session):
    window = window_with_session
    blocks = _panel0_blocks(window)
    # Only cache 1 of 4 blocks (e.g. the rest were skipped as nodata during inference).
    window.block_pixel_predictions[blocks[0].block_id] = np.full((8, 8), 5, dtype=np.uint8)

    stitched = window._build_pixel_class_array(blocks)
    assert np.all(stitched[0:8, 0:8] == 5)
    assert np.all(stitched[8:, :] == -1)  # everything else: no cached prediction


# --------------------------------------------------------------------------
# _refresh_label_overlay: pixelwise / blockwise / graceful fallback routing
# --------------------------------------------------------------------------
def test_refresh_overlay_uses_pixelwise_when_data_available(window_with_session, monkeypatch):
    window = window_with_session
    blocks = _panel0_blocks(window)
    for block in blocks:
        window.block_pixel_predictions[block.block_id] = np.zeros((8, 8), dtype=np.uint8)

    calls = []
    monkeypatch.setattr(window.canvas, "set_pixel_class_overlay", lambda *a, **k: calls.append("pixel"))
    monkeypatch.setattr(window.canvas, "set_label_overlay", lambda *a, **k: calls.append("block"))

    window._refresh_label_overlay()

    assert calls == ["pixel"]


def test_refresh_overlay_falls_back_to_blockwise_when_no_pixel_data(window_with_session, monkeypatch):
    """A cache-hit reload of previously-saved predictions never repopulates
    block_pixel_predictions (only the block-level class survives a save) --
    pixelwise must not render a blank overlay in that case."""
    window = window_with_session
    assert window.prediction_render_mode == "pixelwise"
    assert window.block_pixel_predictions == {}

    calls = []
    monkeypatch.setattr(window.canvas, "set_pixel_class_overlay", lambda *a, **k: calls.append("pixel"))
    monkeypatch.setattr(window.canvas, "set_label_overlay", lambda *a, **k: calls.append("block"))

    window._refresh_label_overlay()

    assert calls == ["block"]


def test_refresh_overlay_uses_blockwise_when_mode_is_blockwise_even_with_pixel_data(
    window_with_session, monkeypatch
):
    window = window_with_session
    blocks = _panel0_blocks(window)
    for block in blocks:
        window.block_pixel_predictions[block.block_id] = np.zeros((8, 8), dtype=np.uint8)
    window.prediction_render_mode = "blockwise"

    calls = []
    monkeypatch.setattr(window.canvas, "set_pixel_class_overlay", lambda *a, **k: calls.append("pixel"))
    monkeypatch.setattr(window.canvas, "set_label_overlay", lambda *a, **k: calls.append("block"))

    window._refresh_label_overlay()

    assert calls == ["block"]


# --------------------------------------------------------------------------
# _toggle_render_mode
# --------------------------------------------------------------------------
def test_toggle_render_mode_switches_state_and_button_text(window_with_session):
    window = window_with_session

    window.render_mode_button.setChecked(True)
    window._toggle_render_mode()
    assert window.prediction_render_mode == "blockwise"
    assert "Block-wise" in window.render_mode_button.text()

    window.render_mode_button.setChecked(False)
    window._toggle_render_mode()
    assert window.prediction_render_mode == "pixelwise"
    assert "Pixel-wise" in window.render_mode_button.text()
