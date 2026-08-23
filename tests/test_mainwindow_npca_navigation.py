"""Tests for MainWindow._on_npca_thumbnail_clicked: click-to-navigate from a
Neural-PCA gallery thumbnail (Class Summary window) to its source block.

Bypasses _load_observation()'s real (heavy: rasterio + modal PreprocessDialog)
implementation via monkeypatching, the same way test_mainwindow_inference.py
avoids driving it for other tests -- this file is about the navigation logic
around resolve_training_imagery_path/_load_observation, not about
_load_observation itself.
"""

from pathlib import Path

import pytest
from rasterio.transform import Affine

from PySide6.QtWidgets import QApplication, QMessageBox

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
    window = MainWindow(tmp_config_dir / "app.yaml", predictions_mode=True)
    window.classes_scheme = load_classes(tmp_config_dir / "classes.yaml")
    window.model_path = Path("fake_checkpoint.pt")  # never actually read (resolve is mocked)

    raster = RasterSource(synthetic_geotiff)
    raster.open()
    grid = Grid(4096, 4096, 16, 8, "TEST_OBS", Affine.identity())  # 2x2 blocks/panel
    labels = LabelStore(grid, "test_user")
    window.session = Session(raster, grid, labels, window.config.to_dict())
    window.current_panel_idx = 0

    yield window
    raster.close()


def _no_dialog(monkeypatch):
    """QMessageBox.information pops a real modal box under a real display;
    under offscreen it's harmless but still worth silencing/asserting on."""
    calls = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: calls.append(a))
    return calls


# --------------------------------------------------------------------------
# Failure paths: informative message, no navigation, no crash
# --------------------------------------------------------------------------
def test_unparseable_source_id_shows_message_and_does_not_navigate(window_with_session, monkeypatch):
    window = window_with_session
    calls = _no_dialog(monkeypatch)
    before_idx = window.session.current_block_idx

    window._on_npca_thumbnail_clicked("not_a_valid_source_id")

    assert len(calls) == 1
    assert window.session.current_block_idx == before_idx


def test_no_model_path_shows_message_and_does_not_navigate(window_with_session, monkeypatch):
    window = window_with_session
    window.model_path = None
    calls = _no_dialog(monkeypatch)
    before_idx = window.session.current_block_idx

    window._on_npca_thumbnail_clicked("mosaic_0_0")

    assert len(calls) == 1
    assert window.session.current_block_idx == before_idx


def test_unresolvable_imagery_path_shows_message_and_does_not_navigate(window_with_session, monkeypatch):
    window = window_with_session
    monkeypatch.setattr(
        "marslabeler.inference.modelio.resolve_training_imagery_path", lambda *a, **k: None
    )
    calls = _no_dialog(monkeypatch)
    before_idx = window.session.current_block_idx

    window._on_npca_thumbnail_clicked("mosaic_0_0")

    assert len(calls) == 1
    assert window.session.current_block_idx == before_idx


def test_stem_mismatch_shows_message_and_does_not_navigate(window_with_session, monkeypatch, tmp_path):
    """resolve_training_imagery_path found *a* file, but it isn't the one
    source_id's stem actually names -- must not silently jump to the wrong file."""
    window = window_with_session
    decoy = tmp_path / "totally_different_mosaic.tif"
    decoy.write_bytes(b"fake")
    monkeypatch.setattr(
        "marslabeler.inference.modelio.resolve_training_imagery_path", lambda *a, **k: decoy
    )
    calls = _no_dialog(monkeypatch)
    before_idx = window.session.current_block_idx

    window._on_npca_thumbnail_clicked("expected_mosaic_0_0")

    assert len(calls) == 1
    assert window.session.current_block_idx == before_idx


# --------------------------------------------------------------------------
# Happy paths
# --------------------------------------------------------------------------
def test_already_open_image_navigates_without_reloading(window_with_session, monkeypatch):
    """The gallery's source mosaic is already the currently-open session --
    must navigate directly, never call _load_observation (which would be a
    pointless reload of what's already loaded)."""
    window = window_with_session
    current_path = window.session.raster.path
    monkeypatch.setattr(
        "marslabeler.inference.modelio.resolve_training_imagery_path",
        lambda *a, **k: current_path,
    )
    load_calls = []
    monkeypatch.setattr(window, "_load_observation", lambda p: load_calls.append(p))

    # Block at pixel (9, 1): block_size=8 -> local_block_col=9//8=1,
    # local_block_row=1//8=0 -> local_idx=0*2+1=1 (blocks_per_panel_col=2) ->
    # global block_idx = panel_idx(0)*blocks_per_panel(4) + 1 = 1.
    source_id = f"{current_path.stem}_9_1"
    window._on_npca_thumbnail_clicked(source_id)

    assert load_calls == []  # never reloaded
    assert window.session.current_block_idx == 1
    assert window.view_mode == "panel"
    assert window.canvas.zoom == 4


def test_different_image_reloads_then_navigates(window_with_session, monkeypatch, tmp_path):
    """A different mosaic than what's open -- must call _load_observation
    with the resolved path, then navigate within whatever session results."""
    window = window_with_session
    other_tif = tmp_path / "other_mosaic.tif"
    other_tif.write_bytes(b"fake")  # never actually opened -- _load_observation is stubbed
    monkeypatch.setattr(
        "marslabeler.inference.modelio.resolve_training_imagery_path",
        lambda *a, **k: other_tif,
    )

    load_calls = []
    new_grid = Grid(4096, 4096, 16, 8, "other_mosaic", Affine.identity())
    new_labels = LabelStore(new_grid, "test_user")

    def fake_load_observation(path):
        load_calls.append(path)
        # Mimic _load_observation's real effect: a fresh session for the new image.
        window.session = Session(window.session.raster, new_grid, new_labels, window.config.to_dict())

    monkeypatch.setattr(window, "_load_observation", fake_load_observation)

    source_id = "other_mosaic_9_1"
    window._on_npca_thumbnail_clicked(source_id)

    assert load_calls == [other_tif]
    assert window.session.current_block_idx == 1  # same layout/arithmetic as the "already open" test
    assert window.session.grid is new_grid


def test_load_observation_failure_leaves_no_session_and_returns_cleanly(
    window_with_session, monkeypatch, tmp_path
):
    """_load_observation can fail/be cancelled (leaves self.session None,
    per its own contract) -- must not crash trying to navigate into it."""
    window = window_with_session
    other_tif = tmp_path / "other_mosaic.tif"
    other_tif.write_bytes(b"fake")
    monkeypatch.setattr(
        "marslabeler.inference.modelio.resolve_training_imagery_path",
        lambda *a, **k: other_tif,
    )

    def failing_load_observation(path):
        window.session = None  # matches _load_observation's own cancel/failure contract

    monkeypatch.setattr(window, "_load_observation", failing_load_observation)

    window._on_npca_thumbnail_clicked("other_mosaic_9_1")  # must not raise

    assert window.session is None
