"""Tests for MainWindow's predictions-mode nodata-skip logic (no model needed).

Deliberately bypasses _load_observation() (which drives the modal preprocessing
dialog + QThread) -- constructs Session/Grid/LabelStore directly and assigns them
onto a bare MainWindow, the same way test_session.py tests Session in isolation.
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
def window_with_session(qapp, synthetic_geotiff):
    """A predictions-mode MainWindow with a session assigned directly (no dialogs)."""
    window = MainWindow(Path("configs/app.yaml"), predictions_mode=True)

    raster = RasterSource(synthetic_geotiff)
    raster.open()
    grid = Grid(4096, 4096, 4096, 512, "TEST_OBS", Affine.identity())
    labels = LabelStore(grid, "test_user")
    window.session = Session(raster, grid, labels, window.config.to_dict())

    yield window
    raster.close()


def test_retire_high_nodata_blocks_marks_only_blocks_above_threshold(window_with_session):
    window = window_with_session
    window.config.inference.nodata_skip_threshold = 0.33
    blocks = list(window.session.grid.iter_blocks())
    keep, skip = blocks[0].block_id, blocks[1].block_id
    window.skip_decisions[keep] = {"nodata_fraction": 0.10, "should_skip": False}
    window.skip_decisions[skip] = {"nodata_fraction": 0.90, "should_skip": True}

    retired = window._retire_high_nodata_blocks()

    assert retired == 1
    assert window.session.labels.get_record(skip).status == "nodata"
    assert window.session.labels.get_record(keep).status != "nodata"


def test_retire_high_nodata_blocks_respects_configured_threshold(window_with_session):
    window = window_with_session
    window.config.inference.nodata_skip_threshold = 0.5
    block_id = next(iter(window.session.grid.iter_blocks())).block_id
    window.skip_decisions[block_id] = {"nodata_fraction": 0.40, "should_skip": False}

    retired = window._retire_high_nodata_blocks()

    assert retired == 0
    assert window.session.labels.get_record(block_id).status != "nodata"


def test_retire_high_nodata_blocks_is_idempotent(window_with_session):
    window = window_with_session
    window.config.inference.nodata_skip_threshold = 0.33
    block_id = next(iter(window.session.grid.iter_blocks())).block_id
    window.skip_decisions[block_id] = {"nodata_fraction": 0.99, "should_skip": True}

    first = window._retire_high_nodata_blocks()
    second = window._retire_high_nodata_blocks()

    assert first == 1
    assert second == 0  # already nodata -- not re-counted
    assert window.session.labels.get_record(block_id).status == "nodata"


def test_retire_high_nodata_blocks_missing_skip_decision_defaults_to_kept(window_with_session):
    """A block with no entry in skip_decisions (e.g. preprocessing skipped it) is
    treated as 0% nodata -- kept, not silently retired."""
    window = window_with_session
    window.skip_decisions.clear()

    retired = window._retire_high_nodata_blocks()

    assert retired == 0


def test_inference_nodata_skip_threshold_default(qapp):
    window = MainWindow(Path("configs/app.yaml"), predictions_mode=True)
    assert window.config.inference.nodata_skip_threshold == pytest.approx(0.33)


def test_summary_refresh_preserves_reviewed_label_even_above_nodata_threshold(
    window_with_session, monkeypatch, tmp_path,
):
    from types import SimpleNamespace
    from PySide6.QtWidgets import QDialog
    import marslabeler.ui.predictdialog as module

    window = window_with_session
    window.classes_scheme = load_classes(window.config.paths.classes_file)
    block = next(window.session.grid.iter_blocks())
    window.session.labels.assign(block.block_id, 1, window.classes_scheme.get_name(1))
    before = window.session.labels.get_record(block.block_id).__dict__.copy()
    window.skip_decisions[block.block_id] = {"nodata_fraction": .9}

    class Dialog:
        worker = SimpleNamespace(wait=lambda: None)
        is_block_classifier = True
        num_pixel_classes = 14
        def __init__(self, *args): pass
        def start(self): pass
        def exec(self): return QDialog.DialogCode.Accepted
        def get_predictions(self): return {block.block_id: 2}
        def get_confidence(self): return {block.block_id: .9}
        def get_pixel_predictions(self): return {}
        def get_local_npca(self): return {}
        def get_block_features(self): return None

    monkeypatch.setattr(module, "PredictDialog", Dialog)
    monkeypatch.setattr(window, "_refresh_view", lambda: None)
    monkeypatch.setattr(window, "_refresh_history", lambda: None)
    window._run_prediction(tmp_path / "model.pt", "test", preserve_labels=True)
    assert window.session.labels.get_record(block.block_id).__dict__ == before
    assert window.block_confidence == {block.block_id: .9}
    assert not window.block_uncertainty and not window.block_laplace


@pytest.fixture
def window_with_multi_panel_session(qapp, synthetic_geotiff):
    """A predictions-mode MainWindow with a 2x2-panel grid, predictions seeded so
    each panel has a distinct, unambiguous majority class -- for overview tests."""
    window = MainWindow(Path("configs/app.yaml"), predictions_mode=True)

    raster = RasterSource(synthetic_geotiff)
    raster.open()
    grid = Grid(4096, 4096, 2048, 512, "TEST_OBS", Affine.identity())  # 2x2 panels, 16 blocks each
    labels = LabelStore(grid, "test_user")
    window.session = Session(raster, grid, labels, window.config.to_dict())
    window.classes_scheme = load_classes(window.config.paths.classes_file)

    # Panel p gets majority class p (panel 0 -> class 0, panel 1 -> class 1, ...),
    # one dissenting block per panel so "majority" is actually exercised.
    for p in range(grid.num_panels):
        blocks = grid.get_panel_blocks(p)
        for i, block in enumerate(blocks):
            class_id = p if i != 0 else (p + 1) % grid.num_panels
            window.session.labels.assign(block.block_id, class_id, f"Class {class_id}")

    yield window
    raster.close()


def test_render_overview_predictions_shows_majority_class_not_done_tint(window_with_multi_panel_session):
    """Regression test: predictions mode must not paint every panel with the
    labeling-mode 'done' tint (every panel is trivially fully-labeled the instant
    inference finishes, so the old behavior painted the whole overview one color)."""
    window = window_with_multi_panel_session
    captured = {}
    window.canvas.set_label_overlay = lambda cell_data, colors: captured.update(
        cell_data=cell_data.copy(), colors=colors
    )

    window._render_overview_predictions(panels_across=2, panels_down=2)

    cell_data = captured["cell_data"]
    assert window.DONE_PANEL_TINT not in cell_data
    # Panel p (row p//2, col p%2) should show class p as its majority.
    for p in range(4):
        pr, pc = divmod(p, 2)
        assert cell_data[pr, pc] == p
    # Not all four panels collapsed to the same value (i.e. real per-panel signal).
    assert len(set(cell_data.ravel().tolist())) == 4


def test_render_overview_predictions_uncertainty_layer_draws_block_scores(window_with_multi_panel_session):
    """Per-block scores (a block classifier's maps) are drawn at their blocks."""
    window = window_with_multi_panel_session
    window.display_layer = "uncertainty"
    window.mahalanobis_maximum = 2.0
    grid = window.session.grid
    # Distinct uncertainty scores per panel so their positions are distinguishable.
    for p in range(grid.num_panels):
        for block in grid.get_panel_blocks(p):
            window.block_uncertainty[block.block_id] = p * 0.1
    window.canvas.canvas_width = window.canvas.canvas_height = 4

    captured = {}
    window.canvas.set_pixel_scalar_overlay = lambda values, **kw: captured.update(values=values.copy(), **kw)

    window._render_overview_predictions(panels_across=2, panels_down=2)

    expected = np.repeat(np.repeat([[0.0, 0.1], [0.2, 0.3]], 2, axis=0), 2, axis=1)
    np.testing.assert_allclose(captured["values"], expected)
    assert captured["maximum"] == 2.0


def test_forward_logits_is_a_tensor_for_an_ig_trained_checkpoint():
    """A checkpoint trained with the Stage-3 IG aux head must still predict.

    `LightweightSegmentationDecoder` exposes the auxiliary interpretive-group
    logits behind an opt-in `return_ig=True`, so the default forward stays a bare
    DC-logits tensor whether or not `head_ig` exists. This pins that: inference
    calls the default form and immediately does `.argmax(dim=1)`, which a tuple
    or dict would break.
    """
    import numpy as np

    torch = pytest.importorskip("torch")  # only ships with the optional [infer] extra

    from marslabeler.inference.modelio import ModelBundle, _forward_logits
    from vision_backend.training.builders import build_simmim_segmentation_model

    config = dict(
        in_channels=1, global_base_grid=32, drop_path=0.0, num_classes=14,
        decoder_channels=32, num_classes_ig=5, model_kind="simmim",
    )
    model = build_simmim_segmentation_model(dict(config)).eval()
    assert model.decoder.head_ig is not None, "fixture must actually have the IG head"

    bundle = ModelBundle(
        model=model, model_kind="simmim", num_classes=14, required_stride=256,
        needs_context=False, device=torch.device("cpu"),
        checkpoint_path=Path("unused.pt"), raw_config={},
    )
    batch = np.zeros((1, 256, 256), dtype=np.uint8)
    with torch.no_grad():
        logits = _forward_logits(bundle, batch, None, None)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (1, 14, 256, 256)  # DC classes, not the 5 IG groups
    assert logits.argmax(dim=1).shape == (1, 256, 256)
