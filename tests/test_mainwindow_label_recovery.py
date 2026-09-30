"""Observation opening can recover safely from unusable saved labels."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rasterio
import yaml
from PySide6.QtWidgets import QApplication, QMessageBox
from rasterio.transform import Affine

from marslabeler.config import load_config
from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import METADATA_KEY
from marslabeler.model.session import Session
from marslabeler.ui.mainwindow import MainWindow


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def recovery_workspace(tmp_path, qapp, monkeypatch):
    """Real raster, saved labels and preprocessing; intercept only modal choices."""
    y, x = np.indices((91, 117))
    pixels = (1000 + y * 117 + x).astype(np.uint16)
    transform = Affine(0.5, 0.025, 100, 0.01, -0.5, 200)

    def write_raster(path, data=pixels):
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path, "w", driver="GTiff", width=117, height=91, count=1,
            dtype="uint16", transform=transform, nodata=0,
        ) as dst:
            dst.write(data, 1)
        return path

    source = write_raster(tmp_path / "OBS.tif")
    labels_dir = tmp_path / "labels"
    predictions_dir = tmp_path / "predictions"
    config = yaml.safe_load((ROOT / "configs/app.yaml").read_text())
    config["paths"].update(
        classes_file=str(ROOT / "configs/classes.yaml"),
        labels_dir=str(labels_dir), predictions_dir=str(predictions_dir),
    )
    config["geometry"] = {"panel_size": 64, "block_size": 16}
    config["display"]["max_canvas_px"] = 128
    config["autosave"] = {"every_n_labels": 1000, "every_seconds": 3600}
    config_path = tmp_path / "app.yaml"
    config_path.write_text(yaml.safe_dump(config))
    grid = Grid(117, 91, 64, 16, source.stem, transform)
    saved = Session.load_or_create(source, grid, load_config(config_path).to_dict(), labels_dir)
    try:
        saved.labels.assign("OBS_0_0", 0, "saved class")
        saved.save_session(labels_dir)
    finally:
        saved.raster.close()

    errors, dialogs, windows = [], [], []
    choice = {"text": "Open without saved labels"}

    def choose_button(dialog):
        texts = [button.text() for button in dialog.buttons()]
        dialogs.append({"title": dialog.windowTitle(), "text": dialog.text(), "buttons": texts})
        assert dialog.icon() == QMessageBox.Icon.Warning
        assert "Open without saved labels" in texts
        assert dialog.button(QMessageBox.StandardButton.Cancel) is not None
        button = next(button for button in dialog.buttons() if button.text() == choice["text"])
        button.click()
        return dialog.result()

    monkeypatch.setattr(QMessageBox, "exec", choose_button)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args[2]))
    monkeypatch.setattr(MainWindow, "_show_help", lambda self: None)

    def new_window(**kwargs):
        win = MainWindow(config_path, match_training_gsd=False, **kwargs)
        win.help_shown_on_startup = True
        windows.append(win)
        win.show()
        qapp.processEvents()
        return win

    workspace = SimpleNamespace(
        root=tmp_path, source=source, labels_dir=labels_dir, predictions_dir=predictions_dir,
        parquet=labels_dir / "OBS.parquet", sidecar=labels_dir / "OBS.session.json",
        new_window=new_window, write_raster=write_raster, pixels=pixels,
        errors=errors, dialogs=dialogs, choice=choice,
    )
    yield workspace
    for win in windows:
        if win.autosave_timer:
            win.autosave_timer.stop()
        if win.session:
            win.session.raster.close()
        win.hide()
        win.deleteLater()
    qapp.processEvents()
    assert not errors, errors


def corrupt_saved_labels(workspace, corruption):
    table = pq.read_table(workspace.parquet)
    if corruption == "metadata":
        table = table.replace_schema_metadata({METADATA_KEY: b"{broken metadata"})
    else:
        rows = table.to_pylist()
        rows[0]["w_px"] += 1
        table = pa.Table.from_pylist(rows, schema=table.schema)
    pq.write_table(table, workspace.parquet)


def saved_bytes(workspace):
    return {path: path.read_bytes() for path in (workspace.parquet, workspace.sidecar)}


def assert_originals_unchanged(originals):
    assert {path: path.read_bytes() for path in originals} == originals


def test_valid_saved_labels_resume_without_a_warning(recovery_workspace):
    ws = recovery_workspace
    win = ws.new_window()

    assert win._load_observation(ws.source) is True

    assert win.session.labels.get_record("OBS_0_0").class_id == 0
    assert win.session.labels.count_labeled() == 1
    assert win.labels_dir == ws.labels_dir
    assert not ws.dialogs


@pytest.mark.parametrize("corruption", ["metadata", "extent"])
def test_open_without_saved_labels_preserves_original_files_on_save(recovery_workspace, corruption):
    ws = recovery_workspace
    corrupt_saved_labels(ws, corruption)
    originals = saved_bytes(ws)
    win = ws.new_window()

    assert win._load_observation(ws.source) is True

    assert len(ws.dialogs) == 1
    assert win.session.labels.count_labeled() == 0
    assert win.labels_dir != ws.labels_dir
    assert win.labels_dir.is_relative_to(ws.labels_dir)
    assert win.workspace_save_button.isEnabled()
    win.session.labels.assign("OBS_0_0", 1, win.classes_scheme.get_name(1))
    assert win._autosave_session() is True
    new_rows = pq.read_table(win.labels_dir / "OBS.parquet").to_pylist()
    assert next(row for row in new_rows if row["block_id"] == "OBS_0_0")["class_id"] == 1
    assert_originals_unchanged(originals)


def test_repeated_recovery_uses_distinct_save_folders(recovery_workspace):
    ws = recovery_workspace
    corrupt_saved_labels(ws, "metadata")
    originals = saved_bytes(ws)
    first, second = ws.new_window(), ws.new_window()
    assert first._load_observation(ws.source) is True
    assert first._autosave_session() is True
    first_saved = (first.labels_dir / "OBS.parquet").read_bytes()

    assert second._load_observation(ws.source) is True
    assert second.labels_dir != first.labels_dir
    assert second.labels_dir.is_relative_to(ws.labels_dir)
    assert second._autosave_session() is True

    assert (first.labels_dir / "OBS.parquet").read_bytes() == first_saved
    assert_originals_unchanged(originals)


@pytest.mark.parametrize("already_open", [False, True])
def test_cancel_recovery_keeps_the_previous_workspace(recovery_workspace, already_open):
    ws = recovery_workspace
    win = ws.new_window()
    if already_open:
        current = ws.write_raster(ws.root / "CURRENT.tif")
        assert win._load_observation(current) is True
        win.session.labels.assign("CURRENT_0_0", 1, win.classes_scheme.get_name(1))
    previous_session, previous_controller = win.session, win.controller
    corrupt_saved_labels(ws, "extent")
    originals = saved_bytes(ws)
    ws.choice["text"] = "Cancel"

    assert win._load_observation(ws.source) is False

    assert len(ws.dialogs) == 1
    assert win.session is previous_session
    assert win.controller is previous_controller
    assert win.labels_dir == ws.labels_dir
    assert_originals_unchanged(originals)
    if already_open:
        np.testing.assert_array_equal(
            win.session.raster.read_window(0, 0, 16, 16, 16, 16), ws.pixels[:16, :16],
        )
        assert win.session.labels.get_record("CURRENT_0_0").class_id == 1
        assert win._autosave_session() is True
        assert win.workspace_save_button.isEnabled()
    else:
        assert not win.workspace_save_button.isEnabled()


def test_source_fingerprint_mismatch_can_open_a_fresh_session(recovery_workspace):
    ws = recovery_workspace
    different_source = ws.write_raster(ws.root / "other" / "OBS.tif", ws.pixels + 1)
    originals = saved_bytes(ws)
    win = ws.new_window()

    assert win._load_observation(different_source) is True

    assert len(ws.dialogs) == 1
    assert win.session.raster.path == different_source
    assert win.session.labels.count_labeled() == 0
    assert win.labels_dir != ws.labels_dir
    assert win.labels_dir.is_relative_to(ws.labels_dir)
    assert win._autosave_session() is True
    assert_originals_unchanged(originals)


def test_fresh_predictions_skip_corrupt_cache_metadata(recovery_workspace, monkeypatch):
    ws = recovery_workspace
    model = ws.root / "checkpoint.pt"
    model.write_bytes(b"the prediction runner is replaced in this test")
    cache_dir = ws.predictions_dir / model.stem
    cache_dir.mkdir(parents=True)
    (cache_dir / "OBS.parquet").write_bytes(b"not a readable parquet file")
    win = ws.new_window(predictions_mode=True, ignore_cached_predictions=True)
    predictions = []
    metadata_reads = []

    def reject_metadata_read(*args, **kwargs):
        metadata_reads.append(args)
        raise AssertionError("--fresh must not inspect saved prediction metadata")

    monkeypatch.setattr(Session, "read_saved_metadata", reject_metadata_read)
    monkeypatch.setattr(win, "_resolve_gsd_scaling", lambda raster: 1.0)
    monkeypatch.setattr(win, "_try_load_npca_gallery", lambda path: None)
    monkeypatch.setattr(win, "_run_prediction", lambda *args: predictions.append(args))

    win.load_for_inference(ws.source, model)

    assert not metadata_reads
    assert not ws.dialogs
    assert win.session is not None
    assert win.session.labels.count_labeled() == 0
    assert len(predictions) == 1
    assert predictions[0][0] == model


def cached_prediction_workspace(workspace, monkeypatch):
    """A real prediction cache with matching provenance and a tiny fake runner."""
    model = workspace.root / "checkpoint.pt"
    model.write_bytes(b"the prediction runner is replaced in this test")
    model_sig = MainWindow._model_signature(model)
    cache_dir = workspace.predictions_dir / model.stem
    cache_dir.mkdir(parents=True)
    parquet = cache_dir / "OBS.parquet"
    sidecar = cache_dir / "OBS.session.json"
    table = pq.read_table(workspace.parquet)
    metadata = json.loads(table.schema.metadata[METADATA_KEY])
    metadata.update(model_sig=model_sig, model_stem=model.stem)
    table = table.replace_schema_metadata({METADATA_KEY: json.dumps(metadata).encode()})
    pq.write_table(table, parquet)
    sidecar.write_text(json.dumps(metadata))
    win = workspace.new_window(predictions_mode=True)
    predictions = []

    def run_prediction(path, signature):
        predictions.append((path, signature))
        assert win.session.labels.count_labeled() == 0
        win._model_sig = signature
        win.session.labels.assign("OBS_0_0", 1, win.classes_scheme.get_name(1))

    def unexpected_information(*args):
        pytest.fail(f"Unexpected informational dialog: {args[1]}")

    monkeypatch.setattr(win, "_resolve_gsd_scaling", lambda raster: 1.0)
    monkeypatch.setattr(win, "_try_load_npca_gallery", lambda path: None)
    monkeypatch.setattr(win, "_run_prediction", run_prediction)
    monkeypatch.setattr(QMessageBox, "information", unexpected_information)
    return SimpleNamespace(
        model=model, model_sig=model_sig, cache_dir=cache_dir, parquet=parquet,
        sidecar=sidecar, win=win, predictions=predictions,
    )


@pytest.mark.parametrize("corruption", ["metadata", "extent"])
def test_corrupt_prediction_cache_recovers_and_saves_separately(
    recovery_workspace, monkeypatch, corruption,
):
    ws = recovery_workspace
    cache = cached_prediction_workspace(ws, monkeypatch)
    corrupt_saved_labels(cache, corruption)
    originals = saved_bytes(cache)

    cache.win.load_for_inference(ws.source, cache.model)

    assert len(ws.dialogs) == 1
    assert cache.predictions == [(cache.model, cache.model_sig)]
    assert cache.win.labels_dir != cache.cache_dir
    assert cache.win.labels_dir.is_relative_to(cache.cache_dir)
    assert cache.win.save_predictions_button.isEnabled()
    cache.win._save_predictions()
    saved_path = cache.win.labels_dir / "OBS.parquet"
    new_rows = pq.read_table(saved_path).to_pylist()
    assert next(row for row in new_rows if row["block_id"] == "OBS_0_0")["class_id"] == 1
    assert Session.read_saved_metadata(cache.win.labels_dir, "OBS")["model_sig"] == cache.model_sig
    assert_originals_unchanged(originals)


def test_matching_prediction_cache_resumes_without_inference(recovery_workspace, monkeypatch):
    ws = recovery_workspace
    cache = cached_prediction_workspace(ws, monkeypatch)
    originals = saved_bytes(cache)

    cache.win.load_for_inference(ws.source, cache.model)

    assert not ws.dialogs
    assert not cache.predictions
    assert cache.win.session is not None
    assert cache.win.session.labels.get_record("OBS_0_0").class_id == 0
    assert cache.win.session.labels.count_labeled() == 1
    assert cache.win.labels_dir == cache.cache_dir
    assert cache.win._model_sig == cache.model_sig
    assert cache.win.save_predictions_button.isEnabled()
    assert_originals_unchanged(originals)


def test_cancel_prediction_recovery_preserves_previous_model_and_save_destination(
    recovery_workspace, monkeypatch,
):
    ws = recovery_workspace
    cache = cached_prediction_workspace(ws, monkeypatch)
    win = cache.win
    win.load_for_inference(ws.source, cache.model)
    win.session.labels.assign("OBS_0_0", 1, win.classes_scheme.get_name(1))
    win.display_layer = "uncertainty"
    win.block_confidence = {"OBS_0_0": 0.8}
    win.block_uncertainty = {"OBS_0_0": 0.2}
    win.npca_gallery = {"previous model": []}
    win.local_npca_gallery = {"previous observation": []}
    win.block_pixel_predictions = {"OBS_0_0": np.ones((16, 16), dtype=np.uint8)}
    win.prediction_render_mode = "blockwise"
    win.gsd_ratio = 1.5
    for button in (win.uncertainty_button, win.render_mode_button):
        button.blockSignals(True)
        button.setEnabled(True)
        button.setChecked(True)
        button.blockSignals(False)
    original_title = win.windowTitle()
    original_labeler = win.config.labeler
    previous_state = {
        name: getattr(win, name) for name in (
            "session", "controller", "classes_scheme", "block_confidence",
            "block_uncertainty", "npca_gallery", "local_npca_gallery", "block_pixel_predictions",
        )
    }
    other_model = ws.root / "other-checkpoint.pt"
    other_model.write_bytes(b"different checkpoint")
    other_dir = ws.predictions_dir / other_model.stem
    other_dir.mkdir()
    other_cache = SimpleNamespace(
        parquet=other_dir / "OBS.parquet", sidecar=other_dir / "OBS.session.json",
    )
    other_cache.parquet.write_bytes(cache.parquet.read_bytes())
    other_cache.sidecar.write_bytes(cache.sidecar.read_bytes())
    corrupt_saved_labels(other_cache, "extent")
    originals = saved_bytes(other_cache)
    ws.choice["text"] = "Cancel"

    win.load_for_inference(ws.source, other_model)

    assert len(ws.dialogs) == 1
    assert not cache.predictions
    for name, original in previous_state.items():
        assert getattr(win, name) is original
    assert win.model_path == cache.model
    assert win._model_sig == cache.model_sig
    assert win.labels_dir == cache.cache_dir
    assert win.config.labeler == original_labeler
    assert win.windowTitle() == original_title
    assert win.gsd_ratio == 1.5
    assert win.display_layer == "uncertainty"
    assert win.prediction_render_mode == "blockwise"
    for button in (win.uncertainty_button, win.render_mode_button):
        assert button.isChecked()
        assert button.isEnabled()
    win._save_predictions()
    rows = pq.read_table(cache.parquet).to_pylist()
    assert next(row for row in rows if row["block_id"] == "OBS_0_0")["class_id"] == 1
    metadata = Session.read_saved_metadata(cache.cache_dir, "OBS")
    assert metadata["model_sig"] == cache.model_sig
    assert metadata["model_stem"] == cache.model.stem
    assert_originals_unchanged(originals)
