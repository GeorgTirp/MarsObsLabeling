"""End-to-end label/save/reopen/export checks using the real Qt event path."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rasterio
import yaml
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox
from rasterio.transform import Affine

from marslabeler.io.raster import RasterSource
from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import LabelStore
from marslabeler.model.session import Session
from marslabeler.model.export import export_coarse_geotiff
from marslabeler.ui.mainwindow import MainWindow

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_labels", ROOT / "scripts/export_labels.py")
export_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_module)
export_probe_set = export_module.export_probe_set


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def workflow(tmp_path, qapp, monkeypatch):
    """Real loader/preprocessor/canvas, asymmetric pixels, ragged edges, real classes."""
    y, x = np.indices((153, 281))
    pixels = (1000 + y * 281 + x).astype(np.uint16)
    raster_path = tmp_path / "OBS.tif"
    transform = Affine(0.5, 0.025, 100, 0.01, -0.5, 200)
    with rasterio.open(raster_path, "w", driver="GTiff", width=281, height=153,
                       count=1, dtype="uint16", transform=transform, nodata=0) as dst:
        dst.write(pixels, 1)
    config = yaml.safe_load((ROOT / "configs/app.yaml").read_text())
    config["paths"]["classes_file"] = str(ROOT / "configs/classes.yaml")
    config["paths"]["labels_dir"] = str(tmp_path / "labels")
    config["geometry"] = {"panel_size": 128, "block_size": 32}
    config["display"]["max_canvas_px"] = 256
    config["autosave"] = {"every_n_labels": 3, "every_seconds": 3600}
    config_path = tmp_path / "app.yaml"
    config_path.write_text(yaml.safe_dump(config))
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args[2]))
    windows = []
    def open_window(path=raster_path):
        win = MainWindow(config_path, match_training_gsd=False)
        win.help_shown_on_startup = True
        windows.append(win)
        win.show()
        win._load_observation(path)
        qapp.processEvents()
        assert not errors, errors
        assert win.session is not None
        return win
    win = open_window()
    yield win, pixels, raster_path, tmp_path, open_window
    for w in windows:
        if w.autosave_timer:
            w.autosave_timer.stop()
        if w.session:
            w.session.raster.close()
        w.hide()
        w.deleteLater()
    qapp.processEvents()


def press(win, key, modifier=Qt.KeyboardModifier.NoModifier):
    QTest.keyClick(win, key, modifier)
    QApplication.processEvents()


def test_close_reopen_and_export_preserve_labels_and_native_crops(workflow):
    win, pixels, source, root, open_window = workflow
    # Different panels and clipped right/bottom edges; real names include commas.
    targets = [(0, 0, Qt.Key_Q, 0), (64, 32, Qt.Key_S, 11),
               (128, 64, Qt.Key_D, 12), (256, 128, Qt.Key_K, 17)]
    expected = {}
    for x, y, key, cid in targets:
        idx = win.session.grid.block_index_at_pixel(x, y)
        win.session.move_to_block(idx)
        win._refresh_view()
        block = win.session.current_block()
        expected[block.block_id] = (x, y, block.w_px, block.h_px, cid)
        press(win, key)
    # Explicit save path, followed by a final edit saved by closing the window.
    win._autosave_session()
    idx = win.session.grid.block_index_at_pixel(0, 0)
    win.session.move_to_block(idx)
    press(win, Qt.Key_W)
    expected[source.stem + "_0_0"] = (0, 0, 32, 32, 1)
    win.close()
    resumed = open_window()
    assert resumed.session.labels.count_labeled() == len(expected)
    for bid, (*_, cid) in expected.items():
        assert resumed.session.labels.get_record(bid).class_id == cid
    output = root / "probe"
    export_probe_set(source, root / "labels/OBS.parquet", ROOT / "configs/classes.yaml", output)
    with (output / "labels.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(expected)
    assert {r["block_id"] for r in rows} == set(expected)
    for row in rows:
        assert None not in row, "class names containing commas must be CSV-quoted"
        x, y, w, h, cid = expected[row["block_id"]]
        assert int(row["class_id"]) == cid
        assert row["class_name"] == resumed.classes_scheme.get_name(cid)
        with Image.open(output / "crops" / (row["block_id"] + ".png")) as image:
            np.testing.assert_array_equal(np.asarray(image), pixels[y:y+h, x:x+w])


def test_count_autosave_runs_on_action(workflow):
    win, _, _, root, _ = workflow
    for key in (Qt.Key_Q, Qt.Key_W, Qt.Key_E):
        press(win, key)
    path = root / "labels/OBS.parquet"
    assert path.exists(), "count autosave must not wait for the timer"
    assert sum(r["status"] == "labeled" for r in pq.read_table(path).to_pylist()) == 3


@pytest.mark.parametrize("action", ["paint", "selection", "undo"])
def test_all_edit_paths_are_autosaved(workflow, action):
    win, _, _, root, _ = workflow
    win._autosave_session()
    if action == "paint":
        win.held_class_id = 11
        win._on_block_paint(0, 0, True)
        win._on_block_paint_end()
    elif action == "selection":
        win.selection_rect = (0, 0, 0, 1)
        win._fill_selection(12)
    else:
        win.session.labels.assign("OBS_0_0", 0, win.classes_scheme.get_name(0))
        win._autosave_session()
        press(win, Qt.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert win.session.label_count_since_autosave > 0
    win.session.last_save_time = 0
    win.session.last_label_time = 0
    win._maybe_autosave()
    saved = pq.read_table(root / "labels/OBS.parquet").to_pylist()
    by_id = {r["block_id"]: r for r in saved}
    expected = {"paint": 11, "selection": 12, "undo": -3}[action]
    assert by_id["OBS_0_0"]["class_id"] == expected


def test_arrow_crossing_panel_keeps_image_and_label_cursor_together(workflow):
    win, *_ = workflow
    win.session.move_to_block(win.session.grid.blocks_per_panel - 1)
    win._refresh_view()
    press(win, Qt.Key_Right)
    assert win.current_panel_idx == win.session.current_panel_idx() == 1
    press(win, Qt.Key_Left)
    assert win.current_panel_idx == win.session.current_panel_idx() == 0


def test_edit_does_not_advance_when_config_disables_it(workflow):
    win, *_ = workflow
    press(win, Qt.Key_Q)
    win.session.move_to_block(0)
    press(win, Qt.Key_W)
    assert win.session.current_block_idx == 0


def test_switching_observations_saves_previous_session(workflow):
    win, _, source, root, _ = workflow
    press(win, Qt.Key_Q)
    other = root / "OTHER.tif"
    other.write_bytes(source.read_bytes())
    win._load_observation(other)
    assert (root / "labels/OBS.parquet").exists()
    saved = pq.read_table(root / "labels/OBS.parquet").to_pylist()
    assert next(r for r in saved if r["block_id"] == "OBS_0_0")["class_id"] == 0


def test_close_save_failure_keeps_window_open(workflow, monkeypatch):
    win, *_ = workflow
    press(win, Qt.Key_Q)
    def fail(*args, **kwargs):
        raise OSError("simulated disk full")
    monkeypatch.setattr(win.session, "save_session", fail)
    event = QCloseEvent()
    win.closeEvent(event)
    assert not event.isAccepted()


def test_saved_labels_survive_missing_cursor_sidecar(workflow):
    win, _, source, root, _ = workflow
    press(win, Qt.Key_Q)
    win.session.save_session(root / "labels")
    (root / "labels/OBS.session.json").unlink()
    restored = Session.load_or_create(source, win.session.grid, win.session.config, root / "labels")
    try:
        assert restored.labels.get_record("OBS_0_0").class_id == 0
    finally:
        restored.raster.close()


def test_failed_parquet_write_preserves_previous_labels(workflow, monkeypatch):
    win, _, _, root, _ = workflow
    path = root / "labels/OBS.parquet"
    win.session.save_session(root / "labels")
    before = path.read_bytes()
    def fail(table, target, **kwargs):
        Path(target).write_bytes(b"partial write")
        raise OSError("simulated disk full")
    monkeypatch.setattr(pq, "write_table", fail)
    with pytest.raises(OSError):
        win.session.save_session(root / "labels")
    assert path.read_bytes() == before


def test_map_coordinates_and_geotiff_affine_match_source(workflow):
    win, _, _, root, _ = workflow
    grid = win.session.grid
    block = grid.get_block(5)
    assert grid.block_to_map(block) == pytest.approx(grid.transform * block.centroid_px())
    export_coarse_geotiff(win.session.labels, grid, root / "coarse.tif")
    with rasterio.open(root / "coarse.tif") as ds:
        assert ds.transform == grid.transform * Affine.scale(grid.block_size)


def test_export_rejects_wrong_image_and_bad_coordinates(workflow):
    win, _, source, root, _ = workflow
    press(win, Qt.Key_Q)
    win.session.save_session(root / "labels")
    wrong = root / "WRONG.tif"
    wrong.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="(?i)(source|observation|image)"):
        export_probe_set(wrong, root / "labels/OBS.parquet", ROOT / "configs/classes.yaml", root / "wrong")
    rows = pq.read_table(root / "labels/OBS.parquet").to_pylist()
    rows[0]["x_px"] = 9000
    pq.write_table(pa.Table.from_pylist(rows), root / "bad.parquet")
    with pytest.raises(ValueError, match="(?i)(extent|coordinates|bounds|block)"):
        export_probe_set(source, root / "bad.parquet", ROOT / "configs/classes.yaml", root / "bad")


def test_export_uint16_csv_commas_and_edge_crop(workflow):
    win, pixels, source, root, _ = workflow
    block = win.session.grid.get_block(win.session.grid.block_index_at_pixel(256, 128))
    win.session.labels.assign(block.block_id, 11, 'old class name')
    win.session.save_session(root / 'labels')
    output = root / 'probe'
    export_probe_set(source, root / 'labels/OBS.parquet', ROOT / 'configs/classes.yaml', output)
    with (output / 'labels.csv').open(newline='') as stream:
        row, = list(csv.DictReader(stream))
    assert None not in row
    assert row['class_name'] == 'Small Ripples: Discontinuous, Bedrock'
    with Image.open(output / 'crops' / (block.block_id + '.png')) as image:
        np.testing.assert_array_equal(np.asarray(image), pixels[128:, 256:])


def test_same_name_and_dimensions_different_image_is_rejected(workflow):
    win, pixels, source, root, _ = workflow
    win.session.save_session(root / 'labels')
    other = root / 'other' / source.name
    other.parent.mkdir()
    with rasterio.open(source) as src:
        profile = src.profile
    with rasterio.open(other, 'w', **profile) as dst:
        dst.write(pixels + 1, 1)
    with pytest.raises(ValueError, match='source image'):
        Session.load_or_create(other, win.session.grid, win.session.config, root / 'labels')
    with pytest.raises(ValueError, match='source image'):
        export_probe_set(other, root / 'labels/OBS.parquet', ROOT / 'configs/classes.yaml', root / 'wrong')


def test_legacy_parquet_without_sidecar_still_loads_labels(workflow):
    win, _, source, root, _ = workflow
    press(win, Qt.Key_Q)
    path = root / 'labels/OBS.parquet'
    path.parent.mkdir(exist_ok=True)
    pq.write_table(win.session.labels.to_parquet_table(), path)
    restored = Session.load_or_create(source, win.session.grid, win.session.config, root / 'labels')
    try:
        assert restored.labels.get_record('OBS_0_0').class_id == 0
    finally:
        restored.raster.close()


def test_broken_or_stale_sidecar_uses_atomic_parquet_snapshot(workflow):
    win, _, source, root, _ = workflow
    press(win, Qt.Key_Q)
    win.session.save_session(root / 'labels')
    (root / 'labels/OBS.session.json').write_text('{partially written')
    restored = Session.load_or_create(source, win.session.grid, win.session.config, root / 'labels')
    try:
        assert restored.labels.get_record('OBS_0_0').class_id == 0
        assert restored.current_block_idx == win.session.current_block_idx
    finally:
        restored.raster.close()


def test_timer_autosave_is_not_delayed_by_continuous_edits(workflow):
    win, *_ = workflow
    win.session.last_save_time = 0
    press(win, Qt.Key_Q)
    # This edit is recent, but the configured time since the last save elapsed.
    assert win.session.label_count_since_autosave == 0


def test_reexport_refuses_stale_crop_directory(workflow):
    win, _, source, root, _ = workflow
    press(win, Qt.Key_Q)
    win.session.save_session(root / 'labels')
    out = root / 'probe'
    export_probe_set(source, root / 'labels/OBS.parquet', ROOT / 'configs/classes.yaml', out)
    with pytest.raises(FileExistsError, match='not empty'):
        export_probe_set(source, root / 'labels/OBS.parquet', ROOT / 'configs/classes.yaml', out)


@pytest.mark.parametrize('corruption', ['duplicate', 'unknown_class', 'status', 'extent'])
def test_export_rejects_corrupted_label_records(workflow, corruption):
    win, _, source, root, _ = workflow
    press(win, Qt.Key_Q)
    rows = win.session.labels.to_parquet_table().to_pylist()
    if corruption == 'duplicate':
        rows.append(rows[0])
    elif corruption == 'unknown_class':
        rows[0]['class_id'] = 99
    elif corruption == 'status':
        rows[1]['class_id'] = 0
    else:
        rows[0]['w_px'] = 9000
    pq.write_table(pa.Table.from_pylist(rows), root / 'bad.parquet')
    with pytest.raises(ValueError):
        export_probe_set(source, root / 'bad.parquet', ROOT / 'configs/classes.yaml', root / 'out')
    assert not (root / 'out').exists()


def test_fresh_session_can_ignore_saved_tile_geometry(workflow):
    win, _, source, root, _ = workflow
    press(win, Qt.Key_Q)
    win.session.save_session(root / 'labels')
    g = win.session.grid
    grid = Grid(g.img_width, g.img_height, g.panel_size, 16, g.obs_id, g.transform, g.crs)
    fresh = Session.load_or_create(source, grid, win.session.config, root / 'labels', resume=False)
    try:
        assert fresh.labels.count_labeled() == 0
        assert fresh.grid.block_size == 16
    finally:
        fresh.raster.close()


def test_resume_rejects_bad_extent_instead_of_discarding_saved_labels(workflow):
    win, _, source, root, _ = workflow
    press(win, Qt.Key_Q)
    win.session.save_session(root / 'labels')
    path = root / 'labels/OBS.parquet'
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]['w_px'] += 1
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='extent'):
        Session.load_or_create(source, win.session.grid, win.session.config, root / 'labels')
    assert path.read_bytes() == before


@pytest.mark.parametrize('method', ['keyboard', 'paint', 'selection'])
def test_padding_outside_image_cannot_become_a_labeled_crop(workflow, method):
    win, *_ = workflow
    grid = win.session.grid
    idx, block = next((i, b) for i, b in enumerate(grid.iter_blocks()) if b.w_px == 0)
    win.session.move_to_block(idx)
    win._refresh_view()
    if method == 'keyboard':
        press(win, Qt.Key_Q)
    elif method == 'paint':
        win.held_class_id = 0
        win._on_block_paint(block.block_row, block.block_col, True)
    else:
        win.selection_rect = (block.block_row, block.block_col, block.block_row, block.block_col)
        win._fill_selection(0)
    assert win.session.labels.get_record(block.block_id).status != 'labeled'


def test_coarse_export_rejects_overflow_instead_of_wrapping_class_ids(workflow):
    win, _, _, root, _ = workflow
    win.session.labels.assign('OBS_0_0', 256, 'bad id')
    with pytest.raises(ValueError, match='Class id'):
        export_coarse_geotiff(win.session.labels, win.session.grid, root / 'bad.tif')
