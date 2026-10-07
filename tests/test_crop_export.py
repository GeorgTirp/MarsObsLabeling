"""Real GUI export, portable loading, corruption detection, and training smoke tests."""

import csv
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
import rasterio
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox
from rasterio.transform import from_origin

from marslabeler.classes import load_classes
from marslabeler.io.raster import RasterSource
from marslabeler.model.cropexport import ExportCancelled, export_probe_set
from marslabeler.model.cropdataset import BlockCropDataset
from marslabeler.model.dataset_integrity import file_sha256, write_integrity
from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import LabelStore
from marslabeler.model.session import Session
from tests.test_labeling_workflow import ROOT, press, qapp, workflow


@pytest.fixture(params=["uint8", "uint16"])
def crop_source(tmp_path, request):
    y, x = np.indices((93, 151))
    values = 1 + (y * 151 + x) % 253 if request.param == "uint8" else 1000 + y * 151 + x
    pixels = values.astype(request.param)
    pixels[5, 6] = 0
    source = tmp_path / "OBS.tif"
    with rasterio.open(source, "w", driver="GTiff", width=151, height=93,
                       count=1, dtype=request.param, nodata=0,
                       transform=from_origin(100, 200, .5, .5)) as dst:
        dst.write(pixels, 1)
    raster = RasterSource(source)
    raster.open()
    grid = Grid(151, 93, 64, 32, "OBS", raster.transform, raster.crs)
    labels = LabelStore(grid, "test scientist")
    classes = load_classes(ROOT / "configs/classes.yaml")
    expected = {}
    for x, y, cid in [(0, 0, 0), (32, 0, 1), (64, 32, 11), (128, 64, 17)]:
        block = grid.get_block(grid.block_index_at_pixel(x, y))
        labels.assign(block.block_id, cid, classes.get_name(cid))
        expected[block.block_id] = (x, y, block.w_px, block.h_px, cid)
    labels.assign(grid.get_block(grid.block_index_at_pixel(0, 32)).block_id, -1, "Abstain")
    labels.set_nodata_bulk([grid.get_block(grid.block_index_at_pixel(32, 32)).block_id])
    session = Session(raster, grid, labels, {})
    session.save_session(tmp_path / "session")
    raster.close()
    return source, tmp_path / "session/OBS.parquet", pixels, expected


@pytest.fixture
def exported(crop_source, tmp_path):
    source, parquet, pixels, expected = crop_source
    output = tmp_path / "dataset"
    count = export_probe_set(source, parquet, ROOT / "configs/classes.yaml", output)
    assert count == len(expected)
    return output, pixels, expected


def test_all_pairs_match_source_pixels_and_labels_after_folder_move(exported, tmp_path):
    output, pixels, expected = exported
    relocated = tmp_path / "copied-to-training-machine"
    shutil.copytree(output, relocated)
    # The original imagery and original export are unavailable to the loader.
    (tmp_path / "OBS.tif").rename(tmp_path / "source-unavailable.tif")
    dataset = BlockCropDataset(relocated)
    assert dataset.validate()["count"] == len(expected)
    assert dataset.num_classes == 18
    assert dataset.class_to_target == dict(zip(range(18), range(18)))
    for row in dataset.records:
        x, y, w, h, cid = expected[row["block_id"]]
        assert int(row["class_id"]) == int(row["target_index"]) == cid
        with Image.open(relocated / "crops" / row["crop_filename"]) as crop:
            raw = np.array(crop)
        assert raw.dtype == pixels.dtype
        np.testing.assert_array_equal(raw, pixels[y:y+h, x:x+w])
    result = subprocess.run(
        [sys.executable, "-m", "marslabeler.model.cropdataset", str(relocated)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Verified 4 crop/label pairs" in result.stdout


def test_shuffled_torch_batch_preserves_pairing_padding_and_trains(exported):
    torch = pytest.importorskip("torch")
    output, pixels, expected = exported
    dataset = BlockCropDataset(output)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True,
                                         generator=torch.Generator().manual_seed(31))
    batch = next(iter(loader))
    assert batch["image"].shape == (4, 1, 32, 32)
    assert batch["image"].dtype == torch.float32
    assert batch["target"].dtype == torch.int64
    assert batch["image"].min() >= -1 and batch["image"].max() <= 1
    for i, bid in enumerate(batch["block_id"]):
        x, y, w, h, cid = expected[bid]
        assert batch["target"][i].item() == batch["class_id"][i].item() == cid
        assert batch["original_size"][i].tolist() == [h, w]
        assert not batch["valid_mask"][i, h:, :].any()
        assert not batch["valid_mask"][i, :, w:].any()
        raw = pixels[y:y+h, x:x+w]
        bounds = dataset.metadata["preprocessing"]["bounds"]
        if bounds:
            low, high = bounds
            converted = np.rint((np.clip(raw.astype(float), low, high) - low) / (high-low) * 255)
        else:
            converted = raw
        normalized = converted.astype(np.float32) / 127.5 - 1
        normalized[raw == 0] = -1
        np.testing.assert_allclose(batch["image"][i, 0, :h, :w].numpy(), normalized)
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3, padding=1), torch.nn.ReLU(),
                                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                torch.nn.Linear(4, dataset.num_classes))
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    before = model[-1].weight.detach().clone()
    loss = torch.nn.functional.cross_entropy(model(batch["image"]), batch["target"])
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model[-1].weight)
    with pytest.raises(ValueError, match="outside a 14-class head"):
        BlockCropDataset(output, num_classes=14)


@pytest.mark.parametrize("damage", ["missing", "truncated", "swapped", "csv_label", "classes", "extra"])
def test_damaged_or_mixed_datasets_fail_validation(exported, damage):
    output, _, _ = exported
    crops = sorted((output / "crops").glob("*.png"))
    if damage == "missing":
        crops[0].unlink()
    elif damage == "truncated":
        crops[0].write_bytes(crops[0].read_bytes()[:20])
    elif damage == "swapped":
        # These two have identical dtype/dimensions; checksums still catch the swap.
        a, b = crops[0].read_bytes(), crops[1].read_bytes()
        crops[0].write_bytes(b)
        crops[1].write_bytes(a)
    elif damage == "csv_label":
        path = output / "labels.csv"
        path.write_text(path.read_text().replace("Non-bedrock: Smooth", "Wrong class"))
    elif damage == "classes":
        (output / "classes.json").write_text('{"classes": []}')
    else:
        shutil.copyfile(crops[0], output / "crops/unlisted.png")
    with pytest.raises((ValueError, FileNotFoundError)):
        BlockCropDataset(output).validate()


def test_wrong_mapping_detected_against_archived_labels_even_with_new_checksums(exported):
    output, _, _ = exported
    path = output / "labels.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0].update(class_id="1", target_index="1", class_name="Non-bedrock: Lineated")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_integrity(output)
    with pytest.raises(ValueError, match="differs from archived class_id"):
        BlockCropDataset(output).validate()


@pytest.mark.parametrize("failure", ["cancel", "write", "bad_png", "archive"])
def test_failed_exports_never_publish_partial_datasets(crop_source, tmp_path, monkeypatch, failure):
    source, parquet, _, _ = crop_source
    out = tmp_path / "not-published"
    progress = []
    kwargs = {"progress_cb": lambda n, total: progress.append(n)}
    if failure == "cancel":
        kwargs["should_cancel"] = lambda: 1 in progress
    elif failure == "write":
        def broken_save(*args, **kwargs):
            raise OSError("simulated disk full")
        monkeypatch.setattr(Image.Image, "save", broken_save)
    elif failure == "bad_png":
        save = Image.Image.save
        def wrong_save(image, path, *args, **kwargs):
            wrong = np.array(image).copy()
            wrong[0, 0] += 1
            save(Image.fromarray(wrong), path, *args, **kwargs)
        monkeypatch.setattr(Image.Image, "save", wrong_save)
    else:
        def broken_archive(folder):
            raise OSError("archive failed")
        kwargs["prepare_archives"] = broken_archive
    with pytest.raises((ExportCancelled, OSError, ValueError)):
        export_probe_set(source, parquet, ROOT / "configs/classes.yaml", out, **kwargs)
    assert not out.exists()
    assert not list(tmp_path.glob(".probe-*"))


def test_real_gui_button_exports_unsaved_labels_and_reexports_fresh_snapshot(workflow, monkeypatch):
    win, pixels, source, root, _ = workflow
    monkeypatch.chdir(root)
    notices, errors = [], []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: notices.append(args[2]))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args[2]))
    press(win, Qt.Key_Q)
    press(win, Qt.Key_S)
    assert not (root / "labels/OBS.parquet").exists()  # Both labels are still unsaved.
    QTest.mouseClick(win.export_button, Qt.MouseButton.LeftButton)
    assert not errors, errors
    first = win.last_export_path
    ds = BlockCropDataset(first)
    assert ds.validate()["count"] == 2
    assert ds.metadata["label_origin"] == "labeling_session"
    assert {int(r["class_id"]) for r in ds.records} == {0, 11}
    for row in ds.records:
        x, y, w, h = (int(row[k]) for k in ("x_px", "y_px", "w_px", "h_px"))
        with Image.open(first / "crops" / row["crop_filename"]) as crop:
            np.testing.assert_array_equal(np.array(crop), pixels[y:y+h, x:x+w])
    with rasterio.open(first / "labels_coarse.tif") as coarse:
        assert coarse.read(1)[0, :2].tolist() == [0, 11]
    original = {p.relative_to(first): file_sha256(p) for p in first.rglob("*") if p.is_file()}
    win.session.labels.clear("OBS_0_0")
    win.session.labels.assign("OBS_32_0", 17, "Periodic Bedrock Ridges")
    QTest.mouseClick(win.export_button, Qt.MouseButton.LeftButton)
    second = win.last_export_path
    assert second != first
    new_ds = BlockCropDataset(second)
    assert new_ds.validate()["count"] == 1
    assert new_ds.records[0]["class_id"] == "17"
    assert not (second / "crops/OBS_0_0.png").exists()
    assert original == {p.relative_to(first): file_sha256(p) for p in first.rglob("*") if p.is_file()}
    assert len(notices) == 2
    assert not errors


def test_gui_failure_reports_error_and_keeps_scientist_labels(workflow, monkeypatch):
    win, _, _, root, _ = workflow
    monkeypatch.chdir(root)
    press(win, Qt.Key_Q)
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args[2]))
    def broken_save(*args, **kwargs):
        raise OSError("simulated disk full")
    monkeypatch.setattr(Image.Image, "save", broken_save)
    QTest.mouseClick(win.export_button, Qt.MouseButton.LeftButton)
    assert errors and "simulated disk full" in errors[0]
    assert win.session.labels.get_record("OBS_0_0").class_id == 0
    assert not list((root / "exports").rglob("dataset-*"))
    assert not list((root / "exports").rglob(".probe-*"))


def test_noncontiguous_class_ids_and_model_mapping_are_explicit(crop_source, tmp_path):
    source, parquet, _, _ = crop_source
    table = pq.read_table(parquet)
    mapping = {0: 42, 1: 7, 11: 99, 17: 42}
    records = table.to_pylist()
    for row in records:
        if row["status"] == "labeled":
            row["class_id"] = mapping[row["class_id"]]
    pq.write_table(pa.Table.from_pylist(records, schema=table.schema), parquet)
    scheme = yaml.safe_load((ROOT / "configs/classes.yaml").read_text())
    scheme["classes"] = [
        {"id": 99, "name": "C", "color": "#0000ff", "hotkey": "e", "model_index": 0},
        {"id": 7, "name": "A", "color": "#ff0000", "hotkey": "q", "model_index": 2},
        {"id": 100, "name": "Absent", "color": "#aaaaaa", "hotkey": "r", "model_index": 3},
        {"id": 42, "name": "B, with comma", "color": "#00ff00", "hotkey": "w", "model_index": 1},
    ]
    classes = tmp_path / "custom.yaml"
    classes.write_text(yaml.safe_dump(scheme))
    output = tmp_path / "custom-dataset"
    export_probe_set(source, parquet, classes, output)
    dataset = BlockCropDataset(output)
    assert dataset.validate()["count"] == 4
    assert dataset.class_to_target == {7: 0, 42: 1, 99: 2, 100: 3}
    assert {c["id"]: c["model_index"] for c in dataset.classes} == {7: 2, 42: 1, 99: 0, 100: 3}
    assert [int(r["target_index"]) for r in dataset.records] == [1, 0, 2, 1]


def test_cancel_dialog_waits_for_worker_and_publishes_nothing(crop_source, tmp_path, qapp):
    from marslabeler.ui.exportdialog import ExportDialog

    source, parquet, _, _ = crop_source
    raster = RasterSource(source)
    raster.open()
    grid = Grid(151, 93, 64, 32, "OBS", raster.transform, raster.crs)
    raster.close()
    output = tmp_path / "cancelled"
    dialog = ExportDialog(source, parquet, ROOT / "configs/classes.yaml", output, grid, False)
    # Reject as soon as progress starts. uint16 quantization and source reading
    # happen in the worker while the main Qt loop processes the cancel request.
    dialog.worker.progress.connect(lambda done, total: dialog.reject() if done == 0 else None)
    QTimer.singleShot(0, dialog.worker.start)
    dialog.exec()
    dialog.worker.wait()
    assert dialog.worker.cancelled
    assert not dialog.worker.isRunning()
    assert not output.exists()
    assert not list(tmp_path.glob(".probe-*"))


def test_empty_export_does_not_publish_a_dataset(crop_source, tmp_path):
    source, parquet, _, _ = crop_source
    table = pq.read_table(parquet)
    records = table.to_pylist()
    for row in records:
        row.update(class_id=-3, status="unlabeled")
    pq.write_table(pa.Table.from_pylist(records, schema=table.schema), parquet)
    with pytest.raises(ValueError, match="No labeled blocks"):
        export_probe_set(source, parquet, ROOT / "configs/classes.yaml", tmp_path / "empty")
    assert not (tmp_path / "empty").exists()
