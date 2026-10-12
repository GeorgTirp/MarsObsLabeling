"""Repeated export skips unchanged work without trusting mutable old files."""

import errno
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rasterio
import yaml
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from marslabeler.io.raster import RasterSource
from marslabeler.model.cropexport import ExportCancelled, export_crop_dataset
from marslabeler.model.cropdataset import BlockCropDataset
from marslabeler.model.dataset_integrity import file_sha256, write_integrity
from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import LabelStore, METADATA_KEY
from tests.test_crop_export import crop_source
from tests.test_labeling_workflow import ROOT, press, qapp, workflow


def export(source, parquet, output, **kwargs):
    return export_crop_dataset(source, parquet, ROOT / "configs/classes.yaml", output, **kwargs)


def edit_rows(path, edits):
    table = pq.read_table(path)
    rows = table.to_pylist()
    for row in rows:
        if row["block_id"] in edits:
            row.update(edits[row["block_id"]])
            row["edit_count"] += 1
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)


def fingerprints(folder):
    return {p.relative_to(folder): file_sha256(p) for p in folder.rglob("*") if p.is_file()}


def assert_pairs(folder, pixels):
    dataset = BlockCropDataset(folder)
    assert dataset.validate()["count"] == len(dataset)
    for row in dataset.records:
        x, y, w, h = (int(row[k]) for k in ("x_px", "y_px", "w_px", "h_px"))
        with Image.open(folder / "crops" / row["crop_filename"]) as crop:
            decoded = np.array(crop)
        assert decoded.dtype == pixels.dtype
        np.testing.assert_array_equal(decoded, pixels[y:y+h, x:x+w])
    return {r["block_id"]: int(r["class_id"]) for r in dataset.records}


def track_crop_work(monkeypatch):
    reads, encodes = [], []
    read, save = RasterSource.read_window, Image.Image.save

    def read_spy(self, *args, **kwargs):
        reads.append(args)
        return read(self, *args, **kwargs)

    def save_spy(self, filename, *args, **kwargs):
        encodes.append(Path(filename).name)
        return save(self, filename, *args, **kwargs)

    monkeypatch.setattr(RasterSource, "read_window", read_spy)
    monkeypatch.setattr(Image.Image, "save", save_spy)
    return reads, encodes


def test_unchanged_export_reuses_verified_folder_without_reading_or_encoding(crop_source, tmp_path, monkeypatch):
    source, parquet, pixels, _ = crop_source
    first = export(source, parquet, tmp_path / "first")
    before = fingerprints(first.folder)
    # A save after navigating changes cursor/timestamp metadata, not the dataset.
    table = pq.read_table(parquet)
    metadata = json.loads(table.schema.metadata[METADATA_KEY])
    metadata.update(current_block_idx=3, timestamp=999999)
    pq.write_table(table.replace_schema_metadata({METADATA_KEY: json.dumps(metadata).encode()}), parquet)
    reads, encodes = track_crop_work(monkeypatch)
    second = export(source, parquet, tmp_path / "unnecessary-copy", reuse=first.cache)
    assert second.unchanged and second.folder == first.folder
    assert (second.count, second.created, second.reused) == (4, 0, 4)
    assert not (tmp_path / "unnecessary-copy").exists()
    assert reads == encodes == []
    assert fingerprints(first.folder) == before
    assert_pairs(first.folder, pixels)


def test_relabels_clears_and_additions_only_generate_new_crop(crop_source, tmp_path, monkeypatch):
    source, parquet, pixels, _ = crop_source
    first = export(source, parquet, tmp_path / "first")
    before = fingerprints(first.folder)
    edit_rows(parquet, {
        "OBS_0_0": {"class_id": 17, "status": "labeled"},
        "OBS_32_0": {"class_id": -3, "status": "unlabeled"},
        "OBS_64_0": {"class_id": 6, "status": "labeled"},
    })
    reads, encodes = track_crop_work(monkeypatch)
    second = export(source, parquet, tmp_path / "second", reuse=first.cache)
    assert not second.unchanged
    assert (second.count, second.created, second.reused) == (4, 1, 3)
    assert len(reads) == 1 and encodes == ["OBS_64_0.png"]
    assert assert_pairs(second.folder, pixels) == {
        "OBS_0_0": 17, "OBS_64_0": 6, "OBS_64_32": 11, "OBS_128_64": 17,
    }
    assert not (second.folder / "crops/OBS_32_0.png").exists()
    assert fingerprints(first.folder) == before
    # Mutating or deleting an older export cannot damage the independent copy.
    old = first.folder / "crops/OBS_0_0.png"
    new = second.folder / "crops/OBS_0_0.png"
    assert not old.samefile(new)
    old.write_bytes(b"changed outside the application")
    shutil.rmtree(first.folder)
    assert_pairs(second.folder, pixels)


def test_class_definition_changes_refresh_manifest_without_recropping(crop_source, tmp_path, monkeypatch):
    source, parquet, pixels, _ = crop_source
    first = export(source, parquet, tmp_path / "first")
    scheme = yaml.safe_load((ROOT / "configs/classes.yaml").read_text())
    scheme["classes"][0].update(name="Renamed class", color="#abcdef", model_index=7)
    classes = tmp_path / "classes.yaml"
    classes.write_text(yaml.safe_dump(scheme))
    reads, encodes = track_crop_work(monkeypatch)
    result = export_crop_dataset(source, parquet, classes, tmp_path / "renamed", reuse=first.cache)
    assert not result.unchanged
    assert result.created == 0 and result.reused == 4
    assert reads == encodes == []
    dataset = BlockCropDataset(result.folder)
    assert dataset.classes[0]["name"] == "Renamed class"
    assert dataset.classes[0]["model_index"] == 7
    assert_pairs(result.folder, pixels)


def test_model_provenance_change_is_archived_without_recropping(crop_source, tmp_path, monkeypatch):
    source, parquet, pixels, _ = crop_source
    first = export(source, parquet, tmp_path / "first")
    table = pq.read_table(parquet)
    metadata = LabelStore.read_metadata(parquet)
    metadata.update(model_sig="different-checkpoint", model_stem="reviewed-model")
    pq.write_table(table.replace_schema_metadata({METADATA_KEY: json.dumps(metadata).encode()}), parquet)
    reads, encodes = track_crop_work(monkeypatch)
    result = export(source, parquet, tmp_path / "new-provenance", reuse=first.cache)
    assert not result.unchanged
    assert result.created == 0 and result.reused == 4
    assert reads == encodes == []
    assert LabelStore.read_metadata(result.folder / "labels.parquet")["model_sig"] == "different-checkpoint"
    assert_pairs(result.folder, pixels)


@pytest.mark.parametrize("damage", ["missing", "truncated", "swapped", "rehashed", "manifest", "integrity", "extra"])
def test_corrupt_previous_export_is_repaired_using_trusted_receipts(crop_source, tmp_path, monkeypatch, damage):
    source, parquet, pixels, _ = crop_source
    first = export(source, parquet, tmp_path / "first")
    crops = sorted((first.folder / "crops").glob("*.png"))
    expected_new = 1
    if damage == "missing":
        crops[0].unlink()
    elif damage == "truncated":
        crops[0].write_bytes(b"broken image")
    elif damage in ("swapped", "rehashed"):
        a, b = crops[0].read_bytes(), crops[1].read_bytes()
        crops[0].write_bytes(b)
        if damage == "swapped":
            crops[1].write_bytes(a)
            expected_new = 2
        else:
            write_integrity(first.folder)  # New on-disk hashes must not hide wrong pixels.
    elif damage == "manifest":
        (first.folder / "labels.csv").write_text("incorrect mappings")
        write_integrity(first.folder)
        expected_new = 0
    elif damage == "integrity":
        (first.folder / "integrity.json").write_text("broken JSON")
        expected_new = 0
    else:
        (first.folder / "crops/stale.png").write_bytes(crops[0].read_bytes())
        expected_new = 0
    damaged = fingerprints(first.folder)
    reads, encodes = track_crop_work(monkeypatch)
    result = export(source, parquet, tmp_path / "repaired", reuse=first.cache)
    assert not result.unchanged
    assert result.created == expected_new and result.reused == 4 - expected_new
    assert len(reads) == len(encodes) == expected_new
    assert_pairs(result.folder, pixels)
    assert fingerprints(first.folder) == damaged


def test_moved_previous_folder_regenerates_complete_dataset(crop_source, tmp_path):
    source, parquet, pixels, _ = crop_source
    first = export(source, parquet, tmp_path / "first")
    first.folder.rename(tmp_path / "moved")
    second = export(source, parquet, tmp_path / "second", reuse=first.cache)
    assert second.created == 4 and second.reused == 0
    assert_pairs(second.folder, pixels)


@pytest.mark.parametrize("failure", ["cancel", "copy"])
def test_failed_incremental_export_preserves_previous_dataset(crop_source, tmp_path, monkeypatch, failure):
    source, parquet, _, _ = crop_source
    first = export(source, parquet, tmp_path / "first")
    before = fingerprints(first.folder)
    edit_rows(parquet, {"OBS_0_0": {"class_id": 17}})
    progress = []
    kwargs = {"progress_cb": lambda done, total: progress.append(done)}
    if failure == "cancel":
        kwargs["should_cancel"] = lambda: 1 in progress
    else:
        def no_space(*args, **kwargs):
            raise OSError(errno.ENOSPC, "disk full")
        monkeypatch.setattr(shutil, "copyfile", no_space)
    with pytest.raises((ExportCancelled, OSError)):
        export(source, parquet, tmp_path / "failed", reuse=first.cache, **kwargs)
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".probe-*"))
    assert fingerprints(first.folder) == before


def test_source_replacement_never_reuses_wrong_pixels(crop_source, tmp_path):
    source, parquet, pixels, _ = crop_source
    first = export(source, parquet, tmp_path / "first")
    replacement = np.full_like(pixels, 125)
    with rasterio.open(source, "r+") as raster:
        raster.write(replacement, 1)
    with pytest.raises(ValueError, match="different or modified source"):
        export(source, parquet, tmp_path / "rejected", reuse=first.cache)
    assert not (tmp_path / "rejected").exists()
    # A fresh, explicitly rebound session still cannot reuse the old image cache.
    table = pq.read_table(parquet)
    metadata = LabelStore.read_metadata(parquet)
    with RasterSource(source) as raster:
        metadata["source"] = raster.fingerprint()
    pq.write_table(table.replace_schema_metadata({METADATA_KEY: json.dumps(metadata).encode()}), parquet)
    result = export(source, parquet, tmp_path / "new-source", reuse=first.cache)
    assert result.created == 4 and result.reused == 0
    assert_pairs(result.folder, replacement)


def test_grid_change_invalidates_crop_cache(crop_source, tmp_path):
    source, _, pixels, _ = crop_source
    parquet = tmp_path / "session/OBS.parquet"
    first = export(source, parquet, tmp_path / "first")
    with RasterSource(source) as raster:
        grid = Grid(151, 93, 64, 16, "OBS", raster.transform, raster.crs)
        labels = LabelStore(grid)
        labels.metadata["source"] = raster.fingerprint()
        labels.assign("OBS_0_0", 0, "Class 0")
        labels.save_parquet(parquet)
    result = export(source, parquet, tmp_path / "smaller-crops", reuse=first.cache)
    assert result.created == 1 and result.reused == 0
    assert BlockCropDataset(result.folder).block_size == 16
    assert_pairs(result.folder, pixels)


def test_gui_repeat_export_reports_unchanged_and_then_reuses_crops(workflow, monkeypatch, windows_flush_semantics):
    win, pixels, _, root, _ = workflow
    monkeypatch.chdir(root)
    notices, errors = [], []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: notices.append(args[2]))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args[2]))
    press(win, Qt.Key_Q)
    press(win, Qt.Key_S)
    QTest.mouseClick(win.export_button, Qt.MouseButton.LeftButton)
    first = win.last_export_path
    before = fingerprints(first)
    reads, encodes = track_crop_work(monkeypatch)
    win.save_action.trigger()  # Different save metadata must not create duplicate exports.
    QTest.mouseClick(win.export_button, Qt.MouseButton.LeftButton)
    assert win.last_export_path == first
    assert "Already up to date" in notices[-1]
    assert reads == encodes == []
    win.session.labels.assign("OBS_0_0", 17, "Periodic Bedrock Ridges")
    win.session.labels.assign("OBS_64_0", 6, "Class 6")
    QTest.mouseClick(win.export_button, Qt.MouseButton.LeftButton)
    second = win.last_export_path
    assert second != first
    assert "Crop images: 2 reused, 1 newly generated" in notices[-1]
    assert len(reads) == 1 and encodes == ["OBS_64_0.png"]
    assert assert_pairs(second, pixels) == {"OBS_0_0": 17, "OBS_32_0": 11, "OBS_64_0": 6}
    assert fingerprints(first) == before
    assert not errors, errors
