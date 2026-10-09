"""Real save/export UI under Windows flush rules, plus live-session rescue."""
import errno
import importlib.util
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from marslabeler.model.cropdataset import BlockCropDataset
from marslabeler.model.session import Session
from tests.test_labeling_workflow import ROOT, workflow, qapp, press

spec = importlib.util.spec_from_file_location("rescue_open_session", ROOT / "scripts/rescue_open_session.py")
rescue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rescue)


def test_save_export_reopen_with_windows_flush_rules(workflow, monkeypatch, windows_flush_semantics):
    win, pixels, source, root, open_window = workflow
    monkeypatch.chdir(root)
    notices, errors = [], []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: notices.append(args[2]))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args[2]))
    win.labels_dir = root / "Données – Mars labels"
    press(win, Qt.Key_Q)
    press(win, Qt.Key_S)
    QTest.mouseClick(win.workspace_save_button, Qt.MouseButton.LeftButton)
    assert win.workspace_save_button.text().startswith("Saved ")
    assert "2 labeled blocks saved" in win.save_feedback_label.text()
    path = win.labels_dir / "OBS.parquet"
    assert str(path) in win.workspace_save_button.toolTip()
    original = path.read_bytes()
    win._refresh_view()
    assert "2 labeled blocks saved" in win.save_feedback_label.text()
    QTest.mouseClick(win.export_button, Qt.MouseButton.LeftButton)
    assert not errors
    dataset = BlockCropDataset(win.last_export_path)
    assert dataset.validate()["count"] == 2
    assert {int(r["class_id"]) for r in dataset.records} == {0, 11}
    for row in dataset.records:
        x, y, w, h = (int(row[k]) for k in ("x_px", "y_px", "w_px", "h_px"))
        with Image.open(win.last_export_path / "crops" / row["crop_filename"]) as image:
            np.testing.assert_array_equal(np.array(image), pixels[y:y+h, x:x+w])
    assert path.read_bytes() == original
    restored = Session.load_or_create(source, win.session.grid, win.session.config, win.labels_dir)
    try:
        assert restored.labels.count_labeled() == 2
        assert restored.labels.get_record("OBS_0_0").class_id == 0
        assert restored.labels.get_record("OBS_32_0").class_id == 11
    finally:
        restored.raster.close()
    assert len(windows_flush_semantics) >= 3  # Labels, sidecar, export snapshot.


def test_failure_is_visible_preserves_progress_and_retry_succeeds(workflow, monkeypatch):
    win, _, _, root, _ = workflow
    press(win, Qt.Key_Q)
    win.save_action.trigger()
    path = root / "labels/OBS.parquet"
    before = path.read_bytes()
    press(win, Qt.Key_S)
    assert win.save_feedback_label.text() == "Unsaved changes"
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args[2]))

    def fail(fd):
        raise OSError(errno.EBADF, "Bad file descriptor")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail)
        win._do_autosave()
        win._do_autosave()
        assert len(errors) == 1  # Automatic failures notify once until recovery.
        win.save_action.trigger()
        assert len(errors) == 2  # Explicit retry gives explicit error feedback.
        assert "Keep it open" in errors[-1]
        assert "SAVE FAILED" in win.save_feedback_label.text()
        assert "Save failed" in win.workspace_save_button.text()
        assert path.read_bytes() == before
        assert win.session.labels.count_labeled() == 2
        assert win.session.label_count_since_autosave > 0
        event = QCloseEvent()
        win.closeEvent(event)
        assert not event.isAccepted()
        previous_session = win.session
        win._load_observation(root / "OTHER.tif")
        assert win.session is previous_session
    win.save_action.trigger()
    assert "2 labeled blocks saved" in win.save_feedback_label.text()
    assert win.session.label_count_since_autosave == 0
    assert sum(r["status"] == "labeled" for r in pq.read_table(path).to_pylist()) == 2


def test_rescue_copy_is_verified_resumable_and_never_overwrites(workflow, tmp_path, windows_flush_semantics):
    win, _, source, root, _ = workflow
    press(win, Qt.Key_Q)
    press(win, Qt.Key_S)
    # Model the completed temporary file left by the failed old Windows save.
    win.session.save_session(root / "capture")
    captured = root / "capture/OBS.parquet"
    original = captured.read_bytes()
    saved, report = rescue.recover_snapshot(captured, root / "recovered")
    second, _ = rescue.recover_snapshot(captured, root / "recovered")
    assert saved != second
    assert captured.read_bytes() == saved.read_bytes() == second.read_bytes() == original
    assert report["labeled_blocks"] == 2
    assert report["class_counts"] == {"0": 1, "11": 1}
    restored = Session.load_or_create(source, win.session.grid, win.session.config, saved.parent)
    try:
        assert restored.labels.count_labeled() == 2
        assert restored.current_block_idx == win.session.current_block_idx
        assert restored.labels.get_record("OBS_32_0").class_id == 11
    finally:
        restored.raster.close()


def test_rescue_rejects_truncated_snapshot(workflow):
    win, _, _, root, _ = workflow
    path = root / "broken.tmp"
    path.write_bytes(b"PAR1incomplete")
    with pytest.raises(Exception):
        rescue.recover_snapshot(path, root / "recovered")
    assert not list((root / "recovered").rglob("recovery-report.json"))
    assert path.read_bytes() == b"PAR1incomplete"


@pytest.mark.parametrize("native", [False, pytest.param(
    True, marks=pytest.mark.skipif(os.name != "nt", reason="Requires actual Windows NTFS ACLs"),
)])
def test_windows_protection_retains_old_writer_snapshot(workflow, monkeypatch, windows_flush_semantics, native):
    win, _, _, root, _ = workflow
    capture = root / "protected-capture"
    try:
        if native:
            rescue.protect_capture_directory(capture)
        else:
            capture.mkdir()
            real_unlink = Path.unlink

            def deny_capture_cleanup(path, *args, **kwargs):
                if path.parent == capture:
                    raise PermissionError("simulated Windows delete protection")
                return real_unlink(path, *args, **kwargs)

            monkeypatch.setattr(Path, "unlink", deny_capture_cleanup)
        # Reproduce the ORIGINAL atomic_output, including failed cleanup.
        from contextlib import contextmanager
        import tempfile

        @contextmanager
        def original_atomic_output(path):
            fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
            os.close(fd)
            temporary = Path(name)
            try:
                yield temporary
                with temporary.open("rb") as stream:
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

        monkeypatch.setattr("marslabeler.model.labelstore.atomic_output", original_atomic_output)
        press(win, Qt.Key_Q)
        with pytest.raises(PermissionError):
            win.session.save_session(capture)
        retained, = capture.glob(".*.parquet.*.tmp")
        saved, report = rescue.recover_snapshot(retained, root / "recovered")
        assert report["labeled_blocks"] == 1
        assert saved.read_bytes() == retained.read_bytes()
    finally:
        if native and capture.exists():
            import subprocess
            subprocess.run(["icacls", str(capture), "/remove:d", "*S-1-1-0"], check=True)
