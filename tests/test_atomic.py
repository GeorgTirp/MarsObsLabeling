"""Cross-platform durability and preservation of the last good snapshot."""
import errno
import os

import pytest

from marslabeler.io.atomic import atomic_output


def test_readonly_flush_reproduces_windows_error(tmp_path, windows_flush_semantics):
    path = tmp_path / "labels"
    path.write_bytes(b"complete labels")
    with path.open("rb") as stream, pytest.raises(OSError) as error:
        os.fsync(stream.fileno())
    assert error.value.errno == errno.EBADF


def test_atomic_save_flushes_writable_file_without_truncating(tmp_path, windows_flush_semantics):
    path = tmp_path / "labels.parquet"
    path.write_bytes(b"previous labels")
    payload = bytes(range(256)) * 100
    with atomic_output(path) as temporary:
        assert temporary.parent == path.parent
        temporary.write_bytes(payload)
        assert path.read_bytes() == b"previous labels"
    assert path.read_bytes() == payload
    assert len(windows_flush_semantics) == 1
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("stage", ["write", "flush", "replace"])
@pytest.mark.parametrize("existing", [False, True])
def test_failed_write_flush_or_replace_preserves_previous_file(tmp_path, monkeypatch, stage, existing):
    path = tmp_path / "labels.parquet"
    if existing:
        path.write_bytes(b"previous labels")

    def fail(*args):
        raise OSError(errno.EIO, f"failed {stage}")

    if stage != "write":
        monkeypatch.setattr(os, "fsync" if stage == "flush" else "replace", fail)
    with pytest.raises(OSError, match=f"failed {stage}"):
        with atomic_output(path) as temporary:
            temporary.write_bytes(b"new labels")
            if stage == "write":
                fail()
    if existing:
        assert path.read_bytes() == b"previous labels"
    else:
        assert not path.exists()
    assert not list(tmp_path.glob(".*.tmp"))
