"""Replace a file only after its complete replacement has reached disk."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


@contextmanager
def atomic_output(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        # Windows FlushFileBuffers (used by os.fsync) requires write access.
        # Opening read-only works on POSIX but raises EBADF on Windows, breaking
        # both session saves and export snapshots before they can be published.
        # r+b preserves the completed contents; wb would truncate them.
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
