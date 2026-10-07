"""Checksums for detecting incomplete, changed, or mixed crop exports."""

import hashlib
import json
from pathlib import Path


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_integrity(folder, *, check_cancelled=None):
    folder = Path(folder)
    files = {}
    for path in sorted(folder.rglob("*")):
        if check_cancelled:
            check_cancelled()
        if path.is_file() and path != folder / "integrity.json":
            files[path.relative_to(folder).as_posix()] = file_sha256(path)
    (folder / "integrity.json").write_text(json.dumps({
        "algorithm": "sha256", "files": files,
    }, indent=2), encoding="utf-8")


def read_integrity(folder):
    data = json.loads((Path(folder) / "integrity.json").read_text(encoding="utf-8"))
    files = data.get("files")
    if data.get("algorithm") != "sha256" or not isinstance(files, dict) or not files:
        raise ValueError("Missing or invalid dataset integrity manifest")
    for name, digest in files.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"Invalid integrity manifest path: {name}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Invalid checksum for {name}")
    return files


def verify_file(folder, files, name):
    if name not in files:
        raise ValueError(f"No checksum recorded for {name}")
    folder = Path(folder).resolve()
    path = (folder / name).resolve()
    if not path.is_relative_to(folder):
        raise ValueError(f"Dataset file points outside its folder: {name}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {name}")
    if file_sha256(path) != files[name]:
        raise ValueError(f"Checksum mismatch: {name}. The export is damaged or has been changed.")
    return path
