"""Rescue labels from an already-running Windows app without restarting it.

Run with the app's Python environment. Only a newly created capture directory
gets a deny-delete ACL; existing labels, code, and application memory are untouched.
The old atomic writer completes its temporary Parquet before failing at fsync.
Preventing cleanup in this isolated directory retains that complete snapshot.
"""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
from rasterio.transform import Affine

from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import LabelStore


def protect_capture_directory(capture: Path) -> None:
    """Create, protect, and test a NEW empty folder on a Windows ACL filesystem."""
    if os.name != "nt":
        raise RuntimeError("Live-session capture requires Windows; validation works on other systems.")
    capture.mkdir(parents=False, exist_ok=False)
    # Numeric Everyone SID is independent of the Windows display language.
    # DE and DC deny only deletion, leaving read/write and ACL editing intact.
    subprocess.run(
        ["icacls", str(capture), "/deny", "*S-1-1-0:(OI)(CI)(DE,DC)"],
        check=True, capture_output=True,
    )
    probe = capture / "protection-check.txt"
    with probe.open("wb") as stream:
        stream.write(b"capture remains readable and writable")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        probe.unlink()
    except PermissionError:
        pass
    else:
        raise RuntimeError("Delete protection did not work. Do not change the app's labels folder.")
    # Verify the exact creation/write/reopen sequence used by the running app.
    fd, name = tempfile.mkstemp(prefix="protection-check-", suffix=".tmp", dir=capture)
    os.close(fd)
    test_path = Path(name)
    test_table = pa.table({"protection_check": [1, 2, 3]})
    pq.write_table(test_table, test_path)
    if not pq.read_table(test_path).equals(test_table):
        raise RuntimeError("Protected-folder Parquet write/read self-check failed")
    try:
        test_path.unlink()
    except PermissionError:
        pass
    else:
        raise RuntimeError("Temporary files did not inherit delete protection. Stop here.")


def verify_snapshot(path: Path) -> dict:
    """Fully decode labels and require strict agreement with embedded geometry."""
    metadata = LabelStore.read_metadata(path)
    required = ("obs_id", "img_width", "img_height", "panel_size", "block_size")
    if any(k not in metadata for k in required):
        raise ValueError("Snapshot is missing embedded observation/geometry metadata")
    obs_id = metadata["obs_id"]
    if not isinstance(obs_id, str) or obs_id in ("", ".", "..") or any(c in obs_id for c in '/\\:'):
        raise ValueError("Unsafe observation ID")
    for k in required[1:]:
        if not isinstance(metadata[k], int) or metadata[k] <= 0:
            raise ValueError(f"Invalid geometry: {k}")
    grid = Grid(**{k: metadata[k] for k in required}, transform=Affine.identity())
    table = pq.read_table(path)
    store = LabelStore.load_parquet(path, grid, strict=True)
    # Atomic session snapshots contain every block, including unlabelled blocks.
    if table.num_rows != grid.num_blocks():
        raise ValueError("Snapshot does not contain the full session grid")
    rows = table.to_pylist()
    if any(row["obs_id"] != obs_id for row in rows):
        raise ValueError("Observation IDs disagree with snapshot metadata")
    counts = Counter(r["status"] for r in rows)
    if store.count_labeled() != counts.get("labeled", 0):
        raise ValueError("Label count changed while validating the snapshot")
    return {
        "observation_id": obs_id, "total_blocks": table.num_rows,
        "labeled_blocks": counts.get("labeled", 0), "status_counts": dict(counts),
        "class_counts": dict(Counter(str(r["class_id"]) for r in rows if r["status"] == "labeled")),
        "session": metadata,
    }


def recover_snapshot(source: Path, recovery_root: Path) -> tuple[Path, dict]:
    """Copy into a fresh folder, then verify; never modify the captured source."""
    recovery_root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="snapshot-", dir=recovery_root))
    candidate = folder / "captured.parquet"
    # Read source once; the independent copy must fully decode before publishing.
    with source.open("rb") as original, candidate.open("xb") as saved:
        shutil.copyfileobj(original, saved)
        saved.flush()
        os.fsync(saved.fileno())
    report = verify_snapshot(candidate)
    destination = folder / f"{report['observation_id']}.parquet"
    if destination != candidate:
        candidate.rename(destination)
    with destination.open("rb") as stream:
        report["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    report["captured_from"] = str(source.resolve())
    (folder / "recovery-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return destination, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path.home() / "MarsLabelRescue")
    parser.add_argument("--recover-file", type=Path, help="Validate/copy an already retained Parquet or .tmp file")
    args = parser.parse_args(argv)
    if args.recover_file:
        saved, report = recover_snapshot(args.recover_file, args.output)
        print(f"VERIFIED: {report['labeled_blocks']:,} labeled blocks\n{saved}")
        return 0
    if os.name != "nt":
        parser.error("Live-session rescue must run on the Windows PC with the open app.")
    root = args.output.resolve() / datetime.now().strftime("rescue-%Y%m%d-%H%M%S-%f")
    root.mkdir(parents=True, exist_ok=False)
    capture = root / "capture"
    protect_capture_directory(capture)
    print(
        "Keep the labeling app open and pause labeling.\n\n"
        "1. In the OPEN app: File > Set Labels Folder, select:\n"
        f"   {capture}\n\n"
        "2. Dismiss the folder-change notice. DO NOT reopen the observation.\n"
        "3. Press Save labels once. A save error is expected in the old app.\n"
        "   This isolated folder retains its temporary snapshot despite the error.\n"
        "4. Return here after the save attempt finishes.\n",
        flush=True,
    )
    while True:
        answer = input("Press Enter to recover, or q to stop (keep the app open): ").strip().lower()
        if answer == "q":
            return 1
        candidates = list(capture.glob(".*.parquet.*.tmp")) + list(capture.glob("*.parquet"))
        if not candidates:
            print("No snapshot yet. Check the selected folder and press Save labels again.")
            continue
        # Use only the newest attempted snapshot; never silently fall back to stale labels.
        latest = max(candidates, key=lambda p: p.stat().st_mtime_ns)
        try:
            saved, report = recover_snapshot(latest, root / "recovered")
        except Exception as exc:
            print(f"NOT VERIFIED: {exc}\nKeep the app open. Try Save labels again.")
            continue
        print(
            f"\nVERIFIED: {report['labeled_blocks']:,} labeled blocks, "
            f"{report['total_blocks']:,} total blocks.\n"
            f"Recovered file: {saved}\n"
            f"Resume folder: {saved.parent}\n\n"
            "Compare this count with the app and copy this recovered folder somewhere safe.\n"
            "Keep the old app open until a SECOND instance of the fixed app has reopened\n"
            "the original image from this resume folder and you have checked your labels.\n"
            "The capture folder is only for rescue; do not use it for future saves.\n"
            "To remove its delete protection later (only after recovery), run:\n"
            f'icacls "{capture}" /remove:d "*S-1-1-0"\n'
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
