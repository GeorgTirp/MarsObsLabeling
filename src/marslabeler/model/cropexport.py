"""Portable image/whole-block-label datasets shared by GUI and CLI exports."""

import csv
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
import numpy as np
from PIL import Image

from marslabeler.classes import load_classes
from marslabeler.io.raster import RasterSource
from marslabeler.inference.engine import compute_global_quantization
from marslabeler.model.labelstore import LabelStore
from marslabeler.model.dataset_integrity import write_integrity


class ExportCancelled(Exception):
    """A cancelled export never publishes a partial dataset."""


def export_probe_set(
    jp2_path: Path,
    parquet_path: Path,
    classes_yaml: Path,
    output_dir: Path,
    min_confidence: float = 0.0,
    *,
    progress_cb: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    prepare_archives: Callable[[Path], None] | None = None,
    label_origin: str = "unspecified",
) -> int:
    """
    Export labeled blocks as crops for training (probe set per Model v2 §6.1).

    Args:
        jp2_path: Path to JP2 observation
        parquet_path: Path to labels Parquet file
        classes_yaml: Path to classes legend YAML
        output_dir: Output directory for crops
        min_confidence: Legacy parameter; only 0 is supported (no measured confidence).

    Returns the number of exported crop/label pairs. An error or cancellation
    leaves no partial output dataset and never overwrites a nonempty folder.
    """
    def check_cancelled():
        if should_cancel is not None and should_cancel():
            raise ExportCancelled()

    if min_confidence != 0:
        raise ValueError("Confidence filtering is not supported: block labels have no confidence score.")
    check_cancelled()

    output_dir = Path(output_dir)
    # A fresh directory prevents orphaned crops from earlier exports being
    # mistaken for current labels by folder-based training loaders.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Export directory is not empty: {output_dir}. Choose a new directory.")

    table = pq.read_table(parquet_path)
    metadata = LabelStore.read_metadata(parquet_path)
    classes_scheme = load_classes(classes_yaml)
    rows = table.to_pylist()
    required = {"block_id", "obs_id", "x_px", "y_px", "w_px", "h_px", "class_id", "status"}
    if not required.issubset(table.column_names):
        raise ValueError("Labels are missing required block coordinates or class columns")

    with RasterSource(jp2_path) as raster:
        raster.validate_fingerprint(metadata.get("source"))
        source_identity = raster.fingerprint()
        for key, actual in (("img_width", raster.width), ("img_height", raster.height)):
            if key in metadata and metadata[key] != actual:
                raise ValueError(f"Source image {key} does not match saved labels")
        if raster.dtype not in ("uint8", "uint16"):
            raise ValueError(f"PNG crop export requires uint8 or uint16 imagery, got {raster.dtype}")

        seen = set()
        labeled = []
        for row in rows:
            check_cancelled()
            bid = row["block_id"]
            if bid in seen:
                raise ValueError(f"Duplicate block id: {bid}")
            seen.add(bid)
            if row["obs_id"] != Path(jp2_path).stem:
                raise ValueError(f"Label observation {row['obs_id']} does not match the source image")
            x, y, w, h = (row[k] for k in ("x_px", "y_px", "w_px", "h_px"))
            if not all(isinstance(v, int) for v in (x, y, w, h)):
                raise ValueError(f"Invalid block coordinates: {bid}")
            if Path(bid).name != bid or "/" in bid or "\\" in bid:
                raise ValueError(f"Unsafe block id: {bid}")
            if bid != f"{row['obs_id']}_{x}_{y}":
                raise ValueError(f"Block id and coordinates disagree: {bid}")
            status, cid = row["status"], row["class_id"]
            if status != "labeled":
                if {"abstain": -1, "nodata": -2, "unlabeled": -3}.get(status) != cid:
                    raise ValueError(f"Invalid class/status for block {bid}")
                continue
            if cid not in classes_scheme.classes:
                raise ValueError(f"Unknown class id {cid} for block {bid}")
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > raster.width or y + h > raster.height:
                raise ValueError(f"Block extent is outside source image bounds: {bid}")
            size = metadata.get("block_size")
            if size and (x % size or y % size or w != min(size, raster.width - x)
                         or h != min(size, raster.height - y)):
                raise ValueError(f"Block extent does not match saved tile geometry: {bid}")
            labeled.append(row)

        if not labeled:
            raise ValueError("No labeled blocks to export. Assign a terrain class to at least one block.")
        if progress_cb:
            progress_cb(0, len(labeled))
        # Store one shared input mapping, never stretch each crop independently.
        bounds = None
        if raster.dtype != "uint8":
            bounds = compute_global_quantization(raster, raster.width, raster.height)
        check_cancelled()
        classes = [
            {"id": cid, "name": c.name, "color": c.color, "target_index": index,
             "model_index": c.model_index if c.model_index is not None else cid}
            for index, (cid, c) in enumerate(sorted(classes_scheme.classes.items()))
        ]
        targets = {c["id"]: c["target_index"] for c in classes}

        # Publish a complete set only after every crop and all manifests succeed.
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".probe-", dir=output_dir.parent) as staging:
            staged = Path(staging) / "export"
            crops_dir = staged / "crops"
            crops_dir.mkdir(parents=True)
            fields = ["block_id", "obs_id", "x_px", "y_px", "class_id", "target_index", "class_name",
                      "w_px", "h_px", "crop_filename", "dtype", "edit_count"]
            with (staged / "labels.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for count, row in enumerate(labeled, 1):
                    check_cancelled()
                    x, y, w, h = (row[k] for k in ("x_px", "y_px", "w_px", "h_px"))
                    data = raster.read_window(x, y, w, h, w, h)
                    filename = row["block_id"] + ".png"
                    # Let Pillow infer L or I;16: forcing L reinterprets uint16 bytes.
                    Image.fromarray(data).save(crops_dir / filename)
                    # Reopen the encoded file before publishing: this checks
                    # decoding, shape, bit depth, and every source pixel.
                    with Image.open(crops_dir / filename) as saved:
                        decoded = np.array(saved)
                    if decoded.dtype != data.dtype or not np.array_equal(decoded, data):
                        raise ValueError(f"Crop failed source-pixel verification: {filename}")
                    writer.writerow({"block_id": row["block_id"], "obs_id": row["obs_id"], "x_px": x, "y_px": y,
                                     "w_px": w, "h_px": h, "class_id": row["class_id"],
                                     "class_name": classes_scheme.get_name(row["class_id"]),
                                     "target_index": targets[row["class_id"]],
                                     "crop_filename": filename, "edit_count": row.get("edit_count", 0),
                                     "dtype": raster.dtype})
                    if progress_cb:
                        progress_cb(count, len(labeled))
            (staged / "classes.json").write_text(json.dumps({"classes": classes}, indent=2))
            block_size = metadata.get("block_size") or max(
                max(r["w_px"], r["h_px"]) for r in labeled
            )
            dataset_metadata = {
                "format": "mars-block-crops", "version": 1,
                "target_kind": "whole_block_majority_class", "label_origin": label_origin,
                "count": len(labeled), "block_size": block_size,
                "dtype": raster.dtype, "nodata": raster.nodata,
                "observation_id": Path(jp2_path).stem,
                "source": source_identity, "gsd_m": raster.gsd,
                "preprocessing": {
                    "method": "identity_uint8" if bounds is None else "global_percentile_uint8",
                    "bounds": bounds, "percentiles": [1, 99] if bounds else None,
                    "normalization": "uint8 / 127.5 - 1",
                    "note": "Global uint16 stretch approximates DRG input; native PNGs remain unchanged.",
                },
            }
            (staged / "dataset.json").write_text(json.dumps(dataset_metadata, indent=2))
            (staged / "README.md").write_text(DATASET_README, encoding="utf-8")
            shutil.copyfile(parquet_path, staged / "labels.parquet")
            if prepare_archives:
                prepare_archives(staged)
            raster.validate_fingerprint(source_identity)
            write_integrity(staged, check_cancelled=check_cancelled)
            from marslabeler.model.cropdataset import BlockCropDataset

            # Validate the portable format against the archived label snapshot.
            BlockCropDataset(staged).validate(should_cancel=should_cancel)
            check_cancelled()
            staged.replace(output_dir)
    return len(labeled)



DATASET_README = """# Block crop / label dataset

`crops/` contains one native uint8/uint16 PNG per labeled block. `labels.csv`
pairs `crop_filename` (relative to crops/) with the stable `class_id`, class
name, and a contiguous `target_index` for training a classifier. The complete
mapping is in `classes.json`; it does not depend on which classes occur here.
Abstain, nodata, and unlabeled blocks have no crop. Edge crops keep their true
width/height. Source coordinates, observation ID, dtype, and edit count travel
with each pair. `labels.parquet` archives the complete label snapshot.

`integrity.json` records SHA-256 checksums. Every crop was reopened and compared
pixel-for-pixel with the source before this folder was published. To recheck
the folder after copying it, without requiring PyTorch:

```bash
python -m marslabeler.model.cropdataset /path/to/this/folder
```

With mars-labeler and PyTorch installed:

```python
from torch.utils.data import DataLoader
from marslabeler.model.cropdataset import BlockCropDataset

dataset = BlockCropDataset("/path/to/this/folder")
batch = next(iter(DataLoader(dataset, batch_size=8, shuffle=True)))
images = batch["image"]    # float32 [N, 1, block_size, block_size], [-1, 1]
targets = batch["target"]  # int64 [N], contiguous indices in classes.json
valid = batch["valid_mask"]  # excludes padding and invalid source pixels
# logits = classifier(images)  # [N, dataset.num_classes]
# loss = torch.nn.functional.cross_entropy(logits, targets)
```

The loader needs only this folder, not the original observation. Native PNGs
are never overwritten. At read time it applies the fixed preprocessing recorded
in dataset.json and pads partial crops on the right/bottom, without resizing.
uint16 input uses one observation-wide percentile stretch, the same approximation
used by mars-inference; this does not recreate the original DRG training pipeline.
`target_index` is for a new classifier; `model_index` records the configured
existing-model channel mapping separately. Check the class names/order and input
scale against any pretrained model. Set `num_classes=` to reject targets outside
a chosen classifier head, or provide a `transform` for different preprocessing.

These labels describe a whole block's majority terrain. They are not dense
segmentation masks and do not plug into Stage-3 segmentation training unchanged.
The optional `*_coarse.tif` is a GIS overview, not a Stage-3 target raster.
Prediction-review exports may contain unreviewed model predictions; consult
`label_origin` in dataset.json. `edit_count` records edits, not review approval.

Split training/validation/test sets by observation or separated spatial regions
to avoid evaluating on immediate neighbors of training crops. Context-dependent
models also need surrounding imagery, which is not included in this crop bundle.
"""
