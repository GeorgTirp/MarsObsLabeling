"""A PyTorch-compatible map dataset for portable block crop exports.

Torch is imported only when retrieving a tensor sample; export itself stays
torch-free. Each target is one whole-block class, not a segmentation mask.
"""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from marslabeler.inference.engine import quantize_to_uint8
from marslabeler.io.preprocess import compute_invalid_mask
from marslabeler.model.dataset_integrity import read_integrity, verify_file


class BlockCropDataset:
    """Load a copied export folder without access to the original observation.

    Default preprocessing uses the export's shared uint8 conversion, normalizes
    to [-1, 1], and pads partial blocks without resizing. A custom ``transform``
    receives a copy of the native 2-D numpy crop and must return a float tensor
    of shape (1, h, w), preserving geometry; invalid/padded pixels become -1.
    Targets use the explicit contiguous ``target_index`` mapping in classes.json.
    """

    def __init__(self, folder, *, num_classes=None, transform=None):
        self.folder = Path(folder)
        self.checksums = read_integrity(self.folder)
        for name in ("dataset.json", "classes.json", "labels.csv", "labels.parquet"):
            verify_file(self.folder, self.checksums, name)
        self.metadata = json.loads((self.folder / "dataset.json").read_text())
        if (self.metadata.get("format"), self.metadata.get("version")) != (
            "mars-block-crops", 1
        ):
            raise ValueError("Unsupported block crop dataset format")
        if self.metadata.get("target_kind") != "whole_block_majority_class":
            raise ValueError("Expected whole-block class labels")
        self.classes = json.loads((self.folder / "classes.json").read_text())["classes"]
        self.class_to_target = {c["id"]: c["target_index"] for c in self.classes}
        if len(self.class_to_target) != len(self.classes) or sorted(
            self.class_to_target.values()
        ) != list(range(len(self.classes))):
            raise ValueError("Invalid or non-contiguous class mapping")
        self.num_classes = len(self.classes) if num_classes is None else num_classes
        if not isinstance(self.num_classes, int) or self.num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        class_names = {c["id"]: c["name"] for c in self.classes}
        self.block_size = int(self.metadata["block_size"])
        if self.block_size <= 0:
            raise ValueError("Invalid block size")
        self.transform = transform
        with (self.folder / "labels.csv").open(newline="", encoding="utf-8") as stream:
            self.records = list(csv.DictReader(stream))
        if not self.records or len(self.records) != self.metadata["count"]:
            raise ValueError("Empty or incomplete crop manifest")
        seen = set()
        for record in self.records:
            filename = record["crop_filename"]
            if Path(filename).name != filename or "/" in filename or "\\" in filename:
                raise ValueError(f"Invalid crop filename: {filename}")
            if record["block_id"] in seen:
                raise ValueError("Duplicate block in crop manifest")
            seen.add(record["block_id"])
            cid, target = int(record["class_id"]), int(record["target_index"])
            if self.class_to_target.get(cid) != target:
                raise ValueError(f"Class mapping mismatch for {filename}")
            if record["class_name"] != class_names[cid]:
                raise ValueError(f"Class name mismatch for {filename}")
            x, y = int(record["x_px"]), int(record["y_px"])
            if (record["obs_id"] != self.metadata["observation_id"]
                    or x < 0 or y < 0 or x % self.block_size or y % self.block_size
                    or record["block_id"] != f"{record['obs_id']}_{x}_{y}"
                    or filename != record["block_id"] + ".png"):
                raise ValueError(f"Block coordinates/filename mismatch for {filename}")
            if not 0 <= target < self.num_classes:
                raise ValueError(
                    f"Class {cid} needs target {target}, outside a {self.num_classes}-class head"
                )
            if not (self.folder / "crops" / filename).is_file():
                raise FileNotFoundError(f"Missing crop: {filename}")
            if f"crops/{filename}" not in self.checksums:
                raise ValueError(f"No crop checksum: {filename}")
            if record["dtype"] != self.metadata["dtype"]:
                raise ValueError(f"Crop dtype disagrees with dataset metadata: {filename}")
            if not all(0 < int(record[k]) <= self.block_size for k in ("h_px", "w_px")):
                raise ValueError(f"Invalid crop dimensions: {filename}")
        prep = self.metadata["preprocessing"]
        expected_crops = {f"crops/{r['crop_filename']}" for r in self.records}
        recorded_crops = {name for name in self.checksums if name.startswith("crops/")}
        if expected_crops != recorded_crops:
            raise ValueError("Crop files and label manifest do not match")
        if prep["method"] not in ("identity_uint8", "global_percentile_uint8"):
            raise ValueError("Unsupported crop preprocessing")
        if prep["method"] == "global_percentile_uint8":
            bounds = np.asarray(prep["bounds"], dtype=float)
            if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
                raise ValueError("Invalid global quantization bounds")

    def __len__(self):
        return len(self.records)

    def _read_crop(self, record):
        path = verify_file(self.folder, self.checksums, f"crops/{record['crop_filename']}")
        with Image.open(path) as image:
            raw = np.array(image)
        h, w = int(record["h_px"]), int(record["w_px"])
        if raw.shape != (h, w) or str(raw.dtype) != record["dtype"]:
            raise ValueError(f"Crop data does not match manifest: {record['crop_filename']}")
        return raw

    def validate(self, *, should_cancel=None):
        """Verify all files and pairings without requiring torch or the source image."""
        import pyarrow.parquet as pq

        def check_cancelled():
            if should_cancel and should_cancel():
                from marslabeler.model.cropexport import ExportCancelled
                raise ExportCancelled()

        actual = {
            path.relative_to(self.folder).as_posix()
            for path in self.folder.rglob("*")
            if path.is_file() and path != self.folder / "integrity.json"
        }
        if actual != set(self.checksums):
            raise ValueError("Dataset file inventory changed: missing or extra files")
        for name in self.checksums:
            check_cancelled()
            verify_file(self.folder, self.checksums, name)
        # Check pairing against the independent label snapshot as well as the
        # manifest, catching misassigned classes or accidentally omitted blocks.
        columns = ["block_id", "obs_id", "x_px", "y_px", "w_px", "h_px", "class_id", "status"]
        archived_rows = pq.read_table(self.folder / "labels.parquet", columns=columns).to_pylist()
        if len({r["block_id"] for r in archived_rows}) != len(archived_rows):
            raise ValueError("Duplicate blocks in archived labels")
        labeled = {r["block_id"]: r for r in archived_rows if r["status"] == "labeled"}
        if set(labeled) != {r["block_id"] for r in self.records}:
            raise ValueError("Crop manifest differs from the archived labeled blocks")
        for record in self.records:
            check_cancelled()
            archived = labeled[record["block_id"]]
            for key in ("x_px", "y_px", "w_px", "h_px", "class_id"):
                if int(record[key]) != archived[key]:
                    raise ValueError(f"Crop/label mapping differs from archived {key}")
            if record["obs_id"] != archived["obs_id"]:
                raise ValueError("Observation id differs from archived labels")
            self._read_crop(record)
        return {"count": len(self), "files_verified": len(self.checksums),
                "observation_id": self.metadata["observation_id"]}

    def __getitem__(self, index):
        import torch

        record = self.records[index]
        raw = self._read_crop(record)
        h, w = int(record["h_px"]), int(record["w_px"])
        invalid = compute_invalid_mask(raw)
        nodata = self.metadata.get("nodata")
        if nodata is not None:
            invalid |= raw == nodata
        if self.transform is None:
            prep = self.metadata["preprocessing"]
            if prep["method"] == "global_percentile_uint8":
                data = quantize_to_uint8(raw, tuple(prep["bounds"]))
            else:
                if raw.dtype != np.uint8:
                    raise ValueError("identity_uint8 preprocessing requires uint8 crops")
                data = raw
            crop = torch.from_numpy(data.astype(np.float32) / 127.5 - 1).unsqueeze(0)
        else:
            crop = self.transform(raw.copy())
            if not isinstance(crop, torch.Tensor) or not crop.is_floating_point():
                raise ValueError("transform must return a floating-point torch tensor")
            if tuple(crop.shape) != (1, h, w):
                raise ValueError("transform must preserve crop shape (1, h, w)")
            crop = crop.to(dtype=torch.float32, device="cpu")
        if not torch.isfinite(crop).all():
            raise ValueError("Preprocessed crop contains non-finite values")
        image = torch.full((1, self.block_size, self.block_size), -1.0)
        image[:, :h, :w] = crop
        valid = torch.zeros((self.block_size, self.block_size), dtype=torch.bool)
        valid[:h, :w] = torch.from_numpy(~invalid)
        image[:, ~valid] = -1.0
        return {
            "image": image,
            "target": int(record["target_index"]),
            "class_id": int(record["class_id"]),
            "block_id": record["block_id"],
            "valid_mask": valid,
            "original_size": torch.tensor([h, w]),
        }


def main():
    """Validate a dataset after export or transfer; exit nonzero on corruption."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Verify exported crop files and label pairings")
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    try:
        result = BlockCropDataset(args.folder).validate()
    except Exception as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"Verified {result['count']} crop/label pairs and {result['files_verified']} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
