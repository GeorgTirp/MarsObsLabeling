"""Export labeled blocks as probe set for training."""

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from marslabeler.io.raster import RasterSource
from marslabeler.model.labelstore import LabelStore
from marslabeler.classes import load_classes


def export_probe_set(
    jp2_path: Path,
    parquet_path: Path,
    classes_yaml: Path,
    output_dir: Path,
    min_confidence: float = 0.0,
) -> None:
    """
    Export labeled blocks as crops for training (probe set per Model v2 §6.1).

    Args:
        jp2_path: Path to JP2 observation
        parquet_path: Path to labels Parquet file
        classes_yaml: Path to classes legend YAML
        output_dir: Output directory for crops
        min_confidence: Minimum confidence to include (not used in v1, reserved for future)
    """
    import csv
    import tempfile

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
        for key, actual in (("img_width", raster.width), ("img_height", raster.height)):
            if key in metadata and metadata[key] != actual:
                raise ValueError(f"Source image {key} does not match saved labels")
        if raster.dtype not in ("uint8", "uint16"):
            raise ValueError(f"PNG crop export requires uint8 or uint16 imagery, got {raster.dtype}")

        seen = set()
        labeled = []
        for row in rows:
            bid = row["block_id"]
            if bid in seen:
                raise ValueError(f"Duplicate block id: {bid}")
            seen.add(bid)
            if row["obs_id"] != Path(jp2_path).stem:
                raise ValueError(f"Label observation {row['obs_id']} does not match the source image")
            x, y, w, h = (row[k] for k in ("x_px", "y_px", "w_px", "h_px"))
            if not all(isinstance(v, int) for v in (x, y, w, h)):
                raise ValueError(f"Invalid block coordinates: {bid}")
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

        # Publish a complete set only after every crop and both manifests succeed.
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".probe-", dir=output_dir.parent) as staging:
            staged = Path(staging) / "export"
            crops_dir = staged / "crops"
            crops_dir.mkdir(parents=True)
            fields = ["block_id", "x_px", "y_px", "class_id", "class_name", "confidence",
                      "w_px", "h_px", "crop_filename", "dtype"]
            with (staged / "labels.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for row in labeled:
                    x, y, w, h = (row[k] for k in ("x_px", "y_px", "w_px", "h_px"))
                    data = raster.read_window(x, y, w, h, w, h)
                    filename = row["block_id"] + ".png"
                    # Let Pillow infer L or I;16: forcing L reinterprets uint16 bytes.
                    Image.fromarray(data).save(crops_dir / filename)
                    writer.writerow({"block_id": row["block_id"], "x_px": x, "y_px": y,
                                     "w_px": w, "h_px": h, "class_id": row["class_id"],
                                     "class_name": classes_scheme.get_name(row["class_id"]),
                                     "confidence": 1.0, "crop_filename": filename,
                                     "dtype": raster.dtype})
            classes = [{"id": cid, "name": c.name, "color": c.color}
                       for cid, c in sorted(classes_scheme.classes.items())]
            (staged / "classes.json").write_text(json.dumps({"classes": classes}, indent=2))
            staged.replace(output_dir)
    print(f"Exported {len(labeled)} native-resolution crops to {output_dir}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export labeled blocks as probe set for training"
    )
    parser.add_argument("jp2_path", type=str, help="Path to JP2 observation")
    parser.add_argument("parquet_path", type=str, help="Path to labels Parquet file")
    parser.add_argument(
        "--classes",
        type=str,
        default="configs/classes.yaml",
        help="Path to classes YAML (default: configs/classes.yaml)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="probe_set",
        help="Output directory (default: probe_set)",
    )

    args = parser.parse_args()

    try:
        export_probe_set(
            Path(args.jp2_path),
            Path(args.parquet_path),
            Path(args.classes),
            Path(args.output),
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
