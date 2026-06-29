"""Export labeled blocks as probe set for training."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from marslabeler.io.raster import RasterSource
from marslabeler.model.grid import Grid
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raster and metadata
    raster = RasterSource(jp2_path)
    raster.open()

    # Load labels
    table = pq.read_table(str(parquet_path))

    # Load classes for metadata
    classes_scheme = load_classes(classes_yaml)

    # Get grid info from first block
    first_row = table.slice(0, 1).to_pydict()
    obs_id = first_row["obs_id"][0]
    img_width = raster.width
    img_height = raster.height
    gsd = raster.gsd

    # Infer block size from first block's w_px
    block_width = first_row["w_px"][0]
    block_height = first_row["h_px"][0]

    # Get panel/block sizes (assume uniform)
    panel_size = 4096  # Default
    block_size = 512   # Default

    grid = Grid(img_width, img_height, panel_size, block_size, obs_id, raster.transform, raster.crs)

    # Export labeled blocks as crops
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    labels_csv_lines = ["block_id,x_px,y_px,class_id,class_name,confidence"]
    crop_count = 0

    for i in range(len(table)):
        row = table.slice(i, 1).to_pydict()

        block_id = row["block_id"][0]
        x_px = row["x_px"][0]
        y_px = row["y_px"][0]
        w_px = row["w_px"][0]
        h_px = row["h_px"][0]
        class_id = row["class_id"][0]
        class_name = row["class_name"][0]
        status = row["status"][0]

        # Skip unlabeled, abstain, and nodata
        if status != "labeled":
            continue

        # Read native-resolution block
        block_data = raster.read_window(x_px, y_px, w_px, h_px, w_px, h_px)

        # Save as PNG
        img = Image.fromarray(block_data, mode="L")
        crop_filename = f"{block_id}.png"
        crop_path = crops_dir / crop_filename
        img.save(crop_path)

        # Write CSV line (confidence=1.0 for v1 since all labeled blocks are included)
        labels_csv_lines.append(
            f"{block_id},{x_px},{y_px},{class_id},{class_name},1.0"
        )

        crop_count += 1

    # Write CSV
    csv_path = output_dir / "labels.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(labels_csv_lines) + "\n")

    # Write class metadata
    classes_list = []
    for class_id, class_obj in sorted(classes_scheme.classes.items()):
        classes_list.append({
            "id": class_id,
            "name": class_obj.name,
            "color": class_obj.color,
        })

    meta_path = output_dir / "classes.json"
    with open(meta_path, "w") as f:
        json.dump({"classes": classes_list}, f, indent=2)

    # Summary
    print(f"Exported probe set to {output_dir}")
    print(f"  Crops: {crop_count} PNG files")
    print(f"  Labels: {csv_path}")
    print(f"  Classes: {meta_path}")

    raster.close()


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
