"""Export labeled blocks as a portable crop/label dataset."""

import argparse
import sys
from pathlib import Path

from marslabeler.model.cropexport import export_probe_set


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
        count = export_probe_set(
            Path(args.jp2_path),
            Path(args.parquet_path),
            Path(args.classes),
            Path(args.output),
        )
        print(f"Exported {count} native-resolution crops to {args.output}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
