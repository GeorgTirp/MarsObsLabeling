"""CLI tool to build external overviews for a raster directory."""

import argparse
import sys
from pathlib import Path

from marslabeler.io.overviews import build_overviews_gdaladdo, has_internal_overviews


def main():
    parser = argparse.ArgumentParser(
        description="Build external overviews for raster files using gdaladdo"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a raster file or directory of rasters",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32],
        help="Overview levels (default: 2 4 8 16 32)",
    )
    parser.add_argument(
        "--resampling",
        type=str,
        default="average",
        help="Resampling algorithm (default: average)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Rebuild overviews even if they exist",
    )

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        print(f"Error: {path} does not exist", file=sys.stderr)
        sys.exit(1)

    raster_paths = []
    if path.is_file():
        raster_paths = [path]
    elif path.is_dir():
        # Find all JP2, GeoTIFF, etc.
        raster_paths = list(path.glob("**/*.jp2")) + list(path.glob("**/*.tif*"))

    if not raster_paths:
        print(f"No rasters found in {path}", file=sys.stderr)
        sys.exit(1)

    for raster_path in raster_paths:
        if not args.force and has_internal_overviews(raster_path):
            print(f"Skipping {raster_path} (overviews already exist)")
            continue

        print(f"Building overviews for {raster_path}...")
        try:
            build_overviews_gdaladdo(raster_path, args.levels, args.resampling)
            print(f"  ✓ Done")
        except Exception as e:
            print(f"  ✗ Error: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"Processed {len(raster_paths)} raster(s)")


if __name__ == "__main__":
    main()
