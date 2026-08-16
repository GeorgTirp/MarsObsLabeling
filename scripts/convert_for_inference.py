"""CLI tool: convert a raster (typically a raw HiRISE JP2) to a tiled GeoTIFF
for fast mars-inference / mars-label reads.

Run this once per source file before mars-inference on a large JP2. See
marslabeler.io.convert for why: JP2 windowed reads are ~100-250x slower than
tiled GeoTIFF reads of the same window, and dominate inference wall time far
more than GPU vs CPU does -- this is very often mistaken for "the GPU isn't
being used" when the GPU is actually just idle waiting on decode.
"""

import argparse
import sys
import time
from pathlib import Path

from marslabeler.io.convert import convert_to_tiled_geotiff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=str, help="Source raster (e.g. a HiRISE .JP2)")
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output path (default: <src stem>.tif next to the source)",
    )
    parser.add_argument(
        "--block-size", type=int, default=512,
        help="Output tile size in pixels (default: 512, matches the default labeling block size)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=4096,
        help="Source read chunk size in pixels (default: 4096; must be a multiple of --block-size)",
    )
    parser.add_argument(
        "--compress", default="deflate", choices=["deflate", "lzw", "zstd", "none"],
    )
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"Error: {src} does not exist", file=sys.stderr)
        sys.exit(1)
    dst = Path(args.out) if args.out else src.with_suffix(".tif")

    t0 = time.time()

    def progress_cb(done, total):
        pct = 100 * done // total
        print(f"\r  {done}/{total} chunks ({pct}%)", end="", flush=True)

    print(f"Converting {src} -> {dst} (block_size={args.block_size}, compress={args.compress})...")
    try:
        convert_to_tiled_geotiff(
            src, dst,
            block_size=args.block_size,
            chunk_size=args.chunk_size,
            compress=args.compress,
            progress_cb=progress_cb,
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone in {time.time() - t0:.0f}s: {dst}")


if __name__ == "__main__":
    main()
