"""Convert a raster (typically a raw HiRISE JP2) to a tiled, compressed GeoTIFF.

JPEG2000 windowed reads at native resolution are slow -- each windowed read pays
a wavelet-decode cost with little reuse across nearby windows, unlike a tiled
GeoTIFF's direct block access. This is why `mars-inference` can appear to "not
use the GPU": the GPU sits idle waiting on JP2 decode between batches (confirmed
in practice: ~0.7s per native-resolution 512x512 JP2 read vs ~0.003s from a tiled
GeoTIFF of the same data -- roughly 250x). AI4ExoMars's own training pipeline
never reads JP2 directly for this reason either (`noahh_alignment/warp_drg.py`
produces a tiled, compressed GeoTIFF up front).

Converting reads the source in large chunks (not block-by-block, which would
make conversion itself slow for the same reason) and writes it out tiled to
`block_size` -- matching the labeling/inference block grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import rasterio
from rasterio.windows import Window


def convert_to_tiled_geotiff(
    src_path: str | Path,
    dst_path: str | Path,
    *,
    block_size: int = 512,
    chunk_size: int = 4096,
    compress: str = "deflate",
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Stream-copy src_path (band 1) into a tiled, compressed GeoTIFF at dst_path.

    Reads in `chunk_size`-square windows (large, to amortize JP2 decode cost) and
    writes in `block_size`-square tiles -- never materializes the whole raster in
    RAM. `chunk_size` should be a multiple of `block_size`.

    Parameters
    ----------
    progress_cb : optional callable(done, total) in chunks (not blocks).
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if chunk_size % block_size != 0:
        raise ValueError(f"chunk_size ({chunk_size}) must be a multiple of block_size ({block_size})")

    with rasterio.open(str(src_path)) as src:
        profile = dict(
            driver="GTiff",
            width=src.width,
            height=src.height,
            count=1,
            dtype=src.dtypes[0],
            crs=src.crs,
            transform=src.transform,
            nodata=src.nodata,
            tiled=True,
            blockxsize=block_size,
            blockysize=block_size,
            compress=compress,
            predictor=2 if compress in ("deflate", "lzw", "zstd") else 1,
            BIGTIFF="YES",
        )

        n_rows = (src.height + chunk_size - 1) // chunk_size
        n_cols = (src.width + chunk_size - 1) // chunk_size
        total = n_rows * n_cols
        done = 0

        with rasterio.open(str(dst_path), "w", **profile) as dst:
            for row in range(0, src.height, chunk_size):
                h = min(chunk_size, src.height - row)
                for col in range(0, src.width, chunk_size):
                    w = min(chunk_size, src.width - col)
                    win = Window(col, row, w, h)
                    data = src.read(1, window=win)
                    dst.write(data, 1, window=win)
                    done += 1
                    if progress_cb is not None:
                        progress_cb(done, total)
