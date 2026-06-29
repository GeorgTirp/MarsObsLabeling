"""Overview management: detect and build external overviews for fast decimated reads."""

import subprocess
from pathlib import Path


def has_internal_overviews(raster_path: str | Path) -> bool:
    """Check if a raster has internal overviews."""
    import rasterio

    raster_path = Path(raster_path)
    try:
        with rasterio.open(str(raster_path)) as src:
            overviews = src.overviews(1)
            return len(overviews) > 0
    except Exception:
        return False


def build_overviews_gdaladdo(
    raster_path: str | Path, levels: list[int] = None, resampling: str = "average"
) -> None:
    """
    Build external overviews using gdaladdo CLI.

    Args:
        raster_path: Path to the raster
        levels: Overview levels (e.g. [2, 4, 8, 16, 32])
        resampling: Resampling algorithm (e.g. 'average', 'nearest')
    """
    if levels is None:
        levels = [2, 4, 8, 16, 32]

    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    cmd = ["gdaladdo", "-r", resampling, str(raster_path)] + [str(level) for level in levels]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gdaladdo failed: {result.stderr}")
