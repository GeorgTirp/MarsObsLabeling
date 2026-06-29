"""Raster reading: windowed and decimated reads via GDAL/rasterio."""

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.transform import Affine

from marslabeler.io.preprocess import compute_invalid_mask


class RasterSource:
    """Wraps a raster (JP2, GeoTIFF) via rasterio for efficient windowed reads."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._dataset: Optional[rasterio.DatasetReader] = None

    def open(self) -> None:
        """Open the raster and read metadata."""
        if self._dataset is not None:
            return
        self._dataset = rasterio.open(str(self.path))

    def close(self) -> None:
        """Close the raster."""
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def width(self) -> int:
        """Image width in pixels."""
        if self._dataset is None:
            raise RuntimeError("Raster not open")
        return self._dataset.width

    @property
    def height(self) -> int:
        """Image height in pixels."""
        if self._dataset is None:
            raise RuntimeError("Raster not open")
        return self._dataset.height

    @property
    def transform(self) -> Affine:
        """Geotransform (affine)."""
        if self._dataset is None:
            raise RuntimeError("Raster not open")
        return self._dataset.transform

    @property
    def crs(self):
        """Coordinate reference system."""
        if self._dataset is None:
            raise RuntimeError("Raster not open")
        return self._dataset.crs

    @property
    def nodata(self) -> Optional[float]:
        """Nodata value."""
        if self._dataset is None:
            raise RuntimeError("Raster not open")
        return self._dataset.nodata

    @property
    def gsd(self) -> float:
        """Ground sample distance (metres/pixel) from affine transform."""
        if self._dataset is None:
            raise RuntimeError("Raster not open")
        # Pixel size from affine (diagonal elements typically, take absolute value)
        return abs(self._dataset.transform.a)

    def detect_overviews(self) -> list[int]:
        """Detect available overview levels."""
        if self._dataset is None:
            raise RuntimeError("Raster not open")
        overviews = self._dataset.overviews(1)  # Check band 1
        return sorted(overviews)

    def read_window(
        self, x: int, y: int, width: int, height: int, out_width: int, out_height: int
    ) -> np.ndarray:
        """
        Read a window from the raster with optional decimation via GDAL.

        Args:
            x, y: top-left pixel coords in the full image
            width, height: window size in full-image pixels
            out_width, out_height: output size (decimation if < window size)

        Returns:
            np.ndarray of shape (out_height, out_width), dtype uint8 (or the native band dtype)
        """
        if self._dataset is None:
            raise RuntimeError("Raster not open")

        # Clamp to image bounds
        x_clamped = max(0, min(x, self.width - 1))
        y_clamped = max(0, min(y, self.height - 1))
        width_clamped = min(width, self.width - x_clamped)
        height_clamped = min(height, self.height - y_clamped)

        if width_clamped <= 0 or height_clamped <= 0:
            return np.zeros((out_height, out_width), dtype=np.uint8)

        # Use rasterio's windowed read with output size (GDAL decimation)
        window = rasterio.windows.Window(x_clamped, y_clamped, width_clamped, height_clamped)
        data = self._dataset.read(1, window=window, out_shape=(out_height, out_width))
        return np.asarray(data, dtype=data.dtype)

    def nodata_fraction(self, x: int, y: int, width: int, height: int) -> float:
        """
        Estimate nodata fraction in a window using a decimated read.

        Uses dtype-aware detection aligned with AI4ExoMars preprocessing:
        - uint8: marks 0 and 255 as invalid
        - uint16: marks 0 as invalid
        - float: marks non-finite and <= -3.0e38 as invalid

        Args:
            x, y, width, height: window in full-image pixel coords

        Returns:
            Fraction of nodata pixels (0.0 to 1.0)
        """
        if self._dataset is None:
            raise RuntimeError("Raster not open")

        # Read decimated version for speed
        decimated = self.read_window(x, y, width, height, 64, 64)
        if decimated.size == 0:
            return 1.0

        # Use AI4ExoMars dtype-aware invalid mask
        invalid_mask = compute_invalid_mask(decimated)
        nodata_count = np.sum(invalid_mask)
        return float(nodata_count) / decimated.size

    def variance(self, x: int, y: int, width: int, height: int) -> float:
        """
        Estimate variance in a window using a decimated read.

        Useful for detecting featureless (saturated/uniform) blocks.

        Args:
            x, y, width, height: window in full-image pixel coords

        Returns:
            Variance of pixel values (excluding nodata)
        """
        if self._dataset is None:
            raise RuntimeError("Raster not open")

        # Read decimated version
        decimated = self.read_window(x, y, width, height, 64, 64)
        if decimated.size == 0:
            return 0.0

        nodata_val = self.nodata
        if nodata_val is not None:
            mask = decimated != nodata_val
            if not mask.any():
                return 0.0
            pixels = decimated[mask]
        else:
            pixels = decimated

        return float(np.var(pixels))
