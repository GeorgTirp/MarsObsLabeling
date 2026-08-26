"""Raster reading: windowed and decimated reads via GDAL/rasterio."""

import math
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

    @property
    def dtype(self) -> str:
        """Band 1's native dtype (e.g. 'uint8', 'uint16')."""
        if self._dataset is None:
            raise RuntimeError("Raster not open")
        return self._dataset.dtypes[0]

    def detect_overviews(self) -> list[int]:
        """Detect available overview levels."""
        if self._dataset is None:
            raise RuntimeError("Raster not open")
        overviews = self._dataset.overviews(1)  # Check band 1
        return sorted(overviews)

    def read_window(
        self, x: int, y: int, width: int, height: int, out_width: int, out_height: int,
        resampling=None,
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
        # Default (nearest) is right for decimation; UPsampling a window to match a
        # model's training GSD needs an interpolating kernel, or the model sees
        # blocky replicated pixels rather than smooth terrain.
        if resampling is None and (out_width > width_clamped or out_height > height_clamped):
            resampling = rasterio.enums.Resampling.bilinear
        read_kwargs = {"window": window, "out_shape": (out_height, out_width)}
        if resampling is not None:
            read_kwargs["resampling"] = resampling
        data = self._dataset.read(1, **read_kwargs)
        return np.asarray(data, dtype=data.dtype)

    def read_window_padded(
        self, x: int, y: int, width: int, height: int, out_width: int, out_height: int,
        resampling=None,
    ) -> np.ndarray:
        """
        Like read_window, but the requested region may extend beyond the image.

        Areas outside the image are padded with black (0) — used by the zoomed-out
        / overview views so you can pan past the swath edges.

        Returns:
            np.ndarray of shape (out_height, out_width), valid data placed proportionally.
        """
        if self._dataset is None:
            raise RuntimeError("Raster not open")

        # Intersection of the requested region with the image bounds
        ix0 = max(0, x)
        iy0 = max(0, y)
        ix1 = min(self.width, x + width)
        iy1 = min(self.height, y + height)

        if ix1 <= ix0 or iy1 <= iy0:
            # Entirely outside the image
            return np.zeros((out_height, out_width), dtype=np.uint8)

        # Where the valid sub-region maps in the output buffer
        sx = out_width / width
        sy = out_height / height
        ox0 = int(round((ix0 - x) * sx))
        oy0 = int(round((iy0 - y) * sy))
        ow = max(1, min(out_width - ox0, int(round((ix1 - ix0) * sx))))
        oh = max(1, min(out_height - oy0, int(round((iy1 - iy0) * sy))))

        data = self.read_window(ix0, iy0, ix1 - ix0, iy1 - iy0, ow, oh, resampling=resampling)
        out = np.zeros((out_height, out_width), dtype=data.dtype)
        out[oy0:oy0 + oh, ox0:ox0 + ow] = data[:oh, :ow]
        return out

    # Cap on the decimated validity mask, in pixels. 64M is ~64 MB as bool and
    # leaves a 1.1-gigapixel HiRISE strip at roughly 1 mask pixel per 5x5 native.
    MAX_MASK_PIXELS = 64_000_000

    def validity_mask(self, max_pixels: int | None = None) -> tuple[np.ndarray, int]:
        """One decimated bool mask of the whole raster: True where data is valid.

        Returns ``(mask, decimation)``. Reading the image once at reduced scale and
        slicing this is O(1) raster reads for any number of blocks, where asking
        each block for its own nodata fraction is O(blocks) windowed decodes --
        the difference between a second and several minutes on a JP2 whose block
        grid is fine.

        Decimation is chosen to respect ``max_pixels``; the mask is therefore a
        coarse estimate, which is all a nodata/off-swath test needs.
        """
        if self._dataset is None:
            raise RuntimeError("Raster not open")
        budget = max_pixels or self.MAX_MASK_PIXELS
        total = self.width * self.height
        decimation = 1
        if total > budget:
            decimation = int(math.ceil(math.sqrt(total / float(budget))))
        out_h = max(1, self.height // decimation)
        out_w = max(1, self.width // decimation)
        data = self._dataset.read(1, out_shape=(out_h, out_w))
        return ~compute_invalid_mask(data), decimation

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
