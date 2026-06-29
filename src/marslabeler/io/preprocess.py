"""Preprocessing utilities: invalid pixel detection, normalization (AI4ExoMars-aligned)."""

import numpy as np


def compute_invalid_mask(array: np.ndarray) -> np.ndarray:
    """
    Mark NULL / saturation pixels (dtype-aware, aligned with AI4ExoMars).

    HiRISE encodes NULL/saturation as special DN values by bit depth:
    - 8-bit (uint8): 0 = NULL/LRS/LIS (black), 255 = HIS/HRS (white)
    - 16-bit (uint16): Only 0 = NULL
    - float: Non-finite or <= -3.0e38

    Args:
        array: Image array (2D or 3D)

    Returns:
        Boolean mask where True = invalid/nodata pixel
    """
    arr = np.asarray(array)

    if np.issubdtype(arr.dtype, np.floating):
        return (~np.isfinite(arr)) | (arr <= -3.0e38)

    # Integer types: check by itemsize (byte count)
    itemsize = np.dtype(arr.dtype).itemsize
    markers = (0, 255) if itemsize == 1 else (0,)

    if arr.ndim == 2:
        # Single channel
        mask = np.zeros(arr.shape, dtype=bool)
        for value in markers:
            mask |= arr == value
        return mask
    elif arr.ndim == 3:
        # Multi-channel: mark pixel as invalid only if invalid across ALL channels
        mask = np.ones(arr.shape[:2], dtype=bool)
        for value in markers:
            channel_mask = np.all(arr == value, axis=2)
            mask &= channel_mask
        return mask
    else:
        raise ValueError(f"Unexpected array shape: {arr.shape}")


def apply_display_stretch_with_mask(
    data: np.ndarray,
    percentiles: tuple[int, int] = (1, 99),
    invalid_mask: np.ndarray = None,
) -> np.ndarray:
    """
    Apply percentile-based display stretch, ignoring invalid pixels.

    Args:
        data: Input array
        percentiles: (low, high) percentiles for stretch
        invalid_mask: Optional boolean mask where True = pixels to ignore

    Returns:
        uint8 array stretched to [0, 255]
    """
    if data.size == 0:
        return np.zeros_like(data, dtype=np.uint8)

    arr = data.astype(np.float32)

    # Use valid pixels for percentile calculation
    if invalid_mask is not None:
        valid = arr[~invalid_mask]
    else:
        valid = arr.ravel()

    if valid.size == 0:
        return np.ones_like(data, dtype=np.uint8) * 128

    p_low, p_high = np.percentile(valid, percentiles)

    if p_low == p_high:
        stretched = np.ones_like(arr, dtype=np.uint8) * 128
    else:
        # Clip and stretch
        clipped = np.clip(arr, p_low, p_high)
        stretched = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)

    # Mark invalid pixels as black
    if invalid_mask is not None:
        stretched[invalid_mask] = 0

    return stretched
