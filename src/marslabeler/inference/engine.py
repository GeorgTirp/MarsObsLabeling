"""Block-level model inference: read padded windows, run predict_fn, majority-vote per block.

Deliberately torch-free: `predict_fn` is a plain numpy-in/numpy-out callable, so this
module has no dependency on torch or the AI4ExoMars model code (that lives in
`inference/modelio.py`, imported lazily) and can be unit-tested without either.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from marslabeler.io.preprocess import compute_invalid_mask
from marslabeler.model.grid import BlockInfo


class RasterLike(Protocol):
    """The subset of RasterSource this module needs -- lets tests use a stub."""

    def read_window_padded(
        self, x: int, y: int, width: int, height: int, out_width: int, out_height: int
    ) -> np.ndarray: ...


# local_batch: (N, H, W) uint8. context_batch: (N, H, W) uint8 or None.
# Returns: (N, H, W) int array of per-pixel predicted class channel indices.
PredictFn = Callable[[np.ndarray, np.ndarray | None], np.ndarray]

# Same inputs; returns (N, H, W) float array of a continuous per-pixel score
# (softmax confidence, Mahalanobis uncertainty, ...).
ScoreFn = Callable[[np.ndarray, np.ndarray | None], np.ndarray]


@dataclass
class InferencePlan:
    """How large a window to feed the model, and whether it needs a context crop."""

    pad_size: int
    needs_context: bool = False
    context_px: int = 0
    batch_size: int = 4


def _iter_batches(
    raster: RasterLike,
    blocks: Sequence[BlockInfo],
    plan: InferencePlan,
    progress_cb: Callable[[int, int], None] | None,
    should_cancel: Callable[[], bool] | None,
):
    """Read (local, context) window batches for `blocks`, advancing progress per batch.

    Shared by `run_block_inference` and `run_block_scores` -- both anchor each
    block's window at its top-left pixel (native resolution, zero-padded past
    image/panel edges), so a block's own `w_px` x `h_px` region always sits at the
    crop's top-left corner regardless of padding.
    """
    total = len(blocks)
    for batch_start in range(0, total, plan.batch_size):
        if should_cancel is not None and should_cancel():
            return

        batch = blocks[batch_start : batch_start + plan.batch_size]
        local_stack = np.stack(
            [
                raster.read_window_padded(
                    b.x_px, b.y_px, plan.pad_size, plan.pad_size, plan.pad_size, plan.pad_size
                )
                for b in batch
            ]
        )

        context_stack = None
        if plan.needs_context:
            context_stack = np.stack([_context_window(raster, b, plan) for b in batch])

        yield batch, local_stack, context_stack

        if progress_cb is not None:
            progress_cb(min(batch_start + len(batch), total), total)


def run_block_inference(
    raster: RasterLike,
    blocks: Sequence[BlockInfo],
    plan: InferencePlan,
    predict_fn: PredictFn,
    progress_cb: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, int]:
    """Run `predict_fn` over `blocks` in batches, majority-voting each block's crop.

    Returns block_id -> predicted class channel index (0-based, as emitted by the
    model -- caller maps this through ClassScheme.model_index_to_id()).
    """
    results: dict[str, int] = {}
    for batch, local_stack, context_stack in _iter_batches(
        raster, blocks, plan, progress_cb, should_cancel
    ):
        predicted = predict_fn(local_stack, context_stack)
        if predicted.shape[0] != len(batch):
            raise ValueError(
                f"predict_fn returned {predicted.shape[0]} results for a batch of "
                f"{len(batch)} blocks"
            )
        for block, pixel_classes in zip(batch, predicted):
            crop = pixel_classes[: block.h_px, : block.w_px]
            results[block.block_id] = _majority_class(crop)
    return results


def run_block_scores(
    raster: RasterLike,
    blocks: Sequence[BlockInfo],
    plan: InferencePlan,
    score_fn: ScoreFn,
    progress_cb: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, float]:
    """Run `score_fn` over `blocks` in batches, averaging each block's crop.

    Same windowing as `run_block_inference`, but for continuous per-pixel scores
    (softmax confidence, Mahalanobis uncertainty, ...) instead of discrete class
    predictions -- each block gets the mean score over its own pixel region.

    Returns block_id -> mean score.
    """
    results: dict[str, float] = {}
    for batch, local_stack, context_stack in _iter_batches(
        raster, blocks, plan, progress_cb, should_cancel
    ):
        scored = score_fn(local_stack, context_stack)
        if scored.shape[0] != len(batch):
            raise ValueError(
                f"score_fn returned {scored.shape[0]} results for a batch of {len(batch)} blocks"
            )
        for block, pixel_scores in zip(batch, scored):
            crop = pixel_scores[: block.h_px, : block.w_px]
            results[block.block_id] = float(np.mean(crop)) if crop.size else 0.0
    return results


def compute_global_quantization(
    raster: RasterLike,
    width: int,
    height: int,
    *,
    sample_size: int = 2000,
    percentiles: tuple[int, int] = (1, 99),
) -> tuple[float, float]:
    """Robust global percentile-stretch bounds for non-uint8 rasters.

    AI4ExoMars trains on 8-bit DRG (Digital Rectified Graphic) mosaics -- a
    cartographic, contrast-enhanced product -- not raw calibrated 16-bit RDR
    imagery, so there is no exact recipe to convert one into the other; this is
    a best-effort approximation, not a reproduction of the DRG pipeline.

    Computed once from a single decimated read of the *whole* image (never
    per-block: a per-block stretch would equalize real brightness/albedo
    differences between blocks, destroying signal the model may rely on), so
    every block in a run gets the same fixed mapping to uint8.
    """
    decimated = raster.read_window_padded(0, 0, width, height, sample_size, sample_size)
    invalid = compute_invalid_mask(decimated)
    valid = decimated[~invalid]
    if valid.size == 0:
        return (0.0, 1.0)
    p_low, p_high = np.percentile(valid.astype(np.float64), percentiles)
    if p_high <= p_low:
        p_high = p_low + 1.0
    return float(p_low), float(p_high)


def quantize_to_uint8(batch: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """Rescale a non-uint8 window into uint8 [0,255] using fixed (p_low, p_high) bounds."""
    p_low, p_high = bounds
    clipped = np.clip(batch.astype(np.float64), p_low, p_high)
    return ((clipped - p_low) / (p_high - p_low) * 255.0).round().astype(np.uint8)


def _context_window(raster: RasterLike, block: BlockInfo, plan: InferencePlan) -> np.ndarray:
    """Larger surrounding crop for context-branch models, decimated to pad_size."""
    cx = block.x_px + block.w_px // 2
    cy = block.y_px + block.h_px // 2
    half = plan.context_px // 2
    return raster.read_window_padded(
        cx - half, cy - half, plan.context_px, plan.context_px, plan.pad_size, plan.pad_size
    )


def _majority_class(crop: np.ndarray) -> int:
    """Most frequent predicted class index within a block's crop."""
    if crop.size == 0:
        return 0
    counts = np.bincount(crop.ravel().astype(np.int64))
    return int(np.argmax(counts))
