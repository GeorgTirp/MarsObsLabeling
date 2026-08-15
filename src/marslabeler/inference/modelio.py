"""Load an AI4ExoMars stage-3 segmentation checkpoint and wrap it for inference.

Everything here that touches torch or `vision_backend` (AI4ExoMars's model code) is
imported lazily, inside functions -- so importing this module is cheap and mars-label
(which never calls it) doesn't need torch installed at all.

Checkpoint format expected (as written by
`AI4ExoMars/vision_backend/train_stage3_segmentation_finetune.py`)::

    {
        "stage": "stage3_segmentation_finetune",
        "epoch": int,
        "model_state": <state dict>,
        "metrics": {...},
        "config": {"model": {...}, "data": {...}, ...},
    }

Reconstruction itself (num_classes inference, model_kind dispatch, state dict
loading) lives in AI4ExoMars's own
`vision_backend.training.builders.load_segmentation_model_from_checkpoint` -- this
module only adds the bits specific to running inference over image tiles: input
windowing/normalization, and wrapping the model as the numpy-in/numpy-out callables
`inference.engine` expects.

Beyond plain class prediction, two optional per-pixel analyses hang off the same
loaded model:

- softmax confidence (`make_confidence_score_fn`) -- always available, no extra
  calibration needed.
- Mahalanobis/epistemic uncertainty (`make_uncertainty_score_fn`) -- needs a fitted
  `MahalanobisStats` artifact (see `AI4ExoMars/vision_backend/uncertainty/fit_gaussians.py`),
  conventionally saved as a `<checkpoint_stem>.uncertainty.pt` sidecar next to the
  checkpoint. Until such an artifact exists (no trained model yet), `sidecar_path`
  + `load_uncertainty_stats` are the pieces callers use to detect that gracefully.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from marslabeler.inference.engine import InferencePlan

# HybridEncoder (model_kind="simmim") hard-requires H, W divisible by 256 (stride 32 x
# window 8) and raises otherwise. ContextAwareConvNeXtSwinEncoder self-pads internally,
# so 32 is a conservative alignment rather than a hard requirement.
REQUIRED_STRIDE = {"simmim": 256, "context": 32}


class VisionBackendNotFound(ImportError):
    """Raised when the AI4ExoMars `vision_backend` package cannot be located."""


def _ensure_vision_backend_importable(explicit_path: str | None = None) -> None:
    try:
        import vision_backend

        return
    except ModuleNotFoundError:
        pass

    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    # Sibling checkout: .../ESA/MarsObsLabeling (this repo) and .../ESA/AI4ExoMars
    repo_root = Path(__file__).resolve().parents[3]
    candidates.append(repo_root.parent / "AI4ExoMars")

    for candidate in candidates:
        if (candidate / "vision_backend").is_dir():
            sys.path.insert(0, str(candidate))
            try:
                import vision_backend  # noqa: F401

                return
            except ModuleNotFoundError:
                continue

    raise VisionBackendNotFound(
        "Could not import 'vision_backend' (the AI4ExoMars model code). Install it "
        "into this environment with 'pip install -e ../AI4ExoMars' (or wherever your "
        "AI4ExoMars checkout lives), or point configs/app.yaml's inference.ai4exomars_path "
        "(or --ai4exomars-path) at that checkout."
    )


@dataclass
class ModelBundle:
    """A loaded, eval()'d segmentation model plus what the engine needs to feed it."""

    model: Any  # torch.nn.Module
    model_kind: str  # "simmim" | "context"
    num_classes: int
    required_stride: int
    needs_context: bool
    device: Any  # torch.device
    checkpoint_path: Path
    raw_config: dict


def sidecar_path(checkpoint_path: str | Path, suffix: str) -> Path:
    """Conventional calibration-artifact path next to a checkpoint.

    e.g. sidecar_path("checkpoints/stage3.pt", "uncertainty.pt")
         -> checkpoints/stage3.uncertainty.pt
    """
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_suffix("").with_suffix(f".{suffix}")


def load_model_bundle(
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
    ai4exomars_path: str | None = None,
) -> ModelBundle:
    """Load a stage-3 segmentation checkpoint and reconstruct the matching model."""
    import torch

    _ensure_vision_backend_importable(ai4exomars_path)
    from vision_backend.training.builders import load_segmentation_model_from_checkpoint
    from vision_backend.training.utils import select_device

    checkpoint_path = Path(checkpoint_path)
    torch_device = select_device(torch) if device == "auto" else torch.device(device)
    model, model_kind, num_classes, config = load_segmentation_model_from_checkpoint(
        checkpoint_path, device=torch_device
    )
    if model_kind not in REQUIRED_STRIDE:
        raise ValueError(f"Unknown model_kind in checkpoint: {model_kind!r} (expected simmim|context)")

    return ModelBundle(
        model=model,
        model_kind=model_kind,
        num_classes=num_classes,
        required_stride=REQUIRED_STRIDE[model_kind],
        needs_context=(model_kind == "context"),
        device=torch_device,
        checkpoint_path=checkpoint_path,
        raw_config=config,
    )


def build_inference_plan(
    bundle: ModelBundle,
    *,
    block_size: int,
    batch_size: int = 4,
    context_multiplier: int = 4,
) -> InferencePlan:
    """Round the labeling block size up to the model's required stride."""
    stride = bundle.required_stride
    pad_size = -(-block_size // stride) * stride  # ceil(block_size / stride) * stride
    context_px = pad_size * context_multiplier if bundle.needs_context else 0
    return InferencePlan(
        pad_size=pad_size,
        needs_context=bundle.needs_context,
        context_px=context_px,
        batch_size=max(1, batch_size),
    )


def _to_tensor(bundle: ModelBundle, batch: np.ndarray, quantization_bounds: tuple[float, float] | None):
    """window -> normalized tensor, matching AI4ExoMars training
    (`seg_dataset.SegmentationCropDataset`): x / 127.5 - 1.0.

    Non-uint8 windows (e.g. 16-bit HiRISE RDR) are first rescaled to uint8 using
    `quantization_bounds` (see `engine.compute_global_quantization` -- computed once
    per raster, not per-block). Without bounds, non-uint8 input is a hard error:
    silently guessing a per-block stretch would be worse than failing loudly.
    """
    import torch

    if batch.dtype != np.uint8:
        if quantization_bounds is None:
            raise NotImplementedError(
                f"Inference expects uint8 imagery (the AI4ExoMars training convention), "
                f"got dtype {batch.dtype} with no quantization bounds computed. This "
                f"raster needs a global percentile stretch first -- see "
                f"engine.compute_global_quantization()."
            )
        from marslabeler.inference.engine import quantize_to_uint8

        batch = quantize_to_uint8(batch, quantization_bounds)
    x = batch.astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(x).unsqueeze(1)  # (N, 1, H, W)
    return tensor.to(bundle.device)


def _forward_logits(
    bundle: ModelBundle,
    local_batch: np.ndarray,
    context_batch: np.ndarray | None,
    quantization_bounds: tuple[float, float] | None,
):
    local_tensor = _to_tensor(bundle, local_batch, quantization_bounds)
    if bundle.needs_context:
        if context_batch is None:
            raise ValueError(f"model_kind={bundle.model_kind!r} requires a context crop")
        context_tensor = _to_tensor(bundle, context_batch, quantization_bounds)
        return bundle.model(local_tensor, context_tensor)
    return bundle.model(local_tensor)


def make_predict_fn(bundle: ModelBundle, quantization_bounds: tuple[float, float] | None = None):
    """Numpy-in/numpy-out class prediction, for `inference.engine.run_block_inference`."""
    import torch

    def predict_fn(local_batch: np.ndarray, context_batch: np.ndarray | None) -> np.ndarray:
        with torch.no_grad():
            logits = _forward_logits(bundle, local_batch, context_batch, quantization_bounds)
            predicted = logits.argmax(dim=1)
            return predicted.to("cpu").numpy().astype(np.int64)

    return predict_fn


def make_confidence_score_fn(bundle: ModelBundle, quantization_bounds: tuple[float, float] | None = None):
    """Numpy-in/numpy-out max-softmax confidence, for `inference.engine.run_block_scores`.

    Needs no calibration -- available for any loaded model, trained or not.
    """
    import torch
    from vision_backend.uncertainty.uncertainty_mapping import softmax_confidence_map

    def score_fn(local_batch: np.ndarray, context_batch: np.ndarray | None) -> np.ndarray:
        with torch.no_grad():
            logits = _forward_logits(bundle, local_batch, context_batch, quantization_bounds)
            confidence = softmax_confidence_map(logits)
            return confidence.to("cpu").numpy().astype(np.float32)

    return score_fn


def load_uncertainty_stats(path: str | Path, ai4exomars_path: str | None = None):
    """Load a fitted `MahalanobisStats` artifact (see `uncertainty/fit_gaussians.py`).

    Raises FileNotFoundError if the sidecar doesn't exist yet -- expected until a
    model has been trained and calibrated. Callable standalone (doesn't depend on
    load_model_bundle() having already run and put vision_backend on sys.path).
    """
    _ensure_vision_backend_importable(ai4exomars_path)
    from vision_backend.uncertainty.malahanobis import load_stats

    return load_stats(path)


def make_uncertainty_score_fn(
    bundle: ModelBundle, stats, quantization_bounds: tuple[float, float] | None = None
):
    """Numpy-in/numpy-out epistemic (Mahalanobis) uncertainty, normalized to [0, 1],
    for `inference.engine.run_block_scores`. Requires fitted `stats` (see
    `load_uncertainty_stats` / `sidecar_path`)."""
    import torch
    from vision_backend.uncertainty.uncertainty_mapping import epistemic_uncertainty_map

    def score_fn(local_batch: np.ndarray, context_batch: np.ndarray | None) -> np.ndarray:
        local_tensor = _to_tensor(bundle, local_batch, quantization_bounds)
        inputs = (local_tensor,)
        if bundle.needs_context:
            if context_batch is None:
                raise ValueError(f"model_kind={bundle.model_kind!r} requires a context crop")
            inputs = (local_tensor, _to_tensor(bundle, context_batch, quantization_bounds))
        with torch.no_grad():
            heat = epistemic_uncertainty_map(bundle.model, *inputs, stats=stats, normalize=True)
            return heat.to("cpu").numpy().astype(np.float32)

    return score_fn
