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


def _resolve_ai4exomars_root(explicit_path: str | None = None) -> Path | None:
    """Best-effort AI4ExoMars repo root -- for resolving paths a checkpoint's
    own saved config stores relative to it (e.g. loader_config_path,
    imagery_path). Distinct from _ensure_vision_backend_importable: that one
    only needs `vision_backend` to become importable (which needs no known
    root at all when it's already `pip install`'d into this environment), so
    it can't be reused here directly.

    Tries, in order: an explicit path, vision_backend's own install location
    (if already importable -- works for both a regular and an editable
    install), then the same sibling-checkout convention
    _ensure_vision_backend_importable() falls back to. Returns None (never
    raises) if nothing pans out.
    """
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    try:
        import vision_backend

        candidates.append(Path(vision_backend.__file__).resolve().parent.parent)
    except ModuleNotFoundError:
        pass
    repo_root = Path(__file__).resolve().parents[3]
    candidates.append(repo_root.parent / "AI4ExoMars")

    for candidate in candidates:
        if (candidate / "vision_backend").is_dir():
            return candidate
    return None


def _resolve_relative_to(path_str: str, root: Path | None) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute() or root is None:
        return path
    return root / path


def resolve_training_imagery_path(
    checkpoint_path: str | Path, *, ai4exomars_path: str | None = None
) -> Path | None:
    """Best-effort: the imagery raster this checkpoint's Stage-3 training run
    (and, by convention, its Neural-PCA calibration pass --
    `fit_neural_pca.py`'s own examples reuse the same `--imagery-path`) point
    at. Lets a Neural-PCA gallery's `source_id` strings (each built from
    `Path(imagery_path).stem` + a pixel col/row, see
    `AI4ExoMars/vision_backend/pc_align/fit_neural_pca.py`) be traced back to
    a real file on disk, for "jump to this gallery thumbnail's location"
    navigation -- see npca_gallery.parse_npca_source_id.

    Returns None (never raises) if anything along the chain is missing: the
    checkpoint has no saved `config["data"]["loader_config_path"]`, that file
    doesn't exist, or the `imagery_path` it names doesn't exist -- this is
    best-effort provenance, not a guaranteed contract, and a checkpoint
    trained before this convention (or moved/retrained data) is expected to
    fail gracefully here rather than raise.
    """
    import json

    import torch

    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except Exception:
        return None
    config = checkpoint.get("config", {}) or {}
    loader_config_path = (config.get("data", {}) or {}).get("loader_config_path")
    if not loader_config_path:
        return None

    root = _resolve_ai4exomars_root(ai4exomars_path)
    loader_config_file = _resolve_relative_to(loader_config_path, root)
    if not loader_config_file.exists():
        return None
    try:
        loader_config = json.loads(loader_config_file.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    imagery_path = loader_config.get("imagery_path")
    if not imagery_path:
        return None
    imagery_file = _resolve_relative_to(imagery_path, root)
    return imagery_file if imagery_file.exists() else None


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
    gsd_ratio: float = 1.0,
) -> InferencePlan:
    """Round the labeling block size up to the model's required stride.

    ``gsd_ratio`` is ``observation_gsd / training_gsd``. At 1.0 the model reads
    native pixels, as before. Otherwise the window read from the observation is
    divided by the ratio and resampled up to ``pad_size``, so each model pixel
    spans the ground distance the model was trained on -- a coarser observation
    (ratio > 1) is read in smaller native windows and magnified.
    """
    stride = bundle.required_stride
    pad_size = -(-block_size // stride) * stride  # ceil(block_size / stride) * stride
    context_px = pad_size * context_multiplier if bundle.needs_context else 0
    native_window = pad_size
    if gsd_ratio and gsd_ratio > 0 and abs(gsd_ratio - 1.0) > 1e-6:
        native_window = max(1, int(round(pad_size / gsd_ratio)))
    return InferencePlan(
        pad_size=pad_size,
        needs_context=bundle.needs_context,
        context_px=context_px,
        batch_size=max(1, batch_size),
        native_window=native_window,
    )


def native_block_size_for_gsd(block_size: int, stride: int, gsd_ratio: float) -> int:
    """Block size (native px) whose ground extent matches one model window.

    When a scale correction is active the model window covers fewer native pixels
    than before, so the labeling grid has to shrink to match -- otherwise a block
    would extend past the window that predicted it.
    """
    pad_size = -(-block_size // stride) * stride
    if not gsd_ratio or gsd_ratio <= 0 or abs(gsd_ratio - 1.0) <= 1e-6:
        return block_size
    return max(1, int(round(pad_size / gsd_ratio)))


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


# Mars equatorial radius (m), for converting a geographic transform's degrees
# into metres. NOAH-H's own source tiles are in a geographic CRS while the
# warped label grid is projected, so a GSD comparison has to handle both.
MARS_RADIUS_M = 3396190.0


def raster_gsd_metres(dataset) -> float | None:
    """Pixel size in metres for an open rasterio dataset, or None if unknowable.

    A projected CRS already reports metres. A geographic one reports degrees, so
    it is converted along a meridian (latitude degrees are very nearly constant
    on a sphere, unlike longitude), which is accurate enough to tell 0.25 m from
    0.5 m.
    """
    try:
        pixel = abs(dataset.transform.a)
    except Exception:
        return None
    if not pixel or pixel <= 0:
        return None
    crs = getattr(dataset, "crs", None)
    if crs is not None and crs.is_geographic:
        import math

        return pixel * (math.pi / 180.0) * MARS_RADIUS_M
    return pixel


def training_gsd_metres(
    checkpoint_path: str | Path, *, ai4exomars_path: str | None = None
) -> float | None:
    """GSD of the imagery this checkpoint was trained on, or None if unresolvable.

    Reads the training raster named by the checkpoint's own saved loader config
    (see `resolve_training_imagery_path`) and measures it, so the comparison is
    against what the model actually saw rather than an assumed constant.
    """
    import rasterio

    imagery_path = resolve_training_imagery_path(
        checkpoint_path, ai4exomars_path=ai4exomars_path
    )
    if imagery_path is None or not Path(imagery_path).exists():
        return None
    try:
        with rasterio.open(str(imagery_path)) as dataset:
            return raster_gsd_metres(dataset)
    except Exception:
        return None


def make_predict_fn(bundle: ModelBundle, quantization_bounds: tuple[float, float] | None = None):
    """Numpy-in/numpy-out class prediction, for `inference.engine.run_block_inference`."""
    import torch

    def predict_fn(local_batch: np.ndarray, context_batch: np.ndarray | None) -> np.ndarray:
        with torch.no_grad():
            logits = _forward_logits(bundle, local_batch, context_batch, quantization_bounds)
            predicted = logits.argmax(dim=1)
            return predicted.to("cpu").numpy().astype(np.int64)

    return predict_fn


def make_predict_fn_with_embeddings(
    bundle: ModelBundle, quantization_bounds: tuple[float, float] | None = None
):
    """`make_predict_fn`, plus a sink collecting each block's pooled feature vector.

    Returns ``(predict_fn, embeddings)``. ``embeddings`` is a list that gains one
    ``[batch, F] float32`` array per batch, in the order the engine feeds them --
    so concatenating it row-aligns with the `blocks` sequence passed to
    `run_block_inference*`.

    The features come off the SAME forward pass that produces the predictions
    (captured on `decoder.head`'s pre-hook, then mean-pooled over H and W), so
    this costs one extra mean and a copy per batch -- no second pass over the
    imagery. That pooled vector is phi(x) in the Neural-PCA method; psi_k(x) for
    any class is just an elementwise product with that class's classifier weight
    vector, so one capture serves every class.
    """
    import torch
    from vision_backend.model.features import hook_pre_classifier_features

    embeddings: list[np.ndarray] = []

    def predict_fn(local_batch: np.ndarray, context_batch: np.ndarray | None) -> np.ndarray:
        with torch.no_grad():
            with hook_pre_classifier_features(bundle.model) as captured:
                logits = _forward_logits(
                    bundle, local_batch, context_batch, quantization_bounds
                )
                features = captured["features"]  # [B, F, H, W]
                embeddings.append(
                    features.mean(dim=(2, 3)).to("cpu").numpy().astype(np.float32)
                )
            return logits.argmax(dim=1).to("cpu").numpy().astype(np.int64)

    return predict_fn, embeddings


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
