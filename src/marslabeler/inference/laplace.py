"""Pixel uncertainty from a fitted diagonal last-layer Laplace posterior.

The sidecar is a tensor dictionary with format ``mars-spatial-laplace-diag-v1``,
``checkpoint_sha256``, ``head_name`` (decoder.head or seg.decoder.head), and
``posterior_variance`` containing weight/bias tensors shaped like that head.
These must be POSTERIOR variances fitted at the checkpoint, including the prior
precision, not a training Fisher/EWC prior. No calibration is inferred here.

Following last-layer Laplace, draw the shared classifier weights from their
Gaussian posterior and report [H(E p) - E H(p)] / log(number of classes).
The backbone runs once per batch. Classifier draws are shared across all tiles
and are evaluated at decoder resolution before upsampling their logits.
Reference: https://aleximmer.com/Laplace/api_reference/lllaplace/
"""

import hashlib
import math
from pathlib import Path

import numpy as np


def checkpoint_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_spatial_posterior(bundle, path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "mars-spatial-laplace-diag-v1":
        raise ValueError(
            "This file is not a supported fitted spatial Laplace posterior. "
            "A Stage-4 training prior or block-head-only posterior cannot be used "
            "as a pixel uncertainty map. The artifact format needs to be checked "
            "against its fitting code before it can be imported."
        )
    name = "seg.decoder.head" if bundle.block_classifier else "decoder.head"
    if payload.get("head_name") != name:
        raise ValueError(f"Spatial Laplace calibration must describe {name}.")
    if payload.get("checkpoint_sha256") != checkpoint_sha256(bundle.checkpoint_path):
        raise ValueError("Laplace calibration belongs to a different checkpoint (SHA-256 mismatch).")
    head = bundle.model.get_submodule(name)
    if not isinstance(head, torch.nn.Conv2d) or head.kernel_size != (1, 1) or head.groups != 1:
        raise ValueError("Spatial Laplace currently requires a plain 1×1 segmentation classifier.")
    variances = payload.get("posterior_variance", {})
    result = {}
    for param_name in ("weight", "bias"):
        param = getattr(head, param_name)
        if param is None:
            if variances.get(param_name) is not None:
                raise ValueError(f"Posterior includes {param_name} absent from the classifier.")
            result[param_name] = None
            continue
        var = variances.get(param_name)
        if not isinstance(var, torch.Tensor) or var.shape != param.shape:
            raise ValueError(f"Laplace {param_name} variance does not match the classifier shape.")
        var = var.float()
        if not torch.isfinite(var).all() or (var < 0).any():
            raise ValueError(f"Laplace {param_name} variance must be finite and nonnegative.")
        result[param_name] = var
    return head, result


def make_laplace_score_fn(bundle, stats_path, quantization_bounds=None, *, samples=64, seed=0):
    import torch
    import torch.nn.functional as F
    from marslabeler.inference.modelio import _to_tensor, _forward_logits
    from vision_backend.model.features import hook_pre_classifier_features

    if samples < 2:
        raise ValueError("Laplace uncertainty needs at least two posterior samples.")
    head, variances = load_spatial_posterior(bundle, stats_path)
    if head.out_channels < 2:
        raise ValueError("Laplace classification uncertainty needs at least two classes.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = []
    for _ in range(samples):
        draw = []
        for name in ("weight", "bias"):
            param, var = getattr(head, name), variances[name]
            if param is None:
                draw.append(None)
            else:
                noise = torch.randn(var.shape, generator=generator) * var.sqrt()
                draw.append((param.detach().float().cpu() + noise).to(bundle.device))
        draws.append(draw)

    def score(local, context):
        with torch.no_grad():
            if bundle.block_classifier:
                features = bundle.model.decode_features(_to_tensor(bundle, local, quantization_bounds))
            else:
                with hook_pre_classifier_features(bundle.model) as captured:
                    _forward_logits(bundle, local, context, quantization_bounds)
                    features = captured["features"]
            features = features.float()
            prob_sum = entropy_sum = None
            for weight, bias in draws:
                logits = F.conv2d(features, weight, bias)
                logits = F.interpolate(logits, size=local.shape[-2:], mode="bilinear", align_corners=False)
                probs = logits.softmax(1)
                entropy = -(probs * probs.clamp_min(1e-12).log()).sum(1)
                if prob_sum is None:
                    prob_sum, entropy_sum = probs, entropy
                else:
                    prob_sum.add_(probs)
                    entropy_sum.add_(entropy)
            mean = prob_sum / samples
            entropy = -(mean * mean.clamp_min(1e-12).log()).sum(1)
            mutual_information = (entropy - entropy_sum / samples) / math.log(head.out_channels)
            return mutual_information.clamp(0, 1).cpu().numpy().astype(np.float32)

    return score
