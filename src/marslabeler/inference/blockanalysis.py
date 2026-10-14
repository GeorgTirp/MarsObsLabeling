"""Block-level uncertainty of a Stage-4 block classifier, computed from phi.

`BlockClassifier.block_features` gives every block phi, the input of the head's
last layer, with logits = offset + fc(phi). The prediction pass keeps phi and the
logits of every block, so both scores here need no second pass over the imagery:

- Laplace (`<stem>.laplace.pt`, AI4ExoMars uncertainty/block_laplace.py): the
  last-layer Laplace posterior's Monte Carlo predictive entropy divided by
  log(number of classes), in [0, 1]. Of the model card's scores, it is the best
  at flagging likely misclassifications.
- Mahalanobis (`<stem>.uncertainty.pt`): the distance of phi to the nearest
  class Gaussian, divided by the file's reference distance (the 95th
  percentile of the training blocks' own distances), so values above 1 are
  unlike nearly every training block.

Everything touching torch or `vision_backend` is imported lazily, like modelio.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from marslabeler.io.atomic import atomic_output

LAPLACE_SAMPLES = 1000  # the model card's checks agree to 0.001 AUROC from 1000 to 2000
_CHUNK = 256  # bounds the [chunk, samples, classes] float64 Monte Carlo tensor
# Mahalanobis display ceiling: 1 (yellow) = the training blocks' 95th
# percentile; another site's blocks measure about 2 or more (red).
MAHALANOBIS_DISPLAY_MAXIMUM = 2.0


@dataclass
class BlockFeatures:
    """phi and logits of the predicted blocks, row-aligned with `block_ids`."""

    block_ids: list[str]
    phi: np.ndarray  # [N, D] float32
    logits: np.ndarray  # [N, K] float32
    head_weight: np.ndarray  # [K, D + 1]: [fc.weight | fc.bias] the rows came from

    def __len__(self) -> int:
        return len(self.block_ids)

    @classmethod
    def from_batches(cls, blocks, phi_batches, logit_batches, model) -> BlockFeatures:
        phi = np.concatenate(phi_batches).astype(np.float32) if phi_batches else np.zeros((0, 0), np.float32)
        logits = (np.concatenate(logit_batches).astype(np.float32)
                  if logit_batches else np.zeros((0, 0), np.float32))
        if len(phi) != len(logits) or len(phi) > len(blocks):
            raise ValueError(f"captured {len(phi)} phi and {len(logits)} logit rows for {len(blocks)} blocks")
        fc = model.fc
        head_weight = np.concatenate(
            [fc.weight.detach().float().cpu().numpy(), fc.bias.detach().float().cpu().numpy()[:, None]], axis=1
        )
        return cls([b.block_id for b in blocks[: len(phi)]], phi, logits, head_weight)

    def save(self, path: Path, model_sig: str) -> None:
        with atomic_output(path) as temporary:
            # A file handle: np.savez would append ".npz" to the temporary name.
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream, block_ids=np.asarray(self.block_ids), phi=self.phi, logits=self.logits,
                    head_weight=self.head_weight, model_sig=np.asarray(model_sig),
                )

    @classmethod
    def load(cls, path: Path, model_sig: str, block_ids) -> BlockFeatures | None:
        """The saved features, or None if absent, unreadable or from another model or grid."""
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["model_sig"]) != model_sig:
                    return None
                ids = [str(i) for i in data["block_ids"]]
                phi, logits, head_weight = data["phi"], data["logits"], data["head_weight"]
        except (OSError, KeyError, ValueError):
            return None
        if (phi.ndim != 2 or logits.ndim != 2 or len(ids) != len(phi) or len(ids) != len(logits)
                or not set(ids) <= set(block_ids)):
            return None
        return cls(ids, phi.astype(np.float32), logits.astype(np.float32), head_weight.astype(np.float32))


def features_path(labels_dir: Path, obs_id: str) -> Path:
    return Path(labels_dir) / f"{obs_id}.block_features.npz"


def block_laplace_scores(
    features: BlockFeatures,
    laplace_path: Path,
    *,
    ai4exomars_path: str | None = None,
    samples: int = LAPLACE_SAMPLES,
    seed: int = 0,
    progress_cb: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, float]:
    """block_id -> Laplace predictive entropy / log K. Empty if cancelled."""
    import torch

    from marslabeler.inference.modelio import _ensure_vision_backend_importable

    _ensure_vision_backend_importable(ai4exomars_path)
    from vision_backend.uncertainty.block_laplace import load_last_layer_laplace

    laplace = load_last_layer_laplace(laplace_path)
    weight = torch.from_numpy(features.head_weight)
    if tuple(weight.shape) != tuple(laplace.weight.shape) or float((weight - laplace.weight).abs().max()) > 1e-5:
        raise ValueError(
            f"{Path(laplace_path).name} was fitted for a different block head than this checkpoint's."
        )
    phi, logits = torch.from_numpy(features.phi), torch.from_numpy(features.logits)
    generator = torch.Generator().manual_seed(seed)
    entropy = []
    total = len(features)
    for start in range(0, total, _CHUNK):
        if should_cancel is not None and should_cancel():
            return {}
        stop = min(start + _CHUNK, total)
        out = laplace.predictive(logits[start:stop], phi[start:stop], method="mc",
                                 n_samples=samples, generator=generator)
        entropy.append(out["entropy"])
        if progress_cb is not None:
            progress_cb(stop, total)
    if not entropy:
        return {}
    normalised = (torch.cat(entropy) / math.log(laplace.num_classes)).clamp(0, 1).numpy()
    return {block_id: float(score) for block_id, score in zip(features.block_ids, normalised)}


def block_mahalanobis_scores(
    features: BlockFeatures, stats_path: Path, *, ai4exomars_path: str | None = None
) -> dict[str, float]:
    """block_id -> Mahalanobis distance of phi / the file's reference distance."""
    import torch

    from marslabeler.inference.modelio import load_uncertainty_stats

    stats = load_uncertainty_stats(stats_path, ai4exomars_path=ai4exomars_path)
    if features.phi.ndim != 2 or features.phi.shape[1] != stats.feature_dim:
        raise ValueError(
            f"{Path(stats_path).name} was fitted on {stats.feature_dim}-dim features, but this "
            f"model's block feature phi is {features.phi.shape[-1]}-dim. Use the block-level "
            "Mahalanobis file fitted for this checkpoint."
        )
    from vision_backend.uncertainty.malahanobis import mahalanobis_distance

    distance = mahalanobis_distance(torch.from_numpy(features.phi), stats)
    distance = (distance / max(float(stats.reference_max_distance or 1.0), 1e-8)).numpy()
    return {block_id: float(score) for block_id, score in zip(features.block_ids, distance)}
