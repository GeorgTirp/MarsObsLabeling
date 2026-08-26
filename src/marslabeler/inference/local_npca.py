"""Rank the blocks of the *currently loaded observation* on the model's Neural-PCA
components.

The offline gallery baked into `<checkpoint>.npca.pt` answers "what do this
class's components look like in the corpus the model was trained on?" -- useful
for understanding the model, but its thumbnails carry pixel coordinates in the
*training* raster, so clicking one cannot navigate the observation an annotator
actually has open (a different file, a different coordinate space, and usually
not on disk at all).

This module answers the sibling question -- "which blocks of THIS observation
most express component k of class c?" -- while reusing the training-fitted PCA
basis, so the components still mean what they meant during fitting. Only the
exemplars become local. Refitting PCA per observation was rejected: components
would then differ image to image, and rare classes rarely have enough blocks.

Input is the per-block pooled feature phi(x) captured during the prediction pass
(`modelio.make_predict_fn_with_embeddings`), so building a local gallery needs no
extra pass over the imagery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LocalGalleryItem:
    """One top-activating block of the loaded observation, for a (class, component)."""

    rank: int
    score: float
    block_id: str
    block_row: int
    block_col: int
    panel_idx: int
    x_px: int  # top-left in this observation's own full-image pixel coords
    y_px: int
    w_px: int  # may be < block_size at the image edge
    h_px: int


# class_id -> component_idx -> ranked items (best first)
LocalNPCAGallery = dict[int, dict[int, list[LocalGalleryItem]]]


def stack_block_embeddings(
    embeddings: Sequence[np.ndarray], blocks: Sequence[Any]
) -> tuple[np.ndarray, list[Any]]:
    """Concatenate per-batch embedding arrays and row-align them with `blocks`.

    The engine feeds batches in `blocks` order, so concatenation restores that
    order. A short tail is tolerated (a cancelled run stops mid-sequence); a
    longer-than-expected stack is a real mismatch and raises.
    """
    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32), []
    stacked = np.concatenate([np.asarray(e, dtype=np.float32) for e in embeddings], axis=0)
    if stacked.shape[0] > len(blocks):
        raise ValueError(
            f"captured {stacked.shape[0]} block embeddings for {len(blocks)} blocks "
            f"-- the embedding sink and the block sequence are out of step"
        )
    return stacked, list(blocks[: stacked.shape[0]])


def build_local_gallery(
    embeddings: np.ndarray,
    blocks: Sequence[Any],
    bases: Mapping[int, Any],
    classifier_weights: Mapping[int, np.ndarray],
    *,
    top_k: int = 6,
    eligible_block_ids: Mapping[int, set[str]] | None = None,
) -> LocalNPCAGallery:
    """Rank `blocks` on each stored class basis.

    Parameters
    ----------
    embeddings : [N, F] pooled phi(x), row-aligned with `blocks`.
    bases : class_id -> NeuralPCAResult, from `load_gallery_bases`.
    classifier_weights : class_id -> [F] w_k, so psi_k = w_k * phi.
    eligible_block_ids : optional per-class restriction, e.g. only blocks the
        model actually predicted as that class -- keeps a component's exemplars
        on-topic instead of surfacing whichever block happens to project highest.
    """
    import torch

    from vision_backend.pc_align.neural_pca import rank_indices_by_component

    gallery: LocalNPCAGallery = {}
    if embeddings.size == 0 or not len(blocks):
        return gallery

    phi_all = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))

    for class_id, pca in bases.items():
        weight = classifier_weights.get(int(class_id))
        if weight is None:
            continue

        rows = list(range(len(blocks)))
        if eligible_block_ids is not None:
            allowed = eligible_block_ids.get(int(class_id))
            if allowed is not None:
                rows = [i for i in rows if getattr(blocks[i], "block_id", None) in allowed]
        if not rows:
            continue

        phi = phi_all[rows]
        if phi.shape[1] != len(weight):
            raise ValueError(
                f"class {class_id}: embeddings are {phi.shape[1]}-dim but the "
                f"classifier weight vector is {len(weight)}-dim"
            )
        psi = phi * torch.from_numpy(np.asarray(weight, dtype=np.float32))

        ranked = rank_indices_by_component(psi, pca, top_k=top_k)
        entry: dict[int, list[LocalGalleryItem]] = {}
        for component_idx, hits in ranked.items():
            items = []
            for rank, (local_row, score) in enumerate(hits, start=1):
                block = blocks[rows[local_row]]
                items.append(
                    LocalGalleryItem(
                        rank=rank,
                        score=score,
                        block_id=block.block_id,
                        block_row=block.block_row,
                        block_col=block.block_col,
                        panel_idx=block.panel_idx,
                        x_px=block.x_px,
                        y_px=block.y_px,
                        w_px=block.w_px,
                        h_px=block.h_px,
                    )
                )
            entry[component_idx] = items
        if entry:
            gallery[int(class_id)] = entry
    return gallery


def classifier_weight_vectors(model, class_ids: Sequence[int]) -> dict[int, np.ndarray]:
    """class_id -> w_k as numpy, read off `decoder.head`."""
    from vision_backend.model.features import get_classifier_weight_vector

    weights: dict[int, np.ndarray] = {}
    for class_id in class_ids:
        try:
            w = get_classifier_weight_vector(model, int(class_id))
        except ValueError:
            continue  # class id outside the head's range
        weights[int(class_id)] = w.detach().cpu().numpy().astype(np.float32)
    return weights
