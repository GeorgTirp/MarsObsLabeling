"""Load a Neural-PCA explainability gallery (see AI4ExoMars vision_backend.pc_align).

The gallery is a plain `{class_id: {component_idx: [GalleryThumbnail, ...]}}` dict of
dataclasses holding only numpy arrays / floats / ints / strings (no torch tensors), so
it's returned as-is -- no MarsObsLabeling-side wrapper needed. Building one requires a
trained checkpoint and a full corpus pass (`AI4ExoMars/vision_backend/pc_align/fit_neural_pca.py`,
run offline); until that artifact exists, `load_npca_gallery` simply raises
`FileNotFoundError`, which callers treat as "not available yet", not an error.
"""

from __future__ import annotations

from pathlib import Path


def load_npca_gallery(path: str | Path, ai4exomars_path: str | None = None):
    """Load a `<checkpoint_stem>.npca.pt` gallery artifact.

    Raises FileNotFoundError if it doesn't exist yet.
    """
    from marslabeler.inference.modelio import _ensure_vision_backend_importable

    _ensure_vision_backend_importable(ai4exomars_path)
    from vision_backend.pc_align.neural_pca import load_gallery

    return load_gallery(path)
