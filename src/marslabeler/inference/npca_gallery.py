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


def parse_npca_source_id(source_id: str) -> tuple[str, int, int] | None:
    """Reverse `fit_neural_pca.py`'s own `source_id` convention --
    `f"{Path(imagery_path).stem}_{record.col}_{record.row}"` -- back into
    (imagery_stem, col, row), for "jump to this gallery thumbnail's location"
    navigation. `col`/`row` are native pixel coordinates in that imagery
    file's own raster (not whatever observation happens to be open in
    MarsObsLabeling right now -- see resolve_training_imagery_path).

    Splits from the right (`rsplit("_", 2)`) since the stem itself often
    contains underscores (e.g. "drg_on_label_grid"); returns None if the last
    two underscore-separated fields aren't both integers, or the stem is empty.
    """
    parts = source_id.rsplit("_", 2)
    if len(parts) != 3:
        return None
    stem, col_str, row_str = parts
    if not stem:
        return None
    try:
        return stem, int(col_str), int(row_str)
    except ValueError:
        return None
