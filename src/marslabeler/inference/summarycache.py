"""Small, JSON-compatible inference summaries stored with reviewed predictions."""

from dataclasses import asdict
from pathlib import Path
import math

from marslabeler.inference.local_npca import LocalGalleryItem
from marslabeler.inference.modelio import sidecar_path


def npca_signature(checkpoint: Path) -> str | None:
    path = sidecar_path(checkpoint, "npca.pt")
    try:
        stat = path.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"
    except FileNotFoundError:
        return None


def encode_summary(confidence, gallery, *, block_classifier, num_pixel_classes,
                   npca_error, checkpoint):
    return {
        "version": 1,
        "confidence": confidence,
        "gallery": {
            str(cid): {str(comp): [asdict(item) for item in items]
                       for comp, items in components.items()}
            for cid, components in gallery.items()
        },
        "block_classifier": block_classifier,
        "num_pixel_classes": num_pixel_classes,
        "npca_error": npca_error,
        "npca_signature": npca_signature(checkpoint),
    }


def decode_summary(value, *, checkpoint, grid):
    """Return a validated summary, or None to rebuild it without changing labels."""
    try:
        if value["version"] != 1 or value["npca_signature"] != npca_signature(checkpoint):
            return None
        blocks = {b.block_id: b for b in grid.iter_blocks()}
        confidence = {bid: float(score) for bid, score in value["confidence"].items()}
        if any(bid not in blocks or not math.isfinite(score) or not 0 <= score <= 1
               for bid, score in confidence.items()):
            return None
        gallery = {}
        for cid, components in value["gallery"].items():
            gallery[int(cid)] = {}
            for comp, raw_items in components.items():
                items = [LocalGalleryItem(**raw) for raw in raw_items]
                for item in items:
                    block = blocks.get(item.block_id)
                    if block is None or not math.isfinite(item.score):
                        return None
                    for name in ("block_row", "block_col", "panel_idx", "x_px", "y_px", "w_px", "h_px"):
                        if getattr(item, name) != getattr(block, name):
                            return None
                gallery[int(cid)][int(comp)] = items
        return dict(confidence=confidence, gallery=gallery,
                    block_classifier=bool(value["block_classifier"]),
                    num_pixel_classes=int(value["num_pixel_classes"]),
                    npca_error=str(value.get("npca_error") or ""))
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
