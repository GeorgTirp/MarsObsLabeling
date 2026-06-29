"""Progress tracking: counts, percentages, per-class statistics."""

from dataclasses import dataclass

from marslabeler.model.labelstore import LabelStore
from marslabeler.model.grid import Grid


@dataclass
class ProgressStats:
    """Label progress statistics."""

    obs_id: str
    total_blocks: int
    labeled: int
    abstained: int
    nodata: int
    unlabeled: int
    percent_complete: float
    class_counts: dict[int, int]  # class_id -> count

    def get_class_percent(self, class_id: int) -> float:
        """Get percentage of blocks with this class."""
        if self.labeled == 0:
            return 0.0
        return (self.class_counts.get(class_id, 0) / self.total_blocks) * 100


def calculate_progress(grid: Grid, label_store: LabelStore) -> ProgressStats:
    """
    Calculate progress statistics for an observation.

    Args:
        grid: Grid with block geometry
        label_store: Label store with current state

    Returns:
        ProgressStats with counts and percentages
    """
    total = grid.num_blocks()
    labeled = label_store.count_labeled()
    abstained = label_store.count_abstained()
    nodata = label_store.count_nodata()
    unlabeled = label_store.count_unlabeled()

    complete = labeled + abstained + nodata
    percent_complete = (complete / total * 100) if total > 0 else 0.0

    class_counts = label_store.class_counts()

    return ProgressStats(
        obs_id=grid.obs_id,
        total_blocks=total,
        labeled=labeled,
        abstained=abstained,
        nodata=nodata,
        unlabeled=unlabeled,
        percent_complete=percent_complete,
        class_counts=class_counts,
    )


def format_progress_text(stats: ProgressStats) -> str:
    """Format progress stats as human-readable text."""
    lines = [
        f"Observation: {stats.obs_id}",
        f"Total blocks: {stats.total_blocks}",
        f"Labeled: {stats.labeled} ({stats.labeled/stats.total_blocks*100:.1f}%)",
        f"Abstained: {stats.abstained} ({stats.abstained/stats.total_blocks*100:.1f}%)",
        f"No data: {stats.nodata} ({stats.nodata/stats.total_blocks*100:.1f}%)",
        f"Unlabeled: {stats.unlabeled} ({stats.unlabeled/stats.total_blocks*100:.1f}%)",
        f"",
        f"Overall complete: {stats.percent_complete:.1f}%",
    ]

    if stats.class_counts:
        lines.append("")
        lines.append("Per-class counts:")
        for class_id in sorted(stats.class_counts.keys()):
            count = stats.class_counts[class_id]
            pct = count / stats.total_blocks * 100
            lines.append(f"  Class {class_id}: {count} ({pct:.1f}%)")

    return "\n".join(lines)
