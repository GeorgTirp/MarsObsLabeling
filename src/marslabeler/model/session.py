"""Session: ties together RasterSource, Grid, and LabelStore with navigation."""

import json
import time
from hashlib import md5
from pathlib import Path
from typing import Literal, Optional

from marslabeler.io.raster import RasterSource
from marslabeler.model.grid import BlockInfo, Grid
from marslabeler.model.labelstore import LabelStore


class Session:
    """Manages labeling session: navigation, auto-advance, and state persistence."""

    def __init__(
        self,
        raster_source: RasterSource,
        grid: Grid,
        label_store: LabelStore,
        config: dict,
    ):
        self.raster = raster_source
        self.grid = grid
        self.labels = label_store
        self.config = config

        # Navigation state
        self.current_block_idx = 0
        self.last_label_time = 0
        self.label_count_since_autosave = 0

    def current_block(self) -> BlockInfo:
        """Get the current block info."""
        return self.grid.get_block(self.current_block_idx)

    def current_panel_idx(self) -> int:
        """Get the current panel index."""
        return self.current_block().panel_idx

    def move_to_block(self, block_idx: int) -> None:
        """Jump to a specific block by index."""
        if block_idx < 0 or block_idx >= self.grid.num_blocks():
            return
        self.current_block_idx = block_idx

    def move_to_panel(self, panel_idx: int) -> None:
        """Jump to first block of a panel."""
        if panel_idx < 0 or panel_idx >= self.grid.num_panels:
            return
        self.current_block_idx = panel_idx * self.grid.blocks_per_panel

    def label_current_block(self, class_id: int, class_name: str) -> None:
        """Label the current block and auto-advance."""
        block = self.current_block()
        self.labels.assign(block.block_id, class_id, class_name)
        self.label_count_since_autosave += 1
        self.last_label_time = time.time()
        self._auto_advance()

    def abstain_current_block(self) -> None:
        """Mark current block as abstain and auto-advance."""
        block = self.current_block()
        self.labels.assign(block.block_id, -1, "Abstain")
        self.label_count_since_autosave += 1
        self.last_label_time = time.time()
        self._auto_advance()

    def clear_current_block(self) -> None:
        """Clear current block back to unlabeled."""
        block = self.current_block()
        self.labels.clear(block.block_id)
        self.label_count_since_autosave += 1

    def relabel_current_block(self, class_id: int, class_name: str) -> None:
        """Edit current block's label (no auto-advance on edit)."""
        block = self.current_block()
        self.labels.assign(block.block_id, class_id, class_name)
        self.label_count_since_autosave += 1

    def _auto_advance(self) -> None:
        """Auto-advance to next target block based on config."""
        advance_mode = self.config.get("navigation", {}).get("advance_mode", "next_unlabeled")

        if advance_mode == "next_unlabeled":
            self._advance_to_next_unlabeled()
        elif advance_mode == "next_sequential":
            self._advance_to_next_sequential()

    def _advance_to_next_unlabeled(self) -> None:
        """Advance to the next unlabeled block (skipping nodata/skip candidates)."""
        start_idx = self.current_block_idx + 1
        current_panel = self.current_block().panel_idx

        # Search in current panel
        for idx in range(start_idx, (current_panel + 1) * self.grid.blocks_per_panel):
            if idx >= self.grid.num_blocks():
                break
            block = self.grid.get_block(idx)
            record = self.labels.get_record(block.block_id)
            if record.status == "unlabeled" and not self._should_skip_block(block):
                self.current_block_idx = idx
                return

        # Move to next incomplete panel and search
        for panel_idx in range(current_panel + 1, self.grid.num_panels):
            if self._is_panel_complete(panel_idx):
                continue
            first_block_idx = panel_idx * self.grid.blocks_per_panel
            self.current_block_idx = first_block_idx
            return

        # If we get here, wrap to first unlabeled in first incomplete panel
        for panel_idx in range(self.grid.num_panels):
            if self._is_panel_complete(panel_idx):
                continue
            first_block_idx = panel_idx * self.grid.blocks_per_panel
            self.current_block_idx = first_block_idx
            return

    def _advance_to_next_sequential(self) -> None:
        """Advance to the next block in order."""
        idx = (self.current_block_idx + 1) % self.grid.num_blocks()
        self.current_block_idx = idx

    def _should_skip_block(self, block: BlockInfo) -> bool:
        """Check if a block should be auto-skipped based on nodata/variance."""
        skip_config = self.config.get("skip", {})
        nodata_threshold = skip_config.get("nodata_skip_threshold", 0.5)
        variance_threshold = skip_config.get("variance_skip_threshold", 0.0)
        skip_low_var = skip_config.get("skip_low_variance", False)

        # Check nodata
        nodata_frac = self.raster.nodata_fraction(block.x_px, block.y_px, block.w_px, block.h_px)
        if nodata_frac > nodata_threshold:
            # Mark as nodata if it wasn't already
            record = self.labels.get_record(block.block_id)
            if record.status == "unlabeled":
                self.labels.set_nodata(block.block_id)
            return True

        # Check variance
        if skip_low_var and variance_threshold > 0:
            var = self.raster.variance(block.x_px, block.y_px, block.w_px, block.h_px)
            if var < variance_threshold:
                return True

        return False

    def _is_panel_complete(self, panel_idx: int) -> bool:
        """Check if a panel has any remaining unlabeled/unskipped blocks."""
        blocks = self.grid.get_panel_blocks(panel_idx)
        for block in blocks:
            record = self.labels.get_record(block.block_id)
            if record.status == "unlabeled" and not self._should_skip_block(block):
                return False
        return True

    def move_to_previous_block(self) -> None:
        """Move to previous block without labeling."""
        if self.current_block_idx > 0:
            self.current_block_idx -= 1

    def move_to_next_block(self) -> None:
        """Move to next block without labeling."""
        if self.current_block_idx < self.grid.num_blocks() - 1:
            self.current_block_idx += 1

    def move_to_first_block_in_panel(self) -> None:
        """Jump to first block of current panel."""
        panel_idx = self.current_block().panel_idx
        self.current_block_idx = panel_idx * self.grid.blocks_per_panel

    def move_to_next_panel(self) -> None:
        """Jump to first block of next panel."""
        current_panel = self.current_block().panel_idx
        next_panel = current_panel + 1
        if next_panel < self.grid.num_panels:
            self.move_to_panel(next_panel)

    def move_to_previous_panel(self) -> None:
        """Jump to first block of previous panel."""
        current_panel = self.current_block().panel_idx
        if current_panel > 0:
            self.move_to_panel(current_panel - 1)

    def should_autosave(self) -> bool:
        """Check if autosave should trigger."""
        save_config = self.config.get("autosave", {})
        every_n = save_config.get("every_n_labels", 25)
        every_secs = save_config.get("every_seconds", 30)

        labels_threshold = self.label_count_since_autosave >= every_n
        time_threshold = (time.time() - self.last_label_time) > every_secs
        return labels_threshold or time_threshold

    def reset_autosave_counter(self) -> None:
        """Reset autosave counters after saving."""
        self.label_count_since_autosave = 0
        self.last_label_time = time.time()

    def save_session(self, labels_dir: Path) -> None:
        """Save labels and session state."""
        labels_dir = Path(labels_dir)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Save labels to Parquet
        parquet_path = labels_dir / f"{self.grid.obs_id}.parquet"
        self.labels.save_parquet(parquet_path)

        # Save session JSON
        session_data = {
            "obs_id": self.grid.obs_id,
            "current_block_idx": self.current_block_idx,
            "panel_size": self.grid.panel_size,
            "block_size": self.grid.block_size,
            "timestamp": int(time.time() * 1000),
        }
        session_path = labels_dir / f"{self.grid.obs_id}.session.json"
        with open(session_path, "w") as f:
            json.dump(session_data, f, indent=2)

    @classmethod
    def load_or_create(
        cls,
        raster_path: Path,
        grid: Grid,
        config: dict,
        labels_dir: Path,
        labeler: str = "unknown",
    ) -> "Session":
        """Load session if it exists, otherwise create new."""
        labels_dir = Path(labels_dir)
        parquet_path = labels_dir / f"{grid.obs_id}.parquet"
        session_path = labels_dir / f"{grid.obs_id}.session.json"

        raster = RasterSource(raster_path)
        raster.open()

        if parquet_path.exists() and session_path.exists():
            # Load existing session
            label_store = LabelStore.load_parquet(parquet_path, grid, labeler)
            session = cls(raster, grid, label_store, config)

            # Restore cursor position
            with open(session_path) as f:
                session_data = json.load(f)
            session.current_block_idx = session_data.get("current_block_idx", 0)

            return session
        else:
            # Create new session
            label_store = LabelStore(grid, labeler)
            session = cls(raster, grid, label_store, config)
            return session
