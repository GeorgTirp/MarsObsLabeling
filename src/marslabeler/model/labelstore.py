"""Label store: in-memory state + Parquet persistence with undo/redo."""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import Array, ChunkedArray, compute as pc

from marslabeler.model.grid import BlockInfo, Grid


class LabelRecord:
    """A single block's label record."""

    def __init__(
        self,
        block_id: str,
        obs_id: str,
        panel_row: int,
        panel_col: int,
        block_row: int,
        block_col: int,
        x_px: int,
        y_px: int,
        w_px: int,
        h_px: int,
        class_id: int = -3,  # -3 = unlabeled sentinel
        class_name: str = "unlabeled",
        status: str = "unlabeled",
        map_x: Optional[float] = None,
        map_y: Optional[float] = None,
        gsd: float = 1.0,
        labeler: str = "unknown",
    ):
        self.block_id = block_id
        self.obs_id = obs_id
        self.panel_row = panel_row
        self.panel_col = panel_col
        self.block_row = block_row
        self.block_col = block_col
        self.x_px = x_px
        self.y_px = y_px
        self.w_px = w_px
        self.h_px = h_px
        self.class_id = class_id
        self.class_name = class_name
        self.status = status
        self.map_x = map_x
        self.map_y = map_y
        self.gsd = gsd
        self.labeler = labeler
        self.created_utc = int(time.time() * 1000)
        self.updated_utc = self.created_utc
        self.edit_count = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for Parquet."""
        return {
            "block_id": self.block_id,
            "obs_id": self.obs_id,
            "panel_row": self.panel_row,
            "panel_col": self.panel_col,
            "block_row": self.block_row,
            "block_col": self.block_col,
            "x_px": self.x_px,
            "y_px": self.y_px,
            "w_px": self.w_px,
            "h_px": self.h_px,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "status": self.status,
            "map_x": self.map_x,
            "map_y": self.map_y,
            "gsd": self.gsd,
            "labeler": self.labeler,
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
            "edit_count": self.edit_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LabelRecord":
        """Create from dictionary (e.g., from Parquet)."""
        record = cls(
            block_id=data["block_id"],
            obs_id=data["obs_id"],
            panel_row=data["panel_row"],
            panel_col=data["panel_col"],
            block_row=data["block_row"],
            block_col=data["block_col"],
            x_px=data["x_px"],
            y_px=data["y_px"],
            w_px=data["w_px"],
            h_px=data["h_px"],
            class_id=data.get("class_id", -3),
            class_name=data.get("class_name", "unlabeled"),
            status=data.get("status", "unlabeled"),
            map_x=data.get("map_x"),
            map_y=data.get("map_y"),
            gsd=data.get("gsd", 1.0),
            labeler=data.get("labeler", "unknown"),
        )
        record.created_utc = data.get("created_utc", int(time.time() * 1000))
        record.updated_utc = data.get("updated_utc", record.created_utc)
        record.edit_count = data.get("edit_count", 0)
        return record


class LabelStore:
    """In-memory label store with Parquet persistence and undo/redo."""

    def __init__(self, grid: Grid, labeler: str = "unknown"):
        self.grid = grid
        self.labeler = labeler
        self.records: dict[str, LabelRecord] = {}
        self.undo_stack: list[dict[str, LabelRecord]] = []
        self.redo_stack: list[dict[str, LabelRecord]] = []

        # Initialize all blocks as unlabeled
        for block in grid.iter_blocks():
            record = LabelRecord(
                block_id=block.block_id,
                obs_id=grid.obs_id,
                panel_row=block.panel_row,
                panel_col=block.panel_col,
                block_row=block.block_row,
                block_col=block.block_col,
                x_px=block.x_px,
                y_px=block.y_px,
                w_px=block.w_px,
                h_px=block.h_px,
                labeler=labeler,
            )
            map_x, map_y = grid.block_to_map(block)
            record.map_x = map_x
            record.map_y = map_y
            record.gsd = abs(grid.transform.a)
            self.records[block.block_id] = record

    def assign(self, block_id: str, class_id: int, class_name: str) -> None:
        """Assign a class to a block."""
        if block_id not in self.records:
            raise ValueError(f"Unknown block: {block_id}")

        # Save current state to undo stack
        self._save_undo_state()

        record = self.records[block_id]
        record.class_id = class_id
        record.class_name = class_name
        record.status = "labeled" if class_id >= 0 else "abstain"
        record.updated_utc = int(time.time() * 1000)
        record.edit_count += 1

        # Clear redo stack on new action
        self.redo_stack.clear()

    def bulk_assign(self, block_ids: list[str], class_id: int, class_name: str) -> int:
        """
        Assign the same class to many blocks with a single undo snapshot.

        Used for "fill remaining unlabeled blocks as NA" on panel completion.
        Returns the number of blocks actually changed.
        """
        targets = [bid for bid in block_ids if bid in self.records]
        if not targets:
            return 0

        self._save_undo_state()
        now = int(time.time() * 1000)
        status = "labeled" if class_id >= 0 else "abstain"
        for block_id in targets:
            record = self.records[block_id]
            record.class_id = class_id
            record.class_name = class_name
            record.status = status
            record.updated_utc = now
            record.edit_count += 1
        self.redo_stack.clear()
        return len(targets)

    def set_nodata(self, block_id: str) -> None:
        """Mark a block as nodata."""
        if block_id not in self.records:
            raise ValueError(f"Unknown block: {block_id}")

        self._save_undo_state()
        record = self.records[block_id]
        record.class_id = -2
        record.class_name = "No data"
        record.status = "nodata"
        record.updated_utc = int(time.time() * 1000)
        record.edit_count += 1
        self.redo_stack.clear()

    def clear(self, block_id: str) -> None:
        """Clear a block back to unlabeled."""
        if block_id not in self.records:
            raise ValueError(f"Unknown block: {block_id}")

        self._save_undo_state()
        record = self.records[block_id]
        record.class_id = -3
        record.class_name = "unlabeled"
        record.status = "unlabeled"
        record.updated_utc = int(time.time() * 1000)
        record.edit_count += 1
        self.redo_stack.clear()

    def undo(self) -> None:
        """Undo the last label action."""
        if not self.undo_stack:
            return

        current_state = {block_id: LabelRecord.from_dict(rec.to_dict())
                        for block_id, rec in self.records.items()}
        self.redo_stack.append(current_state)

        previous_state = self.undo_stack.pop()
        self.records = {block_id: LabelRecord.from_dict(rec.to_dict())
                       for block_id, rec in previous_state.items()}

    def redo(self) -> None:
        """Redo the last undone action."""
        if not self.redo_stack:
            return

        current_state = {block_id: LabelRecord.from_dict(rec.to_dict())
                        for block_id, rec in self.records.items()}
        self.undo_stack.append(current_state)

        next_state = self.redo_stack.pop()
        self.records = {block_id: LabelRecord.from_dict(rec.to_dict())
                       for block_id, rec in next_state.items()}

    def _save_undo_state(self) -> None:
        """Save current state to undo stack."""
        current_state = {block_id: LabelRecord.from_dict(rec.to_dict())
                        for block_id, rec in self.records.items()}
        self.undo_stack.append(current_state)

    def get_record(self, block_id: str) -> LabelRecord:
        """Get a label record by block ID."""
        if block_id not in self.records:
            raise ValueError(f"Unknown block: {block_id}")
        return self.records[block_id]

    def to_parquet_table(self) -> pa.Table:
        """Convert all records to a PyArrow Table."""
        # Extract columns from records
        block_ids = []
        obs_ids = []
        panel_rows = []
        panel_cols = []
        block_rows = []
        block_cols = []
        x_pxs = []
        y_pxs = []
        w_pxs = []
        h_pxs = []
        class_ids = []
        class_names = []
        statuses = []
        map_xs = []
        map_ys = []
        gsds = []
        labelers = []
        created_utcs = []
        updated_utcs = []
        edit_counts = []

        for rec in self.records.values():
            block_ids.append(rec.block_id)
            obs_ids.append(rec.obs_id)
            panel_rows.append(rec.panel_row)
            panel_cols.append(rec.panel_col)
            block_rows.append(rec.block_row)
            block_cols.append(rec.block_col)
            x_pxs.append(rec.x_px)
            y_pxs.append(rec.y_px)
            w_pxs.append(rec.w_px)
            h_pxs.append(rec.h_px)
            class_ids.append(rec.class_id)
            class_names.append(rec.class_name)
            statuses.append(rec.status)
            map_xs.append(rec.map_x)
            map_ys.append(rec.map_y)
            gsds.append(rec.gsd)
            labelers.append(rec.labeler)
            created_utcs.append(rec.created_utc)
            updated_utcs.append(rec.updated_utc)
            edit_counts.append(rec.edit_count)

        schema = pa.schema([
            ("block_id", pa.string()),
            ("obs_id", pa.string()),
            ("panel_row", pa.int32()),
            ("panel_col", pa.int32()),
            ("block_row", pa.int32()),
            ("block_col", pa.int32()),
            ("x_px", pa.int32()),
            ("y_px", pa.int32()),
            ("w_px", pa.int32()),
            ("h_px", pa.int32()),
            ("class_id", pa.int16()),
            ("class_name", pa.string()),
            ("status", pa.string()),
            ("map_x", pa.float64()),
            ("map_y", pa.float64()),
            ("gsd", pa.float32()),
            ("labeler", pa.string()),
            ("created_utc", pa.int64()),
            ("updated_utc", pa.int64()),
            ("edit_count", pa.int16()),
        ])

        arrays = [
            pa.array(block_ids, type=pa.string()),
            pa.array(obs_ids, type=pa.string()),
            pa.array(panel_rows, type=pa.int32()),
            pa.array(panel_cols, type=pa.int32()),
            pa.array(block_rows, type=pa.int32()),
            pa.array(block_cols, type=pa.int32()),
            pa.array(x_pxs, type=pa.int32()),
            pa.array(y_pxs, type=pa.int32()),
            pa.array(w_pxs, type=pa.int32()),
            pa.array(h_pxs, type=pa.int32()),
            pa.array(class_ids, type=pa.int16()),
            pa.array(class_names, type=pa.string()),
            pa.array(statuses, type=pa.string()),
            pa.array(map_xs, type=pa.float64()),
            pa.array(map_ys, type=pa.float64()),
            pa.array(gsds, type=pa.float32()),
            pa.array(labelers, type=pa.string()),
            pa.array(created_utcs, type=pa.int64()),
            pa.array(updated_utcs, type=pa.int64()),
            pa.array(edit_counts, type=pa.int16()),
        ]

        return pa.table(arrays, schema=schema)

    def save_parquet(self, path: Path) -> None:
        """Save labels to Parquet file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = self.to_parquet_table()
        pq.write_table(table, str(path))

    @classmethod
    def load_parquet(cls, path: Path, grid: Grid, labeler: str = "unknown") -> "LabelStore":
        """Load labels from Parquet file."""
        store = cls(grid, labeler)
        table = pq.read_table(str(path))

        for i in range(len(table)):
            row = table.slice(i, 1).to_pydict()
            block_id = row["block_id"][0]
            record = LabelRecord.from_dict({k: v[0] for k, v in row.items()})
            store.records[block_id] = record

        return store

    def count_labeled(self) -> int:
        """Count labeled blocks (not unlabeled, abstain, or nodata)."""
        return sum(1 for r in self.records.values() if r.status == "labeled")

    def count_abstained(self) -> int:
        """Count abstained blocks."""
        return sum(1 for r in self.records.values() if r.status == "abstain")

    def count_nodata(self) -> int:
        """Count nodata blocks."""
        return sum(1 for r in self.records.values() if r.status == "nodata")

    def count_unlabeled(self) -> int:
        """Count unlabeled blocks."""
        return sum(1 for r in self.records.values() if r.status == "unlabeled")

    def class_counts(self) -> dict[int, int]:
        """Count blocks per class."""
        counts: dict[int, int] = {}
        for r in self.records.values():
            if r.status == "labeled":
                counts[r.class_id] = counts.get(r.class_id, 0) + 1
        return counts
