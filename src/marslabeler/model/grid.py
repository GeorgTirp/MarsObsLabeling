"""Panel/block geometry, indexing, and coordinate transformations."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from rasterio.transform import Affine


@dataclass
class BlockInfo:
    """Information about a single block."""

    block_id: str
    panel_idx: int
    panel_row: int
    panel_col: int
    block_row: int
    block_col: int
    x_px: int  # top-left in full-image pixels
    y_px: int  # top-left in full-image pixels
    w_px: int  # width (may be < block_size at edges)
    h_px: int  # height (may be < block_size at edges)

    def centroid_px(self) -> tuple[float, float]:
        """Centroid in full-image pixel coords."""
        return (self.x_px + self.w_px / 2, self.y_px + self.h_px / 2)


class Grid:
    """Manages panel and block geometry for an observation."""

    def __init__(
        self,
        img_width: int,
        img_height: int,
        panel_size: int,
        block_size: int,
        obs_id: str,
        transform: Affine,
    ):
        self.img_width = img_width
        self.img_height = img_height
        self.panel_size = panel_size
        self.block_size = block_size
        self.obs_id = obs_id
        self.transform = transform

        # Validate constraints
        if block_size % 32 != 0:
            raise ValueError(f"block_size must be multiple of 32, got {block_size}")
        if panel_size % block_size != 0:
            raise ValueError(
                f"block_size ({block_size}) must divide panel_size ({panel_size})"
            )

        # Compute grid dimensions
        self.blocks_per_panel_row = panel_size // block_size
        self.blocks_per_panel_col = panel_size // block_size
        self.blocks_per_panel = self.blocks_per_panel_row * self.blocks_per_panel_col

        self.panels_across = (img_width + panel_size - 1) // panel_size
        self.panels_down = (img_height + panel_size - 1) // panel_size
        self.num_panels = self.panels_across * self.panels_down

        # Build full block index
        self._blocks = self._build_block_index()

    def _build_block_index(self) -> list[BlockInfo]:
        """Build complete block index for the observation."""
        blocks = []
        block_idx = 0

        for panel_idx in range(self.num_panels):
            panel_row, panel_col = divmod(panel_idx, self.panels_across)
            panel_x_start = panel_col * self.panel_size
            panel_y_start = panel_row * self.panel_size

            for local_block_row in range(self.blocks_per_panel_row):
                for local_block_col in range(self.blocks_per_panel_col):
                    block_x = panel_x_start + local_block_col * self.block_size
                    block_y = panel_y_start + local_block_row * self.block_size

                    # Handle partial blocks at edges
                    block_w = min(self.block_size, self.img_width - block_x)
                    block_h = min(self.block_size, self.img_height - block_y)

                    block_id = f"{self.obs_id}_{block_x}_{block_y}"
                    info = BlockInfo(
                        block_id=block_id,
                        panel_idx=panel_idx,
                        panel_row=panel_row,
                        panel_col=panel_col,
                        block_row=local_block_row,
                        block_col=local_block_col,
                        x_px=block_x,
                        y_px=block_y,
                        w_px=block_w,
                        h_px=block_h,
                    )
                    blocks.append(info)
                    block_idx += 1

        return blocks

    def get_block(self, idx: int) -> BlockInfo:
        """Get block info by index."""
        if idx < 0 or idx >= len(self._blocks):
            raise IndexError(f"Block index {idx} out of range [0, {len(self._blocks)})")
        return self._blocks[idx]

    def get_block_by_panel_and_local(self, panel_idx: int, local_block_idx: int) -> BlockInfo:
        """Get block by panel index and local block index within panel."""
        block_idx = panel_idx * self.blocks_per_panel + local_block_idx
        return self.get_block(block_idx)

    def get_panel_blocks(self, panel_idx: int) -> list[BlockInfo]:
        """Get all blocks in a panel."""
        if panel_idx < 0 or panel_idx >= self.num_panels:
            raise IndexError(f"Panel index {panel_idx} out of range [0, {self.num_panels})")
        start = panel_idx * self.blocks_per_panel
        end = start + self.blocks_per_panel
        return self._blocks[start:end]

    def get_panel_coords(self, panel_idx: int) -> tuple[int, int, int, int]:
        """Get panel bounds (x, y, width, height) in pixel coords."""
        panel_row, panel_col = divmod(panel_idx, self.panels_across)
        x = panel_col * self.panel_size
        y = panel_row * self.panel_size
        w = min(self.panel_size, self.img_width - x)
        h = min(self.panel_size, self.img_height - y)
        return (x, y, w, h)

    def block_to_map(self, block: BlockInfo) -> tuple[float, float]:
        """Transform block centroid to map (CRS) coords."""
        centroid_px = block.centroid_px()
        # Apply affine transform: map = transform * pixel
        x_map = self.transform.c + centroid_px[0] * self.transform.a
        y_map = self.transform.f + centroid_px[1] * self.transform.e
        return (x_map, y_map)

    def num_blocks(self) -> int:
        """Total number of blocks."""
        return len(self._blocks)

    def iter_blocks(self):
        """Iterate over all blocks."""
        return iter(self._blocks)
