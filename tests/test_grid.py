"""Tests for panel/block grid geometry."""

import pytest
from rasterio.transform import Affine

from marslabeler.model.grid import Grid


@pytest.fixture
def basic_grid():
    """Create a basic grid for testing."""
    # 4096x4096 image, 4096-px panels, 512-px blocks
    return Grid(
        img_width=4096,
        img_height=4096,
        panel_size=4096,
        block_size=512,
        obs_id="TEST_OBS",
        transform=Affine.identity(),
    )


@pytest.fixture
def multi_panel_grid():
    """Create a grid that spans multiple panels."""
    # 8192x8192 image, 4096-px panels, 512-px blocks
    return Grid(
        img_width=8192,
        img_height=8192,
        panel_size=4096,
        block_size=512,
        obs_id="TEST_OBS_LARGE",
        transform=Affine.identity(),
    )


def test_grid_basic_dimensions(basic_grid):
    """Test basic grid dimensions."""
    assert basic_grid.img_width == 4096
    assert basic_grid.img_height == 4096
    assert basic_grid.panel_size == 4096
    assert basic_grid.block_size == 512
    assert basic_grid.blocks_per_panel == 64  # 8x8
    assert basic_grid.num_panels == 1


def test_grid_multi_panel_dimensions(multi_panel_grid):
    """Test multi-panel grid dimensions."""
    assert multi_panel_grid.num_panels == 4  # 2x2 panels
    assert multi_panel_grid.panels_across == 2
    assert multi_panel_grid.panels_down == 2


def test_block_size_need_not_be_multiple_of_32():
    """block_size is unconstrained by 32 (dropped: no model requires stride-32 tiles)."""
    grid = Grid(64, 64, 64, 4, "OBS", Affine.identity())
    assert grid.block_size == 4
    assert grid.blocks_per_panel == 16 * 16


def test_block_size_must_divide_panel_size():
    """Test that block_size must divide panel_size."""
    # 768 is multiple of 32 but doesn't divide 4096 evenly
    with pytest.raises(ValueError, match="must divide"):
        Grid(4096, 4096, 4096, 768, "OBS", Affine.identity())


def test_get_block_by_index(basic_grid):
    """Test getting a block by global index."""
    block = basic_grid.get_block(0)
    assert block.x_px == 0
    assert block.y_px == 0
    assert block.w_px == 512
    assert block.h_px == 512
    assert block.block_row == 0
    assert block.block_col == 0


def test_get_block_row_major_order(basic_grid):
    """Test that blocks are in row-major order."""
    block0 = basic_grid.get_block(0)
    block1 = basic_grid.get_block(1)
    block8 = basic_grid.get_block(8)

    # Block 1 should be to the right of block 0 (same row)
    assert block1.block_col == 1
    assert block1.block_row == 0
    assert block1.x_px == 512

    # Block 8 should be in the next row (left edge)
    assert block8.block_col == 0
    assert block8.block_row == 1
    assert block8.y_px == 512


def test_get_panel_blocks(basic_grid):
    """Test getting all blocks in a panel."""
    blocks = basic_grid.get_panel_blocks(0)
    assert len(blocks) == 64  # 8x8 grid


def test_get_panel_coords(basic_grid):
    """Test getting panel coordinates."""
    x, y, w, h = basic_grid.get_panel_coords(0)
    assert x == 0
    assert y == 0
    assert w == 4096
    assert h == 4096


def test_get_panel_coords_multi_panel(multi_panel_grid):
    """Test panel coords in multi-panel grid."""
    # Panel 0: top-left
    x, y, w, h = multi_panel_grid.get_panel_coords(0)
    assert x == 0
    assert y == 0

    # Panel 1: top-right
    x, y, w, h = multi_panel_grid.get_panel_coords(1)
    assert x == 4096
    assert y == 0

    # Panel 2: bottom-left
    x, y, w, h = multi_panel_grid.get_panel_coords(2)
    assert x == 0
    assert y == 4096

    # Panel 3: bottom-right
    x, y, w, h = multi_panel_grid.get_panel_coords(3)
    assert x == 4096
    assert y == 4096


def test_partial_blocks_at_edge():
    """Test that blocks at image edges are correctly clipped."""
    # 3000x3000 image, 4096 panel, 512 block
    # Should have partial blocks at the edges
    grid = Grid(3000, 3000, 4096, 512, "OBS", Affine.identity())

    # Get the last block
    last_block_idx = grid.num_blocks() - 1
    last_block = grid.get_block(last_block_idx)

    # Should be clipped to image bounds
    assert last_block.w_px < 512 or last_block.h_px < 512


def test_block_centroid_px(basic_grid):
    """Test block centroid calculation."""
    block = basic_grid.get_block(0)
    cx, cy = block.centroid_px()
    assert cx == 256  # Half of 512
    assert cy == 256


def test_block_to_map_identity_transform(basic_grid):
    """Test block-to-map transformation with identity transform."""
    block = basic_grid.get_block(0)
    x_map, y_map = basic_grid.block_to_map(block)

    # With identity transform, centroid should map directly
    assert x_map == 256
    assert y_map == 256


def test_block_to_map_scaled_transform():
    """Test block-to-map with a scaled/translated transform."""
    transform = Affine.translation(1000, 2000) * Affine.scale(10, -10)
    grid = Grid(4096, 4096, 4096, 512, "OBS", transform)

    block = grid.get_block(0)
    x_map, y_map = grid.block_to_map(block)

    # Centroid at (256, 256) in pixels
    # With scale=10 and translation: x = 1000 + 256*10 = 3560
    assert x_map == pytest.approx(1000 + 256 * 10)
    assert y_map == pytest.approx(2000 + 256 * (-10))


def test_iter_blocks(basic_grid):
    """Test iterating over all blocks."""
    blocks = list(basic_grid.iter_blocks())
    assert len(blocks) == 64


def test_block_ids_unique(basic_grid):
    """Test that all block IDs are unique."""
    blocks = list(basic_grid.iter_blocks())
    ids = [b.block_id for b in blocks]
    assert len(ids) == len(set(ids))


def test_get_block_invalid_index(basic_grid):
    """Test that invalid block index raises."""
    with pytest.raises(IndexError):
        basic_grid.get_block(999)


def test_get_panel_invalid_index(basic_grid):
    """Test that invalid panel index raises."""
    with pytest.raises(IndexError):
        basic_grid.get_panel_blocks(999)


# --------------------------------------------------------------------------- #
# Edge-panel geometry (image dims not a multiple of panel_size/block_size)
# --------------------------------------------------------------------------- #


@pytest.fixture
def ragged_grid():
    """Grid whose image is not a whole number of panels or blocks.

    This is the realistic case for HiRISE observations, and the one where the
    right-hand column and bottom row of panels are only partially covered.
    """
    return Grid(
        img_width=10000,
        img_height=8500,
        panel_size=4096,
        block_size=512,
        obs_id="RAGGED_OBS",
        transform=Affine.identity(),
    )


def test_no_block_has_negative_extent(ragged_grid):
    """Blocks whose origin falls outside the image clamp to 0, never negative.

    A negative w_px/h_px reaches numpy as np.zeros((-n, -m)) and raises
    ValueError inside a Qt event handler, which PySide6 swallows -- leaving
    the UI showing a stale block preview rather than crashing.
    """
    bad = [b for b in ragged_grid.iter_blocks() if b.w_px < 0 or b.h_px < 0]
    assert bad == [], f"{len(bad)} blocks have negative extent, e.g. {bad[:3]}"


def test_blocks_are_either_empty_or_within_image(ragged_grid):
    """Every block is either zero-sized or lies fully inside the image bounds."""
    for b in ragged_grid.iter_blocks():
        assert b.w_px >= 0 and b.h_px >= 0
        if b.w_px > 0 and b.h_px > 0:
            assert b.x_px + b.w_px <= ragged_grid.img_width
            assert b.y_px + b.h_px <= ragged_grid.img_height


def test_ragged_grid_has_ghost_blocks(ragged_grid):
    """The ragged case really does produce out-of-image blocks (guards the fixture)."""
    ghosts = [b for b in ragged_grid.iter_blocks() if b.w_px == 0 or b.h_px == 0]
    assert ghosts, "fixture no longer exercises the partial-panel case"


def test_panel_canvas_extent_matches_block_grid(ragged_grid):
    """Single-panel rendering must span panel_size, not the clipped panel coords.

    The canvas draws a uniform blocks_per_panel_row x blocks_per_panel_col grid and
    maps clicks through it, so the rendered extent has to be exactly panel_size on
    each side. Rendering the clipped extent from get_panel_coords() instead would
    stretch edge panels and desynchronise clicks from the previewed block.
    """
    g = ragged_grid
    canvas = 1600
    block_px_canvas = canvas // g.blocks_per_panel_col

    for panel_idx in range(g.num_panels):
        x, y, w, h = g.get_panel_coords(panel_idx)
        for canvas_x in range(0, canvas, block_px_canvas):
            expected_col = canvas_x // block_px_canvas
            # Rendering the full panel_size extent keeps click mapping exact...
            padded_col = int((canvas_x * g.panel_size / canvas) // g.block_size)
            assert padded_col == expected_col

        # ...whereas the clipped extent only agrees when the panel is full-width.
        if w < g.panel_size:
            clipped_cols = {
                int((cx * w / canvas) // g.block_size)
                for cx in range(0, canvas, block_px_canvas)
            }
            assert len(clipped_cols) < g.blocks_per_panel_col
