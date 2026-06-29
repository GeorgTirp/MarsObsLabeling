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


def test_block_size_must_be_multiple_of_32():
    """Test that block_size validation fails for non-multiples of 32."""
    with pytest.raises(ValueError, match="multiple of 32"):
        Grid(4096, 4096, 4096, 511, "OBS", Affine.identity())


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
