"""Tests for session management."""

import json
import pytest
from pathlib import Path
from rasterio.transform import Affine

from marslabeler.io.raster import RasterSource
from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import LabelStore
from marslabeler.model.session import Session


@pytest.fixture
def test_config():
    """Create a test config."""
    return {
        "navigation": {
            "advance_mode": "next_unlabeled",
            "advance_on_edit": False,
        },
        "skip": {
            "nodata_skip_threshold": 0.5,
            "variance_skip_threshold": 0.0,
            "skip_low_variance": False,
        },
        "autosave": {
            "every_n_labels": 5,
            "every_seconds": 60,
        },
    }


@pytest.fixture
def test_session(synthetic_geotiff, test_config):
    """Create a test session."""
    raster = RasterSource(synthetic_geotiff)
    raster.open()
    grid = Grid(4096, 4096, 4096, 512, "TEST_OBS", Affine.identity())
    labels = LabelStore(grid, "test_user")
    session = Session(raster, grid, labels, test_config)
    yield session
    raster.close()


def test_session_initialization(test_session):
    """Test session initializes correctly."""
    assert test_session.current_block_idx == 0
    assert test_session.current_block().block_id == test_session.grid.get_block(0).block_id


def test_current_block(test_session):
    """Test getting current block."""
    block = test_session.current_block()
    assert block.x_px == 0
    assert block.y_px == 0


def test_move_to_block(test_session):
    """Test moving to a specific block."""
    test_session.move_to_block(10)
    assert test_session.current_block_idx == 10
    assert test_session.current_block().block_row == 1
    assert test_session.current_block().block_col == 2


def test_move_to_panel(test_session):
    """Test moving to a panel."""
    test_session.move_to_panel(0)
    assert test_session.current_panel_idx() == 0
    assert test_session.current_block_idx == 0


def test_label_current_block(test_session):
    """Test labeling the current block."""
    block_id = test_session.current_block().block_id
    test_session.label_current_block(0, "Class A")

    record = test_session.labels.get_record(block_id)
    assert record.class_id == 0
    assert record.status == "labeled"


def test_label_auto_advances(test_session):
    """Test that labeling auto-advances."""
    initial_block_idx = test_session.current_block_idx
    test_session.label_current_block(0, "Class A")

    # Should have advanced to next unlabeled block
    assert test_session.current_block_idx > initial_block_idx


def test_abstain_auto_advances(test_session):
    """Test that abstaining auto-advances."""
    initial_block_idx = test_session.current_block_idx
    test_session.abstain_current_block()

    assert test_session.current_block_idx > initial_block_idx


def test_clear_block(test_session):
    """Test clearing a block."""
    block_id = test_session.current_block().block_id
    test_session.label_current_block(0, "Class A")

    # Move back to the block to clear it
    test_session.move_to_block(0)
    assert test_session.labels.get_record(block_id).status == "labeled"

    test_session.clear_current_block()
    assert test_session.labels.get_record(block_id).status == "unlabeled"


def test_relabel_no_advance(test_session):
    """Test relabeling doesn't auto-advance."""
    block_id = test_session.current_block().block_id
    test_session.label_current_block(0, "Class A")

    # Move back to the labeled block
    test_session.move_to_block(0)
    initial_idx = test_session.current_block_idx

    # Relabel (edit mode)
    test_session.relabel_current_block(1, "Class B")

    # Should NOT have advanced
    assert test_session.current_block_idx == initial_idx
    assert test_session.labels.get_record(block_id).class_id == 1


def test_move_backward(test_session):
    """Test moving backward without labeling."""
    test_session.move_to_block(10)
    test_session.move_to_previous_block()
    assert test_session.current_block_idx == 9


def test_move_forward(test_session):
    """Test moving forward without labeling."""
    test_session.move_to_block(10)
    test_session.move_to_next_block()
    assert test_session.current_block_idx == 11


def test_move_to_first_block_in_panel(test_session):
    """Test jumping to first block in panel."""
    test_session.move_to_block(20)
    test_session.move_to_first_block_in_panel()

    # First block of panel 0
    assert test_session.current_block_idx == 0


def test_move_to_next_panel(test_session):
    """Test moving to next panel."""
    test_session.move_to_panel(0)
    # Can't move to next panel if only one exists
    initial_idx = test_session.current_block_idx
    test_session.move_to_next_panel()
    # With 4096x4096 and 4096 panel size, only 1 panel exists
    assert test_session.current_block_idx == initial_idx


def test_autosave_by_label_count(test_session):
    """Test autosave trigger by label count."""
    # Config has every_n_labels: 5
    test_session.reset_autosave_counter()  # Reset to ensure clean state
    assert not test_session.should_autosave()

    for i in range(5):
        test_session.label_current_block(0, "Class A")

    assert test_session.should_autosave()


def test_reset_autosave_counter(test_session):
    """Test resetting autosave counter."""
    for i in range(5):
        test_session.label_current_block(0, "Class A")

    assert test_session.should_autosave()
    test_session.reset_autosave_counter()
    assert not test_session.should_autosave()


def test_save_and_load_session(test_session, tmp_path):
    """Test saving and loading a session."""
    # Add some labels
    block_ids = list(test_session.labels.records.keys())
    for i in range(3):
        test_session.move_to_block(i)
        test_session.label_current_block(0, "Class A")

    # Save
    test_session.save_session(tmp_path)
    parquet_path = tmp_path / "TEST_OBS.parquet"
    session_path = tmp_path / "TEST_OBS.session.json"

    assert parquet_path.exists()
    assert session_path.exists()

    # Load session data
    with open(session_path) as f:
        session_data = json.load(f)
    assert "current_block_idx" in session_data
    assert "obs_id" in session_data


def test_load_or_create_new(synthetic_geotiff, tmp_path, test_config):
    """Test creating a new session when no files exist."""
    grid = Grid(4096, 4096, 4096, 512, "TEST_OBS", Affine.identity())
    session = Session.load_or_create(
        synthetic_geotiff, grid, test_config, tmp_path, "test_user"
    )

    assert session.current_block_idx == 0
    assert session.labels.count_unlabeled() == grid.num_blocks()


def test_load_or_create_existing(synthetic_geotiff, tmp_path, test_config):
    """Test loading an existing session."""
    grid = Grid(4096, 4096, 4096, 512, "TEST_OBS", Affine.identity())

    # Create and save a session
    raster = RasterSource(synthetic_geotiff)
    raster.open()
    labels = LabelStore(grid, "test_user")
    session = Session(raster, grid, labels, test_config)

    # Add a label
    block_id = grid.get_block(5).block_id
    session.labels.assign(block_id, 0, "Class A")
    session.current_block_idx = 7

    # Save
    session.save_session(tmp_path)
    raster.close()

    # Load
    loaded_session = Session.load_or_create(
        synthetic_geotiff, grid, test_config, tmp_path, "test_user"
    )

    # Should have resumed at block 7
    assert loaded_session.current_block_idx == 7
    # Should have the label
    assert loaded_session.labels.count_labeled() == 1
