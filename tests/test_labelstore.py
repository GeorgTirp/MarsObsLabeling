"""Tests for label store."""

import pytest
from rasterio.transform import Affine

from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import LabelRecord, LabelStore


@pytest.fixture
def test_grid():
    """Create a test grid."""
    return Grid(4096, 4096, 4096, 512, "TEST_OBS", Affine.identity())


@pytest.fixture
def test_store(test_grid):
    """Create a test label store."""
    return LabelStore(test_grid, labeler="test_user")


def test_labelstore_initialization(test_store, test_grid):
    """Test that label store initializes with all blocks unlabeled."""
    assert len(test_store.records) == test_grid.num_blocks()
    assert test_store.count_unlabeled() == test_grid.num_blocks()
    assert test_store.count_labeled() == 0
    assert test_store.count_abstained() == 0
    assert test_store.count_nodata() == 0


def test_assign_label(test_store):
    """Test assigning a class to a block."""
    block_id = test_store.records[list(test_store.records.keys())[0]].block_id
    test_store.assign(block_id, 0, "Test Class")

    record = test_store.get_record(block_id)
    assert record.class_id == 0
    assert record.class_name == "Test Class"
    assert record.status == "labeled"
    assert record.edit_count == 1


def test_assign_abstain(test_store):
    """Test assigning abstain to a block."""
    block_id = list(test_store.records.keys())[0]
    test_store.assign(block_id, -1, "Abstain")

    record = test_store.get_record(block_id)
    assert record.status == "abstain"
    assert record.class_id == -1


def test_set_nodata(test_store):
    """Test marking a block as nodata."""
    block_id = list(test_store.records.keys())[0]
    test_store.set_nodata(block_id)

    record = test_store.get_record(block_id)
    assert record.status == "nodata"
    assert record.class_id == -2


def test_clear_block(test_store):
    """Test clearing a block back to unlabeled."""
    block_id = list(test_store.records.keys())[0]
    test_store.assign(block_id, 0, "Test Class")
    assert test_store.get_record(block_id).status == "labeled"

    test_store.clear(block_id)
    record = test_store.get_record(block_id)
    assert record.status == "unlabeled"
    assert record.class_id == -3


def test_edit_count_increments(test_store):
    """Test that edit_count increments on each action."""
    block_id = list(test_store.records.keys())[0]
    record = test_store.get_record(block_id)
    assert record.edit_count == 0

    test_store.assign(block_id, 0, "Class A")
    assert test_store.get_record(block_id).edit_count == 1

    test_store.assign(block_id, 1, "Class B")
    assert test_store.get_record(block_id).edit_count == 2

    test_store.clear(block_id)
    assert test_store.get_record(block_id).edit_count == 3


def test_undo_single_action(test_store):
    """Test undoing a single label action."""
    block_id = list(test_store.records.keys())[0]
    test_store.assign(block_id, 0, "Class A")
    assert test_store.get_record(block_id).status == "labeled"

    test_store.undo()
    record = test_store.get_record(block_id)
    assert record.status == "unlabeled"


def test_undo_redo_sequence(test_store):
    """Test undo/redo sequence."""
    block_id = list(test_store.records.keys())[0]

    # Action 1: assign
    test_store.assign(block_id, 0, "Class A")
    assert test_store.get_record(block_id).class_id == 0

    # Action 2: change class
    test_store.assign(block_id, 1, "Class B")
    assert test_store.get_record(block_id).class_id == 1

    # Undo 2
    test_store.undo()
    assert test_store.get_record(block_id).class_id == 0

    # Undo 1
    test_store.undo()
    assert test_store.get_record(block_id).status == "unlabeled"

    # Redo 1
    test_store.redo()
    assert test_store.get_record(block_id).class_id == 0

    # Redo 2
    test_store.redo()
    assert test_store.get_record(block_id).class_id == 1


def test_redo_stack_clears_on_new_action(test_store):
    """Test that redo stack clears when a new action is taken."""
    block_id = list(test_store.records.keys())[0]

    # Action and undo
    test_store.assign(block_id, 0, "Class A")
    test_store.undo()
    assert len(test_store.redo_stack) == 1

    # New action should clear redo stack
    test_store.assign(block_id, 1, "Class B")
    assert len(test_store.redo_stack) == 0


def test_counts_after_operations(test_store, test_grid):
    """Test that counts are correct after various operations."""
    blocks = list(test_store.records.keys())

    # Label first 10
    for i in range(10):
        test_store.assign(blocks[i], 0, "Class A")
    assert test_store.count_labeled() == 10
    assert test_store.count_unlabeled() == test_grid.num_blocks() - 10

    # Abstain 5
    for i in range(10, 15):
        test_store.assign(blocks[i], -1, "Abstain")
    assert test_store.count_abstained() == 5

    # Nodata 3
    for i in range(15, 18):
        test_store.set_nodata(blocks[i])
    assert test_store.count_nodata() == 3


def test_class_counts(test_store):
    """Test counting blocks per class."""
    blocks = list(test_store.records.keys())

    test_store.assign(blocks[0], 0, "Class A")
    test_store.assign(blocks[1], 0, "Class A")
    test_store.assign(blocks[2], 1, "Class B")
    test_store.assign(blocks[3], 1, "Class B")
    test_store.assign(blocks[4], 1, "Class B")

    counts = test_store.class_counts()
    assert counts[0] == 2
    assert counts[1] == 3


def test_parquet_round_trip(test_store, test_grid, tmp_path):
    """Test saving and loading from Parquet."""
    blocks = list(test_store.records.keys())

    # Create some labels
    test_store.assign(blocks[0], 0, "Class A")
    test_store.assign(blocks[1], 1, "Class B")
    test_store.assign(blocks[2], -1, "Abstain")
    test_store.set_nodata(blocks[3])

    # Save
    parquet_path = tmp_path / "test.parquet"
    test_store.save_parquet(parquet_path)
    assert parquet_path.exists()

    # Load
    loaded_store = LabelStore.load_parquet(parquet_path, test_grid)
    assert loaded_store.count_labeled() == 2
    assert loaded_store.count_abstained() == 1
    assert loaded_store.count_nodata() == 1

    # Verify specific records
    assert loaded_store.get_record(blocks[0]).class_id == 0
    assert loaded_store.get_record(blocks[1]).class_id == 1
    assert loaded_store.get_record(blocks[2]).class_id == -1
    assert loaded_store.get_record(blocks[3]).class_id == -2


def test_label_record_to_from_dict():
    """Test LabelRecord serialization."""
    record = LabelRecord(
        block_id="TEST_0_0",
        obs_id="TEST",
        panel_row=0,
        panel_col=0,
        block_row=0,
        block_col=0,
        x_px=0,
        y_px=0,
        w_px=512,
        h_px=512,
        class_id=1,
        class_name="Class B",
        status="labeled",
        map_x=100.5,
        map_y=200.5,
    )

    data = record.to_dict()
    restored = LabelRecord.from_dict(data)

    assert restored.class_id == 1
    assert restored.class_name == "Class B"
    assert restored.status == "labeled"
    assert restored.map_x == 100.5
