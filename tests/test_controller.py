"""Tests for keyboard controller."""

import pytest
from rasterio.transform import Affine
from PySide6.QtGui import QKeyEvent
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from marslabeler.io.raster import RasterSource
from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import LabelStore
from marslabeler.model.session import Session
from marslabeler.ui.controller import KeyboardController
from marslabeler.classes import load_classes


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for all tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def test_grid():
    """Create a test grid."""
    return Grid(4096, 4096, 4096, 512, "TEST_OBS", Affine.identity())


@pytest.fixture
def test_session(synthetic_geotiff, test_grid):
    """Create a test session."""
    raster = RasterSource(synthetic_geotiff)
    raster.open()
    labels = LabelStore(test_grid, "test_user")
    config = {
        "navigation": {"advance_mode": "next_unlabeled", "advance_on_edit": False},
        "skip": {"nodata_skip_threshold": 0.5, "variance_skip_threshold": 0.0},
        "autosave": {"every_n_labels": 5, "every_seconds": 60},
    }
    session = Session(raster, test_grid, labels, config)
    yield session
    raster.close()


@pytest.fixture
def test_classes(tmp_config_dir):
    """Load test classes."""
    return load_classes(tmp_config_dir / "classes.yaml")


@pytest.fixture
def controller(test_session, test_classes):
    """Create a keyboard controller."""
    return KeyboardController(test_session, test_classes)


def test_controller_creation(controller):
    """Test controller initializes."""
    assert controller is not None
    assert controller.session is not None
    assert controller.classes_scheme is not None


def test_class_hotkey_label(qapp, controller):
    """Test pressing a class hotkey labels and auto-advances."""
    block_id = controller.session.current_block().block_id
    initial_idx = controller.session.current_block_idx

    # Simulate pressing 'q' (Class A, id=0)
    event = QKeyEvent(QKeyEvent.Type.KeyPress, 0, Qt.KeyboardModifier.NoModifier, "q")
    result = controller.handle_key_press(event)

    assert result is True
    record = controller.session.labels.get_record(block_id)
    assert record.status == "labeled"
    assert record.class_id == 0
    # Should have auto-advanced
    assert controller.session.current_block_idx > initial_idx


def test_abstain_hotkey(qapp, controller):
    """Test abstain key advances."""
    block_id = controller.session.current_block().block_id
    initial_idx = controller.session.current_block_idx

    event = QKeyEvent(QKeyEvent.Type.KeyPress, 0, Qt.KeyboardModifier.NoModifier, " ")
    result = controller.handle_key_press(event)

    assert result is True
    record = controller.session.labels.get_record(block_id)
    assert record.status == "abstain"
    assert controller.session.current_block_idx > initial_idx


def test_arrow_keys_move_without_label(qapp, controller):
    """Test arrow keys move without labeling."""
    initial_idx = controller.session.current_block_idx
    block_id = controller.session.current_block().block_id

    # Press right arrow
    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier)
    result = controller.handle_key_press(event)

    assert result is True
    assert controller.session.current_block_idx == initial_idx + 1
    # Original block should still be unlabeled
    record = controller.session.labels.get_record(block_id)
    assert record.status == "unlabeled"


def test_arrow_left_moves_back(qapp, controller):
    """Test left arrow moves cursor back."""
    controller.session.move_to_block(10)
    initial_idx = controller.session.current_block_idx

    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Left, Qt.KeyboardModifier.NoModifier)
    result = controller.handle_key_press(event)

    assert result is True
    assert controller.session.current_block_idx == initial_idx - 1


def test_backspace_clears_block(qapp, controller):
    """Test Backspace clears current block."""
    block_id = controller.session.current_block().block_id

    # First label it
    event = QKeyEvent(QKeyEvent.Type.KeyPress, 0, Qt.KeyboardModifier.NoModifier, "q")
    controller.handle_key_press(event)
    assert controller.session.labels.get_record(block_id).status == "labeled"

    # Move back
    controller.session.move_to_block(0)

    # Clear
    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier)
    result = controller.handle_key_press(event)

    assert result is True
    record = controller.session.labels.get_record(block_id)
    assert record.status == "unlabeled"


def test_undo_redo(qapp, controller):
    """Test Ctrl+Z undo and Ctrl+Shift+Z redo."""
    block_id = controller.session.current_block().block_id

    # Label
    event = QKeyEvent(QKeyEvent.Type.KeyPress, 0, Qt.KeyboardModifier.NoModifier, "q")
    controller.handle_key_press(event)
    assert controller.session.labels.get_record(block_id).status == "labeled"

    # Move back to verify
    controller.session.move_to_block(0)

    # Undo
    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    result = controller.handle_key_press(event)
    assert result is True
    record = controller.session.labels.get_record(block_id)
    assert record.status == "unlabeled"

    # Redo
    event = QKeyEvent(
        QKeyEvent.Type.KeyPress,
        Qt.Key.Key_Z,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    )
    result = controller.handle_key_press(event)
    assert result is True
    record = controller.session.labels.get_record(block_id)
    assert record.status == "labeled"


def test_page_down_next_panel(qapp, controller):
    """Test PageDown moves to next panel."""
    initial_panel = controller.session.current_block().panel_idx

    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_PageDown, Qt.KeyboardModifier.NoModifier)
    result = controller.handle_key_press(event)

    assert result is True
    # With 4096x4096 and 4096 panel, only 1 panel, so should stay at 0
    assert controller.session.current_block().panel_idx == initial_panel


def test_autosave_check(qapp, controller):
    """Test autosave threshold detection."""
    # Config has every_n_labels: 5
    assert not controller.should_autosave()

    # Label 5 blocks
    for i in range(5):
        event = QKeyEvent(QKeyEvent.Type.KeyPress, 0, Qt.KeyboardModifier.NoModifier, "q")
        controller.handle_key_press(event)

    # Should trigger autosave
    assert controller.should_autosave()

    # Reset
    controller.reset_autosave()
    assert not controller.should_autosave()


def test_last_action_tracking(qapp, controller):
    """Test that last action type is tracked."""
    assert controller.last_action_type is None

    # Label action
    event = QKeyEvent(QKeyEvent.Type.KeyPress, 0, Qt.KeyboardModifier.NoModifier, "q")
    controller.handle_key_press(event)
    assert controller.last_action_type == "label"

    # Clear action
    controller.session.move_to_block(0)
    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier)
    controller.handle_key_press(event)
    assert controller.last_action_type == "clear"

    # Move action (shouldn't change last_action_type)
    old_action = controller.last_action_type
    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier)
    controller.handle_key_press(event)
    assert controller.last_action_type == old_action


def test_callback_invocation(qapp, controller):
    """Test that callbacks are invoked on label change."""
    callback_invoked = []

    def on_label_changed():
        callback_invoked.append(True)

    controller.on_label_changed = on_label_changed

    event = QKeyEvent(QKeyEvent.Type.KeyPress, 0, Qt.KeyboardModifier.NoModifier, "q")
    controller.handle_key_press(event)

    assert len(callback_invoked) == 1


def test_invalid_key_not_handled(qapp, controller):
    """Test that invalid keys are not handled."""
    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_F1, Qt.KeyboardModifier.NoModifier)
    result = controller.handle_key_press(event)

    assert result is False


def test_autorepeat_ignored(qapp, controller):
    """Test that autorepeat key events are ignored."""
    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier)
    event.setAutoRepeat(True)
    result = controller.handle_key_press(event)

    assert result is False
