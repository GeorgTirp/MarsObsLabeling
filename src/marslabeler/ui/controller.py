"""Keyboard input controller: maps keys to Session actions and UI updates."""

from typing import Optional, Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent

from marslabeler.model.session import Session
from marslabeler.classes import ClassScheme


class KeyboardController:
    """Handles keyboard input and dispatches to Session + UI callbacks."""

    def __init__(self, session: Session, classes_scheme: ClassScheme):
        self.session = session
        self.classes_scheme = classes_scheme

        # Callbacks for UI updates
        self.on_label_changed: Optional[Callable[[], None]] = None
        self.on_panel_changed: Optional[Callable[[], None]] = None
        self.on_cursor_changed: Optional[Callable[[], None]] = None
        self.on_show_help: Optional[Callable[[], None]] = None

        # Track last action for autosave
        self.last_action_type = None

    def handle_key_press(self, event: QKeyEvent) -> bool:
        """
        Handle keyboard event.

        Returns:
            True if event was handled, False to propagate
        """
        if event.isAutoRepeat():
            return False

        key = event.key()
        text = event.text()

        # Navigation keys
        if key == Qt.Key.Key_Right:
            return self._handle_move_right()
        elif key == Qt.Key.Key_Left:
            return self._handle_move_left()
        elif key == Qt.Key.Key_Up:
            return self._handle_move_up()
        elif key == Qt.Key.Key_Down:
            return self._handle_move_down()
        elif key == Qt.Key.Key_PageDown:
            return self._handle_next_panel()
        elif key == Qt.Key.Key_PageUp:
            return self._handle_previous_panel()
        elif key == Qt.Key.Key_Home:
            return self._handle_first_block_in_panel()
        elif key == Qt.Key.Key_Delete or key == Qt.Key.Key_Backspace:
            return self._handle_clear_block()

        # Editing keys
        elif key == Qt.Key.Key_Z and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            return self._handle_undo()
        elif key == Qt.Key.Key_Z and event.modifiers() == (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
            return self._handle_redo()

        # Help key
        elif text == "?":
            return self._handle_show_help()

        # Class hotkeys
        else:
            class_id = self.classes_scheme.hotkey_to_id.get(text)
            if class_id is not None:
                if class_id == -1:  # Abstain
                    return self._handle_abstain()
                else:  # User class
                    return self._handle_label_class(class_id)

        return False

    def _handle_label_class(self, class_id: int) -> bool:
        """Label current block with class and auto-advance."""
        block = self.session.current_block()
        class_name = self.classes_scheme.get_name(class_id)

        # Check if this is an edit (block already labeled)
        record = self.session.labels.get_record(block.block_id)
        is_edit = record.status in ("labeled", "abstain")

        self.session.label_current_block(class_id, class_name)
        self.last_action_type = "label"

        if self.on_label_changed:
            self.on_label_changed()

        # If this causes panel rollover, notify panel change
        if self.session.current_block().panel_idx != self.session.current_panel_idx:
            if self.on_panel_changed:
                self.on_panel_changed()

        return True

    def _handle_abstain(self) -> bool:
        """Abstain on current block and auto-advance."""
        self.session.abstain_current_block()
        self.last_action_type = "abstain"

        if self.on_label_changed:
            self.on_label_changed()

        # Check for panel rollover
        if self.session.current_block().panel_idx != self.session.current_panel_idx:
            if self.on_panel_changed:
                self.on_panel_changed()

        return True

    def _handle_clear_block(self) -> bool:
        """Clear current block back to unlabeled."""
        self.session.clear_current_block()
        self.last_action_type = "clear"

        if self.on_label_changed:
            self.on_label_changed()

        return True

    def _handle_move_right(self) -> bool:
        """Move to next block without labeling."""
        self.session.move_to_next_block()
        if self.on_cursor_changed:
            self.on_cursor_changed()
        return True

    def _handle_move_left(self) -> bool:
        """Move to previous block without labeling."""
        self.session.move_to_previous_block()
        if self.on_cursor_changed:
            self.on_cursor_changed()
        return True

    def _handle_move_up(self) -> bool:
        """Move up in grid (block_col stays same, block_row decreases)."""
        block = self.session.current_block()
        if block.block_row > 0:
            idx = self.session.current_block_idx - self.session.grid.blocks_per_panel_col
            self.session.move_to_block(idx)
            if self.on_cursor_changed:
                self.on_cursor_changed()
        return True

    def _handle_move_down(self) -> bool:
        """Move down in grid (block_col stays same, block_row increases)."""
        block = self.session.current_block()
        if block.block_row < self.session.grid.blocks_per_panel_row - 1:
            idx = self.session.current_block_idx + self.session.grid.blocks_per_panel_col
            self.session.move_to_block(idx)
            if self.on_cursor_changed:
                self.on_cursor_changed()
        return True

    def _handle_next_panel(self) -> bool:
        """Jump to next panel."""
        self.session.move_to_next_panel()
        if self.on_panel_changed:
            self.on_panel_changed()
        return True

    def _handle_previous_panel(self) -> bool:
        """Jump to previous panel."""
        self.session.move_to_previous_panel()
        if self.on_panel_changed:
            self.on_panel_changed()
        return True

    def _handle_first_block_in_panel(self) -> bool:
        """Jump to first block in current panel."""
        self.session.move_to_first_block_in_panel()
        if self.on_cursor_changed:
            self.on_cursor_changed()
        return True

    def _handle_undo(self) -> bool:
        """Undo last label action."""
        self.session.labels.undo()
        if self.on_label_changed:
            self.on_label_changed()
        return True

    def _handle_redo(self) -> bool:
        """Redo last undone action."""
        self.session.labels.redo()
        if self.on_label_changed:
            self.on_label_changed()
        return True

    def _handle_show_help(self) -> bool:
        """Show keyboard help overlay."""
        if self.on_show_help:
            self.on_show_help()
        return True

    def should_autosave(self) -> bool:
        """Check if autosave should trigger based on last action."""
        if self.last_action_type in ("label", "abstain", "clear"):
            return self.session.should_autosave()
        return False

    def reset_autosave(self) -> None:
        """Reset autosave counter after saving."""
        self.session.reset_autosave_counter()
