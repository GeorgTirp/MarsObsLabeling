"""Elena's workflow: progress, panel entry, erasing, keys, and larger viewer."""

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QLabel, QMessageBox

from marslabeler.classes import load_classes
from marslabeler.shortcuts import save_shortcuts
from marslabeler.ui.helpdialog import HelpDialog
from marslabeler.ui.shortcutdialog import ShortcutDialog
from tests.test_labeling_workflow import ROOT, press, qapp, workflow


def records(win, indices):
    return [win.session.labels.get_record(win.session.grid.get_block(i).block_id) for i in indices]


def seed(win, indices, cid=0):
    ids = [r.block_id for r in records(win, indices)]
    win.session.labels.bulk_assign(ids, cid, win.classes_scheme.get_name(cid))
    win._refresh_view()
    win._refresh_history()


def drag(win, start, end, modifier=Qt.KeyboardModifier.NoModifier):
    canvas = win.canvas
    def point(cell):
        r, c = cell
        return canvas.mapFromScene(QPointF((c + 0.5) * canvas.block_width,
                                           (r + 0.5) * canvas.block_height))
    a, b = point(start), point(end)
    QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, modifier, a)
    # Deliberately deliver just one move event, crossing intermediate blocks.
    move = QMouseEvent(QEvent.Type.MouseMove, QPointF(b), QPointF(canvas.viewport().mapToGlobal(b)),
                       Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, modifier)
    QApplication.sendEvent(canvas.viewport(), move)
    QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, modifier, b)
    QApplication.processEvents()


def test_progress_counts_real_edge_blocks_and_survives_resume(workflow):
    win, _, _, _, reopen = workflow
    assert win.label_progress.maximum() == 45  # ceil(281/32) * ceil(153/32), not padded 96
    assert win.label_progress.value() == 0
    press(win, Qt.Key_Q)
    press(win, Qt.Key_Space)
    assert win.label_progress.value() == 1
    assert "1 skipped" in win.label_progress.toolTip()
    seed(win, [2], cid=1)
    win.session.labels.set_nodata_bulk([records(win, [3])[0].block_id])
    win._refresh_history()
    assert win.label_progress.value() == 2
    assert "1 no data" in win.label_progress.toolTip()
    assert "2 / 45" in win.label_progress.format()
    assert win._autosave_session(note="test")
    resumed = reopen()
    assert resumed.label_progress.value() == 2
    assert resumed.label_progress.maximum() == 45


@pytest.mark.parametrize("entry", ["history", "overview", "pageup", "pagedown", "automatic"])
def test_enter_panel_starts_at_first_unlabeled(workflow, entry):
    win, *_ = workflow
    seed(win, [16, 17], cid=1)
    seed(win, [18], cid=-1)
    win.session.labels.set_nodata_bulk([records(win, [19])[0].block_id])
    if entry == "history":
        win._on_panel_selected(1)
    elif entry == "overview":
        win._set_view("overview")
        win.canvas.on_block_clicked(0, 1)
    elif entry == "pageup":
        win._on_panel_selected(2)
        press(win, Qt.Key_PageUp)
    elif entry == "pagedown":
        press(win, Qt.Key_PageDown)
    else:
        seed(win, range(15))
        win.session.move_to_block(15)
        press(win, Qt.Key_Q)
    assert win.session.current_block_idx == 20
    assert win.current_panel_idx == 1
    assert win.session.current_block().block_id == records(win, [20])[0].block_id
    # A direct click can still review a labeled block.
    win.canvas.on_block_clicked(0, 0)
    assert win.session.current_block_idx == 16


def test_complete_panel_remains_reviewable_and_empty_padding_is_skipped(workflow):
    win, *_ = workflow
    seed(win, range(16))
    win._on_panel_selected(0)
    assert win.session.current_block_idx == 0
    # Narrow right-edge panel: next in-image block is one row below, not padded col 1.
    seed(win, [32])
    win._on_panel_selected(2)
    assert win.session.current_block_idx == 36


def test_erase_drag_crosses_all_cells_and_is_one_undo_step(workflow):
    win, *_ = workflow
    seed(win, [0, 1, 2, 3, 4], cid=11)
    before = len(win.session.labels.undo_stack)
    win.erase_button.click()
    drag(win, (0, 0), (0, 3))
    assert [r.status for r in records(win, range(5))] == ["unlabeled"] * 4 + ["labeled"]
    assert win.label_progress.value() == 1
    assert len(win.session.labels.undo_stack) == before + 1
    press(win, Qt.Key_Z, Qt.ControlModifier)
    assert [r.class_id for r in records(win, range(5))] == [11] * 5
    assert win.label_progress.value() == 5
    press(win, Qt.Key_Z, Qt.ControlModifier | Qt.ShiftModifier)
    assert win.label_progress.value() == 1


def test_erase_ignores_empty_start_and_nodata_preserves_undo_and_disk(workflow):
    win, _, _, _, reopen = workflow
    seed(win, [1], cid=-1)
    seed(win, [2], cid=12)
    win.session.labels.set_nodata_bulk([records(win, [3])[0].block_id])
    before = len(win.session.labels.undo_stack)
    win.erase_button.click()
    drag(win, (0, 0), (0, 3))
    assert [r.status for r in records(win, range(4))] == ["unlabeled"] * 3 + ["nodata"]
    assert len(win.session.labels.undo_stack) == before + 1
    assert win._autosave_session(note="test erase")
    resumed = reopen()
    assert [r.status for r in records(resumed, range(4))] == ["unlabeled"] * 3 + ["nodata"]
    press(win, Qt.Key_Z, Qt.ControlModifier)
    assert [r.class_id for r in records(win, range(4))] == [-3, -1, 12, -2]


def test_rectangular_erase_undo_and_noop_does_not_add_undo(workflow):
    win, *_ = workflow
    seed(win, range(16))
    win.saved_complete_panels.add(0)
    before = len(win.session.labels.undo_stack)
    drag(win, (0, 0), (1, 1), Qt.ShiftModifier)
    press(win, Qt.Key_Backspace)
    assert win.selection_rect is None
    assert win.label_progress.value() == 12
    assert 0 not in win.saved_complete_panels
    assert len(win.session.labels.undo_stack) == before + 1
    win.erase_button.click()
    drag(win, (0, 0), (0, 1))
    assert len(win.session.labels.undo_stack) == before + 1
    press(win, Qt.Key_Z, Qt.ControlModifier)
    assert win.label_progress.value() == 16


def test_releasing_brush_key_before_mouse_still_finalizes_progress_and_autosave(workflow):
    win, _, _, _, reopen = workflow
    win.held_class_id = 11
    for c in range(3):
        win._on_block_paint(0, c, c == 0)
    win.held_class_id = None
    win._on_block_paint_end()
    assert win.label_progress.value() == 3
    resumed = reopen()  # fixture autosaves every 3 changed blocks
    assert [r.class_id for r in records(resumed, range(3))] == [11] * 3


def test_overview_click_does_not_erase_new_panel(workflow):
    win, *_ = workflow
    seed(win, range(16, 32), cid=12)
    win.erase_button.click()
    win._set_view("overview")
    drag(win, (0, 1), (0, 1))
    assert win.current_panel_idx == 1
    assert [r.class_id for r in records(win, range(16, 32))] == [12] * 16


@pytest.mark.parametrize("starting_view", ["panel", "overview"])
def test_large_block_preview_and_focus_keep_labeling_available(workflow, starting_view):
    win, *_ = workflow
    assert win.inspector_scroll.isVisible()
    assert win.details_button.isChecked()
    assert win.preview.image_label.pixmap().width() >= 500
    width = win.canvas.width()
    win.details_button.click()
    QApplication.processEvents()
    assert not win.inspector_scroll.isVisible()
    assert win.canvas.width() > width
    win.details_button.click()
    QApplication.processEvents()
    sizes = win.main_layout.sizes()
    crop_size = win.preview.image_label.pixmap().width()
    win._set_view(starting_view)
    win.focus_button.click()
    QApplication.processEvents()
    assert win.preview.image_label.pixmap().width() > crop_size
    assert win.legend_panel.isVisible()
    assert win.view_mode == "panel"
    assert not win.canvas.isVisible()
    assert not win.history_panel.isVisible()
    # Hidden divider handles free a few pixels; the legend must stay narrow.
    assert abs(win.legend_panel.width() - sizes[2]) <= 2 * win.main_layout.handleWidth()
    assert win.export_button.isVisible() and win.next_panel_button.isVisible()
    assert win.label_progress.isVisible()
    # Use the actual keyboard focus: no hidden canvas or focused button may
    # swallow class/skip keys while decisions are made from the large crop.
    press(QApplication.focusWidget() or win, Qt.Key_Q)
    press(QApplication.focusWidget() or win, Qt.Key_Space)
    assert [r.class_id for r in records(win, [0, 1])] == [0, -1]
    assert win.session.current_block_idx == 2
    assert win.session.current_block().block_id in win.preview.info_label.text()
    win._update_legend_panel()
    win._update_history_panel()
    assert win.legend_panel.isVisible() and not win.history_panel.isVisible()
    win.focus_button.click()
    QApplication.processEvents()
    assert win.inspector_scroll.isVisible() and win.legend_panel.isVisible()
    assert win.canvas.isVisible()
    assert win.main_layout.sizes() == sizes


def test_custom_keys_apply_immediately_and_persist_without_changing_labels(workflow, monkeypatch):
    win, _, _, _, reopen = workflow
    seed(win, [0], cid=12)
    before = records(win, [0])[0].to_dict()
    original_yaml = (ROOT / "configs/classes.yaml").read_bytes()
    def customize(dialog):
        dialog.fields[0].setText("1")
        dialog.fields[-1].setText("2")
        assert not dialog.error_label.text()
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(ShortcutDialog, "exec", customize)
    win._customize_shortcuts()
    assert win.controller.classes_scheme.hotkey_to_id["1"] == 0
    assert win.legend_panel.classes_scheme.hotkey_to_id["2"] == -1
    assert records(win, [0])[0].to_dict() == before
    assert (ROOT / "configs/classes.yaml").read_bytes() == original_yaml
    win.session.move_to_block(1)
    press(win, Qt.Key_1)
    press(win, Qt.Key_2)
    assert [r.class_id for r in records(win, range(3))] == [12, 0, -1]
    assert win._autosave_session(note="test keys")
    resumed = reopen()
    assert resumed.classes_scheme.hotkey_to_id["1"] == 0
    resumed.session.move_to_block(3)
    press(resumed, Qt.Key_1, Qt.ControlModifier)  # must never become a terrain edit
    assert records(resumed, [3])[0].status == "unlabeled"
    press(resumed, Qt.Key_1)
    assert records(resumed, [3])[0].class_id == 0
    # Old Q binding is gone, not a duplicate hidden shortcut.
    press(resumed, Qt.Key_Q)
    assert records(resumed, [4])[0].status == "unlabeled"


def test_shortcut_dialog_rejects_collisions_and_can_restore_defaults(qapp, tmp_config_dir):
    scheme = load_classes(tmp_config_dir / "classes.yaml")
    dialog = ShortcutDialog(scheme, scheme)
    save = dialog.buttons.button(QDialogButtonBox.StandardButton.Save)
    for bad in ("w", "space", "O", "?", "+", "ab"):
        dialog.fields[0].setText(bad)
        assert not save.isEnabled(), bad
        assert dialog.error_label.text()
        assert scheme.classes[0].hotkey == "q"  # invalid edits leave live scheme unchanged
    dialog.fields[0].setText("X")
    assert save.isEnabled()
    assert dialog.selected_scheme.hotkey_to_id["x"] == 0
    dialog.fields[0].setText("")
    assert "q" not in dialog.selected_scheme.hotkey_to_id
    dialog._restore_defaults()
    assert dialog.selected_scheme.hotkey_to_id == scheme.hotkey_to_id
    dialog.deleteLater()


def test_failed_shortcut_save_keeps_previous_preferences(workflow, monkeypatch):
    win, *_ = workflow
    save_shortcuts(win.shortcuts_path, win.classes_scheme)
    before = win.shortcuts_path.read_bytes()
    def customize(dialog):
        dialog.fields[0].setText("1")
        return QDialog.DialogCode.Accepted
    warnings = []
    monkeypatch.setattr(ShortcutDialog, "exec", customize)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    def fail_replace(*args):
        raise OSError("simulated locked preferences")
    monkeypatch.setattr("marslabeler.io.atomic.os.replace", fail_replace)
    win._customize_shortcuts()
    assert warnings and "locked" in warnings[0]
    assert win.shortcuts_path.read_bytes() == before
    assert win.classes_scheme.hotkey_to_id["q"] == 0


def test_help_and_abstain_use_requested_wording(qapp):
    scheme = load_classes(ROOT / "configs/classes.yaml")
    assert scheme.abstain.name == "Abstain/Skip (no label assigned)"
    dialog = HelpDialog(scheme)
    text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
    assert "Hold a class key and select blocks" in text
    assert "Hold a class key + drag" not in text
    assert scheme.abstain.name in text
    dialog.deleteLater()
