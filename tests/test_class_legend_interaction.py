"""Legend correctness: no hotkey may shadow an app shortcut, and rows are clickable.

Two bugs motivated these. `o` was bound to both "Large Ripples: Isolated" and the
Overview toggle, and since mainwindow.keyPressEvent handles Overview BEFORE class
keys, that class was silently unreachable from the keyboard. And labelling was
keyboard-only, so a new user had to memorise every hotkey before they could
label anything.
"""

from __future__ import annotations

import pytest

from PySide6.QtWidgets import QApplication

from marslabeler.classes import load_classes
from marslabeler.ui.legendpanel import ClassRow, LegendPanel

# Single-character keys the app consumes before class hotkeys are considered.
# 'z' is NOT here: undo/redo are Ctrl+Z / Ctrl+Shift+Z, so plain z is free.
APP_RESERVED_KEYS = {"o"}


@pytest.fixture(scope="module")
def qtbot_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def scheme():
    return load_classes("configs/classes.yaml")


def test_no_class_uses_an_app_reserved_key(scheme):
    """A class bound to one would be unreachable: the app handler runs first."""
    clashes = {
        c.hotkey: c.name for c in scheme.classes.values() if c.hotkey in APP_RESERVED_KEYS
    }
    assert not clashes, (
        f"these classes are shadowed by an app shortcut and cannot be typed: {clashes}"
    )


def test_every_hotkey_is_unique(scheme):
    seen: dict[str, list[str]] = {}
    for cls in list(scheme.classes.values()) + [scheme.abstain]:
        if cls.hotkey:
            seen.setdefault(cls.hotkey, []).append(cls.name)
    duplicates = {k: v for k, v in seen.items() if len(v) > 1}
    assert not duplicates, f"duplicate hotkeys: {duplicates}"


def test_every_class_has_a_hotkey(scheme):
    missing = [c.name for c in scheme.classes.values() if not c.hotkey]
    assert not missing, f"classes with no hotkey are unreachable by keyboard: {missing}"


def test_the_extra_terrain_classes_are_present(scheme):
    names = {c.name for c in scheme.classes.values()}
    for expected in ("Craters", "Yardangs", "Dunes", "Periodic Bedrock Ridges"):
        assert expected in names, f"{expected!r} missing from the legend"


def test_model_index_mapping_has_no_collisions(scheme):
    """Classes beyond the model's channel count must not collide with real ones.

    The stage-3 checkpoints emit 14 channels; ids 14+ are human-only. They still
    need distinct model_index values or `model_index_to_id` raises.
    """
    mapping = scheme.model_index_to_id()
    assert len(mapping) == len(scheme.classes)
    # the 14 real model channels still resolve to the original classes
    for channel in range(14):
        assert mapping[channel] == channel


def test_colors_are_unique(scheme):
    colors: dict[str, list[str]] = {}
    for cls in scheme.classes.values():
        colors.setdefault(cls.color.lower(), []).append(cls.name)
    duplicates = {k: v for k, v in colors.items() if len(v) > 1}
    assert not duplicates, f"classes sharing a color are indistinguishable: {duplicates}"


def test_legend_renders_one_clickable_row_per_class(qtbot_app, scheme):
    panel = LegendPanel(scheme)
    rows = panel.findChildren(ClassRow)
    # every user class plus abstain
    assert len(rows) == len(scheme.classes) + 1
    ids = {row._class_id for row in rows}
    assert ids == set(scheme.classes) | {scheme.abstain.id}


def test_clicking_a_row_emits_that_class_id(qtbot_app, scheme):
    panel = LegendPanel(scheme)
    received: list[int] = []
    panel.on_class_clicked = received.append

    row = next(r for r in panel.findChildren(ClassRow) if r._class_id == 0)
    row.clicked.emit(row._class_id)
    assert received == [0]


def test_row_children_do_not_swallow_the_click(qtbot_app, scheme):
    """The name label covers most of the row; if it ate clicks, only the thin
    padding would respond and clicking would seem intermittent."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel

    panel = LegendPanel(scheme)
    row = panel.findChildren(ClassRow)[0]
    for child in row.findChildren(QLabel):
        assert child.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents), (
            f"{child.text()!r} would swallow the row click"
        )
