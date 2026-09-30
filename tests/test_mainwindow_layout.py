"""Tests for MainWindow's resizable-splitter layout (history | canvas | right
sidebar), added so the legend panel can be dragged wider instead of being
capped at a fixed width. No session/model needed for the pure-layout checks;
the history-panel-swap check follows test_mainwindow_inference.py's pattern
of assigning a Session directly rather than driving the modal load dialog.
"""

import pytest
from pathlib import Path
from rasterio.transform import Affine

from PySide6.QtWidgets import QApplication, QSplitter

from marslabeler.io.raster import RasterSource
from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import LabelStore
from marslabeler.model.session import Session
from marslabeler.ui.mainwindow import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def window(qapp):
    return MainWindow(Path("configs/app.yaml"), predictions_mode=True)


def test_main_layout_is_a_splitter(window):
    assert isinstance(window.main_layout, QSplitter)


@pytest.mark.parametrize("predictions_mode", [False, True])
def test_workspace_fits_laptop_on_first_show(qapp, predictions_mode):
    win = MainWindow(Path("configs/app.yaml"), predictions_mode=predictions_mode)
    win.resize(1440, 900)
    win.show()
    qapp.processEvents()
    assert win.width() == 1440
    assert win.height() == 900
    assert not win.workspace_save_button.isEnabled()
    assert win.empty_workspace.isVisible()
    win.close()
    win.deleteLater()


def test_splitter_has_history_canvas_legend_and_right_sidebar(window):
    splitter = window.main_layout
    assert splitter.count() == 4
    assert splitter.widget(0) is window.history_panel
    assert splitter.widget(1) is window.canvas
    assert splitter.widget(2) is window.legend_panel
    # widget(3) is the right-sidebar container (preview + action buttons),
    # not stored on window by name.


def test_splitter_children_not_collapsible(window):
    """A dragged-to-zero pane would hide the legend/canvas entirely with no
    way to get it back short of restarting -- must not be possible."""
    for i in range(window.main_layout.count()):
        assert window.main_layout.isCollapsible(i) is False


def test_legend_panel_placeholder_has_no_maximum_width_cap(window):
    """The old LegendPanel.setMaximumWidth(250) blocked dragging the splitter
    handle wider than 250px for that pane; nothing in the window should cap
    the right-sidebar column's placeholder either."""
    # Qt's unset-maximum sentinel is QWIDGETSIZE_MAX (16777215); anything far
    # below that indicates an explicit cap is still in effect.
    assert window.legend_panel.maximumWidth() > 100_000


def test_history_panel_swap_keeps_splitter_column_count(window, synthetic_geotiff):
    """_update_history_panel replaces a widget inside the QSplitter (which has
    no QLayout.replaceWidget()) -- must not leave a stray extra pane behind."""
    raster = RasterSource(synthetic_geotiff)
    raster.open()
    grid = Grid(4096, 4096, 4096, 512, "TEST_OBS", Affine.identity())
    labels = LabelStore(grid, "test_user")
    window.session = Session(raster, grid, labels, window.config.to_dict())

    old_history_panel = window.history_panel
    window._update_history_panel()

    assert window.main_layout.count() == 4
    assert window.main_layout.widget(0) is window.history_panel
    assert window.history_panel is not old_history_panel

    raster.close()


def test_legend_panel_swap_keeps_splitter_column_count_and_position(
    window, tmp_config_dir, synthetic_geotiff
):
    """The legend lives in its own QSplitter column now -- swapping the
    placeholder for the real LegendPanel must keep the column count and the
    legend's index, not append a 5th pane."""
    from marslabeler.classes import load_classes
    from marslabeler.ui.legendpanel import LegendPanel

    window.classes_scheme = load_classes(tmp_config_dir / "classes.yaml")
    old_legend = window.legend_panel

    window._update_legend_panel()

    assert window.main_layout.count() == 4
    assert window.main_layout.widget(2) is window.legend_panel
    assert window.legend_panel is not old_legend
    assert isinstance(window.legend_panel, LegendPanel)


def test_legend_panel_swap_preserves_column_width(window, tmp_config_dir):
    """A user-dragged legend width must survive the placeholder -> real-legend
    swap (the swap re-inserts a widget, which would otherwise re-equalize).

    Needs a realistically-sized window: on an unshown one the splitter's total
    is a few hundred px, so LegendPanel's own minimumWidth dominates and Qt
    legitimately redistributes regardless of what we asked for.
    """
    from marslabeler.classes import load_classes

    window.classes_scheme = load_classes(tmp_config_dir / "classes.yaml")
    # resize (not show) -- enough for the splitter to lay out at a realistic
    # width, without the offscreen-platform teardown noise show() brings.
    window.main_layout.resize(1800, 900)
    QApplication.processEvents()

    window.main_layout.setSizes([180, 900, 400, 300])
    sizes_before = window.main_layout.sizes()
    assert sizes_before[2] >= 400  # precondition: legend column really got its width

    window._update_legend_panel()

    assert window.main_layout.sizes() == sizes_before
