"""Tests for UI components (headless)."""

import numpy as np
import pytest
from pathlib import Path
from rasterio.transform import Affine

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor

from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import LabelStore
from marslabeler.ui.panelcanvas import PanelCanvas
from marslabeler.ui.sidepreview import SidePreview
from marslabeler.ui.legendpanel import LegendPanel
from marslabeler.ui.historypanel import HistoryPanel
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
def test_store(test_grid):
    """Create a test label store with some labels."""
    store = LabelStore(test_grid, "test_user")
    blocks = list(store.records.keys())

    # Add some labels
    store.assign(blocks[0], 0, "Class A")
    store.assign(blocks[1], 1, "Class B")
    store.assign(blocks[2], -1, "Abstain")

    return store


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Create temp config files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    classes_yaml = config_dir / "classes.yaml"
    classes_yaml.write_text("""
classes:
  - { id: 0,  name: "Class A", color: "#FF0000", hotkey: "q" }
  - { id: 1,  name: "Class B", color: "#00FF00", hotkey: "w" }
abstain:
  id: -1
  name: "Abstain"
  color: "#000000"
  hotkey: "space"
nodata:
  id: -2
  name: "No data"
  color: "#222222"
""")

    return config_dir


def test_panel_canvas_creation(qapp):
    """Test PanelCanvas widget creation."""
    canvas = PanelCanvas()
    assert canvas is not None
    assert canvas.canvas_width == 1600
    assert canvas.canvas_height == 1600


def test_panel_canvas_set_image(qapp):
    """Test setting panel image."""
    canvas = PanelCanvas()
    panel_data = np.random.randint(0, 256, (1600, 1600), dtype=np.uint8)
    canvas.set_panel_image(panel_data)

    assert canvas.image_item is not None


def test_panel_canvas_set_grid(qapp):
    """Test setting grid."""
    canvas = PanelCanvas()
    canvas.set_grid(8, 8)

    assert canvas.blocks_per_row == 8
    assert canvas.blocks_per_col == 8
    assert canvas.block_width == 200
    assert canvas.block_height == 200
    assert canvas.grid_item is not None


def test_panel_canvas_set_label_overlay(qapp):
    """Test setting label overlay."""
    canvas = PanelCanvas()
    canvas.set_grid(8, 8)

    block_data = np.zeros((8, 8), dtype=np.int16)
    block_data[0, 0] = 0  # Class A
    block_data[0, 1] = 1  # Class B

    class_colors = {0: "#FF0000", 1: "#00FF00"}
    canvas.set_label_overlay(block_data, class_colors)

    assert canvas.label_overlay_item is not None


def test_panel_canvas_highlight(qapp):
    """Test setting current block highlight."""
    canvas = PanelCanvas()
    canvas.set_grid(8, 8)
    canvas.set_current_block_highlight(2, 3)

    assert canvas.highlight_item is not None


def test_side_preview_creation(qapp):
    """Test SidePreview widget creation."""
    preview = SidePreview()
    assert preview is not None


def test_side_preview_set_image(qapp):
    """Test setting block image."""
    preview = SidePreview()
    block_data = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    preview.set_block_image(block_data, "TEST_BLOCK_0_0")

    assert preview.image_label.pixmap() is not None


def test_side_preview_empty_image(qapp):
    """Test setting empty image."""
    preview = SidePreview()
    block_data = np.array([], dtype=np.uint8).reshape(0, 0)
    preview.set_block_image(block_data)

    assert preview.image_label.pixmap() is None or preview.image_label.text() == "(empty)"


def test_legend_panel_creation(qapp, tmp_config_dir):
    """Test LegendPanel widget creation."""
    classes_scheme = load_classes(tmp_config_dir / "classes.yaml")
    legend = LegendPanel(classes_scheme)

    assert legend is not None


def test_history_panel_creation(qapp, test_grid, test_store):
    """Test HistoryPanel widget creation."""
    history = HistoryPanel(test_grid, test_store)
    assert history is not None


def test_history_panel_counts(qapp, test_grid, test_store):
    """Test that history panel reflects label counts."""
    history = HistoryPanel(test_grid, test_store)

    # History panel should be created with 1 panel
    assert test_grid.num_panels == 1
    assert test_store.count_labeled() == 2  # Two labeled blocks
    assert test_store.count_abstained() == 1  # One abstained block
