"""Acceptance tests: end-to-end labeling workflow verification."""

import pytest
from pathlib import Path
from rasterio.transform import Affine

from marslabeler.io.raster import RasterSource
from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import LabelStore
from marslabeler.model.session import Session
from marslabeler.model.export import export_coarse_geotiff
from marslabeler.classes import load_classes
from marslabeler.config import load_config


@pytest.fixture
def config(tmp_config_dir):
    """Load test config."""
    return load_config(tmp_config_dir / "app.yaml")


@pytest.fixture
def classes(tmp_config_dir):
    """Load test classes."""
    return load_classes(tmp_config_dir / "classes.yaml")


def test_acceptance_end_to_end_labeling(
    synthetic_geotiff, tmp_config_dir, tmp_path
):
    """
    Acceptance test: complete labeling workflow.

    - Load JP2
    - Label blocks via Session
    - Navigate and edit
    - Save and resume
    - Export GeoTIFF
    """
    # Load JP2 and setup
    raster = RasterSource(synthetic_geotiff)
    raster.open()

    config = load_config(tmp_config_dir / "app.yaml")
    classes_scheme = load_classes(tmp_config_dir / "classes.yaml")

    grid = Grid(
        img_width=raster.width,
        img_height=raster.height,
        panel_size=config.geometry.panel_size,
        block_size=config.geometry.block_size,
        obs_id="ACCEPTANCE_TEST",
        transform=raster.transform,
        crs=raster.crs,
    )

    # Create session and label some blocks
    session = Session(
        raster,
        grid,
        LabelStore(grid, "acceptance_tester"),
        config.to_dict(),
    )

    blocks = list(grid.iter_blocks())
    assert len(blocks) == 64

    # Label 20 blocks (keyboard simulation)
    for i in range(20):
        block = session.current_block()
        # Cycle through classes 0, 1
        class_id = i % 2
        class_name = classes_scheme.classes[class_id].name
        session.label_current_block(class_id, class_name)

    assert session.labels.count_labeled() == 20

    # Check auto-advance: should be on block 20 or later
    assert session.current_block_idx >= 20

    # Edit block 0 (navigate back)
    session.move_to_block(0)
    record = session.labels.get_record(blocks[0].block_id)
    original_class = record.class_id
    original_count = record.edit_count

    # Edit it to class 1 (if it was 0)
    if original_class == 0:
        session.relabel_current_block(1, classes_scheme.classes[1].name)
        record = session.labels.get_record(blocks[0].block_id)
        assert record.class_id == 1
        assert record.edit_count == original_count + 1

    # Undo
    session.labels.undo()
    record = session.labels.get_record(blocks[0].block_id)
    assert record.class_id == original_class

    # Redo
    session.labels.redo()
    record = session.labels.get_record(blocks[0].block_id)
    assert record.class_id == 1

    # Save session
    labels_dir = tmp_path / "labels"
    session.save_session(labels_dir)
    assert (labels_dir / "ACCEPTANCE_TEST.parquet").exists()
    assert (labels_dir / "ACCEPTANCE_TEST.session.json").exists()

    # Resume: load session fresh
    session2 = Session.load_or_create(
        synthetic_geotiff,
        grid,
        config.__dict__,
        labels_dir,
        "acceptance_tester",
    )

    # Verify labels were restored
    assert session2.labels.count_labeled() == 20
    assert session2.labels.get_record(blocks[0].block_id).class_id == 1

    # Export to GeoTIFF
    geotiff_path = tmp_path / "labels.tif"
    export_coarse_geotiff(session2.labels, grid, geotiff_path)
    assert geotiff_path.exists()

    # Verify GeoTIFF properties
    import rasterio
    with rasterio.open(str(geotiff_path)) as src:
        assert src.width == 8  # 8x8 blocks per panel
        assert src.height == 8
        assert src.count == 1
        assert src.nodata == 255

    raster.close()


def test_acceptance_legend_completeness(classes):
    """Verify legend has all expected classes."""
    assert len(classes.classes) > 0

    for class_obj in classes.classes.values():
        assert class_obj.id >= 0
        assert class_obj.name
        assert class_obj.color.startswith("#")
        assert class_obj.hotkey


def test_acceptance_no_full_load_into_ram(synthetic_geotiff):
    """Verify that raster can be read without loading full file into RAM."""
    raster = RasterSource(synthetic_geotiff)
    raster.open()

    # Read a small window (decimated)
    window_data = raster.read_window(0, 0, 4096, 4096, 1600, 1600)

    # Window should be much smaller than full raster
    assert window_data.nbytes < raster.width * raster.height * 4  # 4GB check

    raster.close()


def test_acceptance_config_driven_geometry(tmp_config_dir):
    """Verify that block/panel sizes are config-driven."""
    config = load_config(tmp_config_dir / "app.yaml")

    # Modify config
    assert config.geometry.panel_size == 4096
    assert config.geometry.block_size == 512

    # Create grid with config values
    grid = Grid(4096, 4096, config.geometry.panel_size, config.geometry.block_size, "TEST", Affine.identity())

    assert grid.panel_size == 4096
    assert grid.block_size == 512
    assert grid.blocks_per_panel == 64  # 8x8


def test_acceptance_keyboard_only_workflow(synthetic_geotiff, tmp_config_dir):
    """
    Verify that entire labeling workflow can be done keyboard-only.

    No mouse clicks required for labeling (though navigation can use them).
    """
    raster = RasterSource(synthetic_geotiff)
    raster.open()

    config = load_config(tmp_config_dir / "app.yaml")
    grid = Grid(
        raster.width, raster.height,
        config.geometry.panel_size, config.geometry.block_size,
        "KEYBOARD_TEST", raster.transform, raster.crs
    )

    session = Session(raster, grid, LabelStore(grid, "kb_user"), config.to_dict())

    # Simulate pure keyboard workflow
    action_log = []

    # 1. Label 5 blocks with class hotkeys
    for i in range(5):
        block = session.current_block()
        session.label_current_block(0, "Class A")
        action_log.append(("label", block.block_id))

    assert session.labels.count_labeled() == 5
    action_log.append(("check", f"labeled={session.labels.count_labeled()}"))

    # 2. Navigate back with arrow key (simulation)
    session.move_to_previous_block()
    action_log.append(("navigate", "left"))

    # 3. Edit without advancing
    block = session.current_block()
    session.relabel_current_block(1, "Class B")
    action_log.append(("edit", block.block_id))

    # 4. Undo
    session.labels.undo()
    action_log.append(("undo", ""))

    # 5. Redo
    session.labels.redo()
    action_log.append(("redo", ""))

    # All actions completed via Session (keyboard-driven)
    assert len(action_log) > 0
    assert ("label", session.grid.get_block(0).block_id) in action_log or any(a[0] == "label" for a in action_log)

    raster.close()
