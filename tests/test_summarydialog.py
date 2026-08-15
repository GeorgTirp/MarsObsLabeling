"""Tests for the class Summary window's stats logic (coverage, mean scores)."""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from rasterio.transform import Affine

from marslabeler.classes import load_classes
from marslabeler.io.raster import RasterSource
from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import LabelStore
from marslabeler.model.session import Session
from marslabeler.ui.summarydialog import ClassSummaryDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def classes_scheme(tmp_config_dir):
    return load_classes(tmp_config_dir / "classes.yaml")


@pytest.fixture
def test_session(synthetic_geotiff):
    raster = RasterSource(synthetic_geotiff)
    raster.open()
    grid = Grid(4096, 4096, 4096, 512, "TEST_OBS", Affine.identity())
    labels = LabelStore(grid, "test_user")
    session = Session(raster, grid, labels, {})
    yield session
    raster.close()


def _label_n_blocks(session, class_id, class_name, n):
    block_ids = list(session.labels.records.keys())[:n]
    for bid in block_ids:
        session.labels.assign(bid, class_id, class_name)
    return block_ids


def test_coverage_counts_class_against_non_nodata_total(qapp, classes_scheme, test_session):
    total_blocks = test_session.grid.num_blocks()
    _label_n_blocks(test_session, 0, "Class A", 5)

    dialog = ClassSummaryDialog(classes_scheme, test_session)
    count, total = dialog._coverage(0)

    assert count == 5
    assert total == total_blocks  # nothing marked nodata in this fixture


def test_coverage_excludes_nodata_blocks_from_total(qapp, classes_scheme, test_session):
    total_blocks = test_session.grid.num_blocks()
    block_ids = list(test_session.labels.records.keys())
    # Distinct blocks for nodata vs. labeled, so the label step doesn't overwrite
    # the nodata status back to "labeled" (assign() would resurrect them).
    test_session.labels.set_nodata(block_ids[0])
    test_session.labels.set_nodata(block_ids[1])
    for bid in block_ids[2:5]:
        test_session.labels.assign(bid, 0, "Class A")

    dialog = ClassSummaryDialog(classes_scheme, test_session)
    count, total = dialog._coverage(0)

    assert count == 3
    assert total == total_blocks - 2


def test_coverage_zero_when_class_unused(qapp, classes_scheme, test_session):
    dialog = ClassSummaryDialog(classes_scheme, test_session)
    count, total = dialog._coverage(1)
    assert count == 0
    assert total == test_session.grid.num_blocks()


def test_mean_score_averages_only_blocks_of_that_class(qapp, classes_scheme, test_session):
    block_ids = _label_n_blocks(test_session, 0, "Class A", 3)
    scores = {block_ids[0]: 0.2, block_ids[1]: 0.4, block_ids[2]: 0.6}

    dialog = ClassSummaryDialog(classes_scheme, test_session)
    avg = dialog._mean_score(0, scores)

    assert avg == pytest.approx(0.4)


def test_mean_score_ignores_scores_for_other_classes(qapp, classes_scheme, test_session):
    block_ids = list(test_session.labels.records.keys())
    test_session.labels.assign(block_ids[0], 0, "Class A")
    test_session.labels.assign(block_ids[1], 1, "Class B")
    scores = {block_ids[0]: 0.1, block_ids[1]: 0.9}

    dialog = ClassSummaryDialog(classes_scheme, test_session)
    avg = dialog._mean_score(0, scores)

    assert avg == pytest.approx(0.1)


def test_mean_score_none_when_no_data(qapp, classes_scheme, test_session):
    dialog = ClassSummaryDialog(classes_scheme, test_session)
    assert dialog._mean_score(0, {}) is None


def test_model_index_for_class_defaults_to_id(qapp, classes_scheme, test_session):
    dialog = ClassSummaryDialog(classes_scheme, test_session)
    cls = classes_scheme.classes[1]
    assert dialog._model_index_for_class(cls) == 1


def test_model_index_for_class_uses_explicit_override(qapp, test_session, tmp_config_dir):
    classes_path = tmp_config_dir / "classes.yaml"
    content = classes_path.read_text()
    content = content.replace(
        '{ id: 1,  name: "Class B", color: "#DD8452", hotkey: "w" }',
        '{ id: 1,  name: "Class B", color: "#DD8452", hotkey: "w", model_index: 7 }',
    )
    classes_path.write_text(content)
    scheme = load_classes(classes_path)

    dialog = ClassSummaryDialog(scheme, test_session)
    assert dialog._model_index_for_class(scheme.classes[1]) == 7


def test_summary_dialog_builds_with_no_analysis_data(qapp, classes_scheme, test_session):
    """No inference has run yet -- dialog should still construct cleanly (placeholders)."""
    dialog = ClassSummaryDialog(classes_scheme, test_session)
    assert dialog is not None


def test_summary_dialog_builds_with_npca_gallery(qapp, classes_scheme, test_session):
    """A loaded gallery (numpy thumbnails keyed by model channel index) renders without error."""

    class FakeThumbnail:
        def __init__(self, rank, score):
            self.rank = rank
            self.score = score
            self.thumbnail = np.zeros((16, 16), dtype=np.uint8)
            self.source_id = "fake_source"

    gallery = {0: {0: [FakeThumbnail(1, 0.9)], 1: [FakeThumbnail(1, 0.5)]}}
    dialog = ClassSummaryDialog(classes_scheme, test_session, npca_gallery=gallery)
    assert dialog is not None
