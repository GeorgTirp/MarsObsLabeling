"""Tests for npca_gallery.py's source_id parsing (click-to-navigate support
for the Class Summary window's Neural-PCA thumbnails)."""

import pytest

from marslabeler.inference.npca_gallery import parse_npca_source_id


def test_parse_simple_stem():
    assert parse_npca_source_id("imagery_1234_5678") == ("imagery", 1234, 5678)


def test_parse_stem_with_underscores():
    """fit_neural_pca.py's real imagery stems (e.g. 'drg_on_label_grid') have
    underscores themselves -- only the trailing two fields are col/row."""
    assert parse_npca_source_id("drg_on_label_grid_1234_5678") == ("drg_on_label_grid", 1234, 5678)
    assert parse_npca_source_id("drg_on_label_grid_aoi_42_7") == ("drg_on_label_grid_aoi", 42, 7)


def test_parse_zero_coordinates():
    assert parse_npca_source_id("mosaic_0_0") == ("mosaic", 0, 0)


def test_parse_returns_none_for_missing_coordinates():
    assert parse_npca_source_id("just_a_stem") is None
    assert parse_npca_source_id("onlyonefield") is None
    assert parse_npca_source_id("") is None


def test_parse_returns_none_when_trailing_fields_not_integers():
    assert parse_npca_source_id("mosaic_abc_def") is None
    assert parse_npca_source_id("mosaic_123_def") is None


def test_parse_returns_none_for_empty_stem():
    assert parse_npca_source_id("_123_456") is None


@pytest.mark.parametrize(
    "col,row",
    [(0, 0), (1, 999999), (42, 7), (1000000, 2000000)],
)
def test_parse_round_trips_with_construction_convention(col, row):
    """Matches fit_neural_pca.py line 134's own f-string exactly."""
    from pathlib import Path

    imagery_path = "data/some/dir/drg_on_label_grid.tif"
    source_id = f"{Path(imagery_path).stem}_{col}_{row}"
    assert parse_npca_source_id(source_id) == ("drg_on_label_grid", col, row)
