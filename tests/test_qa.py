"""Tests for QA and analysis tools."""

import numpy as np
import pytest
from pathlib import Path
from rasterio.transform import Affine

from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import LabelStore
from marslabeler.qa.progress import calculate_progress, format_progress_text, ProgressStats
from marslabeler.qa.agreement import calculate_cohens_kappa, compare_labelers


@pytest.fixture
def test_grid():
    """Create a test grid."""
    return Grid(4096, 4096, 4096, 512, "TEST_OBS", Affine.identity())


@pytest.fixture
def test_store(test_grid):
    """Create a label store with some labels."""
    store = LabelStore(test_grid, "test_user")
    blocks = list(store.records.keys())

    # Add labels
    for i in range(10):
        store.assign(blocks[i], i % 4, f"Class {i % 4}")
    for i in range(10, 12):
        store.assign(blocks[i], -1, "Abstain")
    for i in range(12, 15):
        store.set_nodata(blocks[i])

    return store


def test_calculate_progress(test_grid, test_store):
    """Test progress calculation."""
    stats = calculate_progress(test_grid, test_store)

    assert stats.obs_id == "TEST_OBS"
    assert stats.total_blocks == test_grid.num_blocks()
    assert stats.labeled == 10
    assert stats.abstained == 2
    assert stats.nodata == 3
    assert stats.unlabeled == test_grid.num_blocks() - 15


def test_progress_percent_complete(test_grid, test_store):
    """Test percent complete calculation."""
    stats = calculate_progress(test_grid, test_store)

    expected_complete = 15 / test_grid.num_blocks() * 100
    assert abs(stats.percent_complete - expected_complete) < 0.1


def test_progress_class_counts(test_grid, test_store):
    """Test per-class counts."""
    stats = calculate_progress(test_grid, test_store)

    # 10 labeled blocks: 0,1,2,3,0,1,2,3,0,1 → class 0: 3, 1: 3, 2: 2, 3: 2
    assert stats.class_counts[0] == 3
    assert stats.class_counts[1] == 3
    assert stats.class_counts[2] == 2
    assert stats.class_counts[3] == 2


def test_format_progress_text(test_grid, test_store):
    """Test progress text formatting."""
    stats = calculate_progress(test_grid, test_store)
    text = format_progress_text(stats)

    assert "TEST_OBS" in text
    assert "Labeled:" in text
    assert "Cohen's kappa" not in text  # Not in progress text


def test_cohens_kappa_perfect_agreement():
    """Test kappa with perfect agreement."""
    cm = np.array([[10, 0], [0, 10]], dtype=int)
    kappa = calculate_cohens_kappa(cm)

    # Perfect agreement should be 1.0
    assert abs(kappa - 1.0) < 0.01


def test_cohens_kappa_random_agreement():
    """Test kappa with random agreement."""
    # Perfectly random 50/50 agreement
    cm = np.array([[5, 5], [5, 5]], dtype=int)
    kappa = calculate_cohens_kappa(cm)

    # Should be 0.0 for random
    assert abs(kappa - 0.0) < 0.01


def test_cohens_kappa_no_disagreement():
    """Test kappa with no disagreement."""
    cm = np.array([[10, 0, 0], [0, 5, 0], [0, 0, 3]], dtype=int)
    kappa = calculate_cohens_kappa(cm)

    # Perfect agreement
    assert abs(kappa - 1.0) < 0.01


def test_cohens_kappa_empty():
    """Test kappa with empty confusion matrix."""
    cm = np.array([[0, 0], [0, 0]], dtype=int)
    kappa = calculate_cohens_kappa(cm)

    assert kappa == 0.0


def test_compare_labelers_matching(tmp_path):
    """Test comparing labelers with matching labels."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Create two identical label files
    block_ids = [f"TEST_0_{i}" for i in range(10)]
    class_ids = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    data1 = {
        "block_id": block_ids,
        "obs_id": ["TEST"] * 10,
        "class_id": class_ids,
        "status": ["labeled"] * 10,
    }

    table1 = pa.table(data1)
    path1 = tmp_path / "labels1.parquet"
    pq.write_table(table1, str(path1))

    data2 = {
        "block_id": block_ids,
        "obs_id": ["TEST"] * 10,
        "class_id": class_ids,
        "status": ["labeled"] * 10,
    }

    table2 = pa.table(data2)
    path2 = tmp_path / "labels2.parquet"
    pq.write_table(table2, str(path2))

    result = compare_labelers(path1, path2, "TEST")

    assert result["agreement_pct"] == 100.0
    assert abs(result["cohens_kappa"] - 1.0) < 0.01


def test_compare_labelers_disagreement(tmp_path):
    """Test comparing labelers with some disagreement."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    block_ids = [f"TEST_0_{i}" for i in range(10)]

    data1 = {
        "block_id": block_ids,
        "obs_id": ["TEST"] * 10,
        "class_id": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "status": ["labeled"] * 10,
    }

    table1 = pa.table(data1)
    path1 = tmp_path / "labels1.parquet"
    pq.write_table(table1, str(path1))

    # Different labels
    data2 = {
        "block_id": block_ids,
        "obs_id": ["TEST"] * 10,
        "class_id": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # Flipped
        "status": ["labeled"] * 10,
    }

    table2 = pa.table(data2)
    path2 = tmp_path / "labels2.parquet"
    pq.write_table(table2, str(path2))

    result = compare_labelers(path1, path2, "TEST")

    assert result["agreement_pct"] == 0.0  # No agreement
    assert result["cohens_kappa"] < 0.0  # Negative kappa for worse-than-random
