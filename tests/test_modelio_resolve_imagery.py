"""Tests for modelio.py's checkpoint -> training-imagery-path resolution
(click-to-navigate support for the Class Summary window's Neural-PCA
thumbnails -- see resolve_training_imagery_path's docstring)."""

import json

import pytest
import torch

from marslabeler.inference.modelio import (
    _resolve_ai4exomars_root,
    _resolve_relative_to,
    resolve_training_imagery_path,
)


# --------------------------------------------------------------------------
# _resolve_relative_to
# --------------------------------------------------------------------------
def test_resolve_relative_to_absolute_path_passthrough(tmp_path):
    abs_path = tmp_path / "x" / "y.tif"
    assert _resolve_relative_to(str(abs_path), tmp_path / "unrelated") == abs_path


def test_resolve_relative_to_joins_with_root(tmp_path):
    assert _resolve_relative_to("data/imagery.tif", tmp_path) == tmp_path / "data" / "imagery.tif"


def test_resolve_relative_to_no_root_returns_relative_as_is():
    from pathlib import Path

    assert _resolve_relative_to("data/imagery.tif", None) == Path("data/imagery.tif")


# --------------------------------------------------------------------------
# _resolve_ai4exomars_root
# --------------------------------------------------------------------------
def test_resolve_ai4exomars_root_prefers_explicit_path_with_vision_backend(tmp_path):
    root = tmp_path / "my_checkout"
    (root / "vision_backend").mkdir(parents=True)
    assert _resolve_ai4exomars_root(str(root)) == root


def test_resolve_ai4exomars_root_falls_through_when_explicit_path_invalid(tmp_path):
    bad_root = tmp_path / "not_a_checkout"
    bad_root.mkdir()
    # No vision_backend/ under bad_root -- must fall through to a later
    # candidate (vision_backend's own install location, in this environment)
    # rather than returning bad_root or None.
    result = _resolve_ai4exomars_root(str(bad_root))
    assert result != bad_root


def test_resolve_ai4exomars_root_finds_something_when_vision_backend_importable():
    """This test environment has vision_backend installed (editable), so the
    no-explicit-path case must resolve to a real directory containing it."""
    result = _resolve_ai4exomars_root(None)
    assert result is not None
    assert (result / "vision_backend").is_dir()


# --------------------------------------------------------------------------
# resolve_training_imagery_path
# --------------------------------------------------------------------------
def _save_checkpoint(path, config):
    torch.save({"model_state": {}, "config": config}, path)


def test_resolve_training_imagery_path_happy_path(tmp_path):
    imagery_file = tmp_path / "drg_on_label_grid.tif"
    imagery_file.write_bytes(b"fake tif bytes")

    loader_config_file = tmp_path / "seg_loader.json"
    loader_config_file.write_text(json.dumps({"imagery_path": str(imagery_file)}))

    checkpoint_path = tmp_path / "ckpt.pt"
    _save_checkpoint(
        checkpoint_path,
        {"data": {"loader_config_path": str(loader_config_file)}},
    )

    result = resolve_training_imagery_path(checkpoint_path)
    assert result == imagery_file


def test_resolve_training_imagery_path_relative_paths_resolved_against_root(tmp_path):
    root = tmp_path / "ai4exomars_checkout"
    (root / "vision_backend").mkdir(parents=True)
    derived_dir = root / "data" / "derived"
    derived_dir.mkdir(parents=True)
    (derived_dir / "mosaic.tif").write_bytes(b"fake")
    loader_config_file = derived_dir / "loader.json"
    loader_config_file.write_text(json.dumps({"imagery_path": "data/derived/mosaic.tif"}))

    checkpoint_path = tmp_path / "ckpt.pt"
    _save_checkpoint(
        checkpoint_path,
        {"data": {"loader_config_path": "data/derived/loader.json"}},
    )

    result = resolve_training_imagery_path(checkpoint_path, ai4exomars_path=str(root))
    assert result == derived_dir / "mosaic.tif"


def test_resolve_training_imagery_path_missing_loader_config_path_key_returns_none(tmp_path):
    checkpoint_path = tmp_path / "ckpt.pt"
    _save_checkpoint(checkpoint_path, {"data": {}})
    assert resolve_training_imagery_path(checkpoint_path) is None


def test_resolve_training_imagery_path_loader_config_file_missing_returns_none(tmp_path):
    checkpoint_path = tmp_path / "ckpt.pt"
    _save_checkpoint(
        checkpoint_path,
        {"data": {"loader_config_path": str(tmp_path / "does_not_exist.json")}},
    )
    assert resolve_training_imagery_path(checkpoint_path) is None


def test_resolve_training_imagery_path_loader_config_missing_imagery_path_key_returns_none(tmp_path):
    loader_config_file = tmp_path / "loader.json"
    loader_config_file.write_text(json.dumps({"manifest_path": "whatever.csv"}))
    checkpoint_path = tmp_path / "ckpt.pt"
    _save_checkpoint(
        checkpoint_path,
        {"data": {"loader_config_path": str(loader_config_file)}},
    )
    assert resolve_training_imagery_path(checkpoint_path) is None


def test_resolve_training_imagery_path_imagery_file_missing_returns_none(tmp_path):
    loader_config_file = tmp_path / "loader.json"
    loader_config_file.write_text(
        json.dumps({"imagery_path": str(tmp_path / "nonexistent_mosaic.tif")})
    )
    checkpoint_path = tmp_path / "ckpt.pt"
    _save_checkpoint(
        checkpoint_path,
        {"data": {"loader_config_path": str(loader_config_file)}},
    )
    assert resolve_training_imagery_path(checkpoint_path) is None


def test_resolve_training_imagery_path_nonexistent_checkpoint_returns_none(tmp_path):
    assert resolve_training_imagery_path(tmp_path / "no_such_checkpoint.pt") is None


def test_resolve_training_imagery_path_malformed_loader_config_json_returns_none(tmp_path):
    loader_config_file = tmp_path / "loader.json"
    loader_config_file.write_text("{not valid json")
    checkpoint_path = tmp_path / "ckpt.pt"
    _save_checkpoint(
        checkpoint_path,
        {"data": {"loader_config_path": str(loader_config_file)}},
    )
    assert resolve_training_imagery_path(checkpoint_path) is None
