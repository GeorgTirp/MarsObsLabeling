"""Tests for config loading and validation."""

from pathlib import Path

import pytest

from marslabeler.config import load_config


def test_load_config_valid(tmp_config_dir):
    """Test loading a valid config file."""
    config_path = tmp_config_dir / "app.yaml"
    config = load_config(config_path)

    assert config.geometry.panel_size == 4096
    assert config.geometry.block_size == 512
    assert config.navigation.advance_mode == "next_unlabeled"
    assert config.autosave.every_n_labels == 25


def test_config_file_not_found():
    """Test error when config file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")


def test_geometry_validation_block_size_not_multiple_of_32(tmp_config_dir):
    """Test that block_size must be multiple of 32."""
    config_path = tmp_config_dir / "app.yaml"
    content = config_path.read_text()
    content = content.replace("block_size: 512", "block_size: 511")
    config_path.write_text(content)

    with pytest.raises(ValueError, match="must be a multiple of 32"):
        load_config(config_path)


def test_geometry_validation_block_size_divides_panel_size(tmp_config_dir):
    """Test that block_size must divide panel_size evenly."""
    config_path = tmp_config_dir / "app.yaml"
    content = config_path.read_text()
    # Use 256 (multiple of 32) but doesn't divide 4096 evenly
    content = content.replace("block_size: 512", "block_size: 256")
    content = content.replace("panel_size: 4096", "panel_size: 1500")
    config_path.write_text(content)

    with pytest.raises(ValueError, match="must divide"):
        load_config(config_path)


def test_path_resolution(tmp_config_dir):
    """Test that relative paths are resolved relative to config dir."""
    config_path = tmp_config_dir / "app.yaml"
    config = load_config(config_path)

    # Paths should be absolute after loading
    assert Path(config.paths.classes_file).is_absolute()
    assert Path(config.paths.labels_dir).is_absolute()
