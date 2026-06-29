"""Tests for class scheme loading and validation."""

from pathlib import Path

import pytest

from marslabeler.classes import load_classes


def test_load_classes_valid(tmp_config_dir):
    """Test loading a valid classes file."""
    classes_path = tmp_config_dir / "classes.yaml"
    scheme = load_classes(classes_path)

    assert len(scheme.classes) == 2
    assert 0 in scheme.classes
    assert 1 in scheme.classes
    assert scheme.classes[0].name == "Class A"
    assert scheme.classes[0].hotkey == "q"
    assert scheme.abstain.id == -1
    assert scheme.nodata.id == -2


def test_classes_file_not_found():
    """Test error when classes file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_classes("nonexistent_classes.yaml")


def test_duplicate_hotkeys(tmp_config_dir):
    """Test that duplicate hotkeys are rejected."""
    classes_path = tmp_config_dir / "classes.yaml"
    content = classes_path.read_text()
    # Change Class B's hotkey to match Class A's
    content = content.replace('hotkey: "w"', 'hotkey: "q"')
    classes_path.write_text(content)

    with pytest.raises(ValueError, match="Hotkey.*used by both"):
        load_classes(classes_path)


def test_invalid_color_hex(tmp_config_dir):
    """Test that invalid hex colors are rejected."""
    classes_path = tmp_config_dir / "classes.yaml"
    content = classes_path.read_text()
    # Replace a valid hex with invalid
    content = content.replace('color: "#4C72B0"', 'color: "#GGGGGG"')
    classes_path.write_text(content)

    with pytest.raises(ValueError, match="not valid hex"):
        load_classes(classes_path)


def test_color_wrong_format(tmp_config_dir):
    """Test that colors without # prefix are rejected."""
    classes_path = tmp_config_dir / "classes.yaml"
    content = classes_path.read_text()
    content = content.replace('color: "#4C72B0"', 'color: "4C72B0"')
    classes_path.write_text(content)

    with pytest.raises(ValueError, match="must be #RRGGBB"):
        load_classes(classes_path)


def test_hotkey_to_id_mapping(tmp_config_dir):
    """Test hotkey->id mapping is correct."""
    classes_path = tmp_config_dir / "classes.yaml"
    scheme = load_classes(classes_path)

    assert scheme.hotkey_to_id["q"] == 0
    assert scheme.hotkey_to_id["w"] == 1
    assert scheme.hotkey_to_id["space"] == -1


def test_id_to_name_mapping(tmp_config_dir):
    """Test id->name mapping is correct."""
    classes_path = tmp_config_dir / "classes.yaml"
    scheme = load_classes(classes_path)

    assert scheme.id_to_name[0] == "Class A"
    assert scheme.id_to_name[1] == "Class B"
    assert scheme.id_to_name[-1] == "Abstain"
    assert scheme.id_to_name[-2] == "No data"


def test_get_name(tmp_config_dir):
    """Test get_name method."""
    classes_path = tmp_config_dir / "classes.yaml"
    scheme = load_classes(classes_path)

    assert scheme.get_name(0) == "Class A"
    assert scheme.get_name(-1) == "Abstain"


def test_get_color(tmp_config_dir):
    """Test get_color method."""
    classes_path = tmp_config_dir / "classes.yaml"
    scheme = load_classes(classes_path)

    assert scheme.get_color(0) == "#4C72B0"
    assert scheme.get_color(-1) == "#000000"


def test_reserved_ids_not_usable_by_classes(tmp_config_dir):
    """Test that reserved IDs (-1, -2) cannot be used by user classes."""
    classes_path = tmp_config_dir / "classes.yaml"
    content = classes_path.read_text()
    # Add a class with reserved ID
    content = content.replace('classes:', 'classes:\n  - { id: -1,  name: "BadClass", color: "#FF0000", hotkey: "x" }')
    classes_path.write_text(content)

    with pytest.raises(ValueError, match="conflict with reserved"):
        load_classes(classes_path)
