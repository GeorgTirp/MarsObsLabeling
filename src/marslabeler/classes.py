"""Terrain class scheme: load, validate, and manage class definitions."""

from dataclasses import dataclass
from pathlib import Path

import yaml

# Reserved class IDs that must not be used by user-assignable classes
RESERVED_IDS = {-1, -2}
# Navigation keys that must not collide with class hotkeys (not including space, which is abstain)
RESERVED_HOTKEYS = {"?"}  # Help overlay


@dataclass
class TerrainClass:
    id: int
    name: str
    color: str
    hotkey: str | None = None

    def validate_color(self) -> None:
        """Ensure color is valid hex."""
        if not self.color.startswith("#") or len(self.color) != 7:
            raise ValueError(
                f'Color "{self.color}" for class "{self.name}" must be #RRGGBB hex'
            )
        try:
            int(self.color[1:], 16)
        except ValueError:
            raise ValueError(
                f'Color "{self.color}" for class "{self.name}" is not valid hex'
            )


@dataclass
class ClassScheme:
    """Manages terrain classes, validation, and hotkey mapping."""

    classes: dict[int, TerrainClass]
    abstain: TerrainClass
    nodata: TerrainClass
    id_to_name: dict[int, str]
    hotkey_to_id: dict[str, int]

    def validate(self) -> None:
        """Validate class scheme for consistency."""
        # Check reserved IDs
        user_ids = set(self.classes.keys())
        if user_ids & RESERVED_IDS:
            conflict = user_ids & RESERVED_IDS
            raise ValueError(
                f"User class IDs {conflict} conflict with reserved IDs {RESERVED_IDS}"
            )

        # Check hotkey uniqueness
        hotkeys_used = {}
        for cls in self.classes.values():
            if cls.hotkey and cls.hotkey in hotkeys_used:
                raise ValueError(
                    f'Hotkey "{cls.hotkey}" used by both "{cls.name}" and '
                    f'"{hotkeys_used[cls.hotkey]}"'
                )
            if cls.hotkey:
                hotkeys_used[cls.hotkey] = cls.name

        # Check hotkey collisions with reserved keys
        if self.abstain.hotkey and self.abstain.hotkey in RESERVED_HOTKEYS:
            raise ValueError(
                f'Abstain hotkey "{self.abstain.hotkey}" collides with reserved keys'
            )
        for cls in self.classes.values():
            if cls.hotkey and cls.hotkey in RESERVED_HOTKEYS:
                raise ValueError(
                    f'Class "{cls.name}" hotkey "{cls.hotkey}" collides with reserved keys'
                )

        # Validate colors
        for cls in self.classes.values():
            cls.validate_color()
        self.abstain.validate_color()
        self.nodata.validate_color()

    def get_name(self, class_id: int) -> str:
        """Get class name by ID."""
        if class_id in self.classes:
            return self.classes[class_id].name
        if class_id == self.abstain.id:
            return self.abstain.name
        if class_id == self.nodata.id:
            return self.nodata.name
        raise ValueError(f"Unknown class ID: {class_id}")

    def get_color(self, class_id: int) -> str:
        """Get class color by ID."""
        if class_id in self.classes:
            return self.classes[class_id].color
        if class_id == self.abstain.id:
            return self.abstain.color
        if class_id == self.nodata.id:
            return self.nodata.color
        raise ValueError(f"Unknown class ID: {class_id}")


def load_classes(yaml_path: str | Path) -> ClassScheme:
    """Load and validate class scheme from YAML."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Classes file not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Parse user classes
    classes: dict[int, TerrainClass] = {}
    for item in data.get("classes", []):
        cls = TerrainClass(
            id=item["id"],
            name=item["name"],
            color=item.get("color", "#808080"),
            hotkey=item.get("hotkey"),
        )
        classes[cls.id] = cls

    # Parse reserved classes
    abstain_data = data.get("abstain", {})
    abstain = TerrainClass(
        id=abstain_data.get("id", -1),
        name=abstain_data.get("name", "Abstain"),
        color=abstain_data.get("color", "#000000"),
        hotkey=abstain_data.get("hotkey", "space"),
    )

    nodata_data = data.get("nodata", {})
    nodata = TerrainClass(
        id=nodata_data.get("id", -2),
        name=nodata_data.get("name", "No data"),
        color=nodata_data.get("color", "#222222"),
        hotkey=None,
    )

    # Build lookup tables
    id_to_name = {cls.id: cls.name for cls in classes.values()}
    id_to_name[abstain.id] = abstain.name
    id_to_name[nodata.id] = nodata.name

    hotkey_to_id = {}
    for cls in classes.values():
        if cls.hotkey:
            hotkey_to_id[cls.hotkey] = cls.id
    if abstain.hotkey:
        hotkey_to_id[abstain.hotkey] = abstain.id

    scheme = ClassScheme(
        classes=classes,
        abstain=abstain,
        nodata=nodata,
        id_to_name=id_to_name,
        hotkey_to_id=hotkey_to_id,
    )
    scheme.validate()
    return scheme
