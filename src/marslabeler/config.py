"""Configuration system: YAML -> typed dataclasses with validation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class PathsConfig:
    classes_file: str
    labels_dir: str

    def resolve(self, config_dir: Path) -> None:
        """Resolve relative paths relative to config directory."""
        if not Path(self.classes_file).is_absolute():
            self.classes_file = str(config_dir / self.classes_file)
        if not Path(self.labels_dir).is_absolute():
            self.labels_dir = str(config_dir / self.labels_dir)


@dataclass
class GeometryConfig:
    panel_size: int
    block_size: int

    def validate(self) -> None:
        """Ensure geometry constraints."""
        if self.block_size % 32 != 0:
            raise ValueError(
                f"block_size ({self.block_size}) must be a multiple of 32 (model stride)"
            )
        if self.panel_size % self.block_size != 0:
            raise ValueError(
                f"block_size ({self.block_size}) must divide panel_size ({self.panel_size})"
            )


@dataclass
class NavigationConfig:
    advance_mode: Literal["next_unlabeled", "next_sequential"]
    advance_on_edit: bool


@dataclass
class DisplayConfig:
    max_canvas_px: int
    stretch_percentiles: list[int]


@dataclass
class SkipConfig:
    nodata_skip_threshold: float
    variance_skip_threshold: float
    skip_low_variance: bool


@dataclass
class AutosaveConfig:
    every_n_labels: int
    every_seconds: int


@dataclass
class ExportConfig:
    full_res: bool


@dataclass
class AppConfig:
    paths: PathsConfig
    geometry: GeometryConfig
    navigation: NavigationConfig
    display: DisplayConfig
    skip: SkipConfig
    autosave: AutosaveConfig
    export: ExportConfig
    labeler: str | None

    def validate(self) -> None:
        """Run all validation checks."""
        self.geometry.validate()

    def to_dict(self) -> dict:
        """Convert to nested dict for Session."""
        from dataclasses import asdict
        return asdict(self)


def load_config(config_path: str | Path) -> AppConfig:
    """Load and validate app config from YAML."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    config = AppConfig(
        paths=PathsConfig(**data["paths"]),
        geometry=GeometryConfig(**data["geometry"]),
        navigation=NavigationConfig(**data["navigation"]),
        display=DisplayConfig(**data["display"]),
        skip=SkipConfig(**data["skip"]),
        autosave=AutosaveConfig(**data["autosave"]),
        export=ExportConfig(**data["export"]),
        labeler=data.get("labeler"),
    )

    config.paths.resolve(config_path.parent)
    config.validate()
    return config
