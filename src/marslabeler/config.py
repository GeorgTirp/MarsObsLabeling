"""Configuration system: YAML -> typed dataclasses with validation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class PathsConfig:
    classes_file: str
    labels_dir: str
    predictions_dir: str = "./predictions"

    def resolve(self, config_dir: Path) -> None:
        """Resolve relative paths relative to config directory."""
        if not Path(self.classes_file).is_absolute():
            self.classes_file = str(config_dir / self.classes_file)
        if not Path(self.labels_dir).is_absolute():
            self.labels_dir = str(config_dir / self.labels_dir)
        if not Path(self.predictions_dir).is_absolute():
            self.predictions_dir = str(config_dir / self.predictions_dir)


@dataclass
class GeometryConfig:
    panel_size: int
    block_size: int

    def validate(self) -> None:
        """Ensure geometry constraints."""
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
class InferenceConfig:
    """Settings for `mars-inference` (model loading + windowing); unused by mars-label."""

    # Path to an AI4ExoMars checkout providing `vision_backend`. null -> auto-detect a
    # sibling `../AI4ExoMars` directory next to this repo, or an already-installed package.
    ai4exomars_path: str | None = None
    device: str = "auto"  # auto | cpu | cuda | mps
    batch_size: int = 4
    # Context crop size (context-branch models only) = context_multiplier * local window size.
    context_multiplier: int = 4
    # Blocks with more than this fraction nodata (reusing the same preprocessing pass
    # mars-label's skip.nodata_skip_threshold uses) are never sent through the model --
    # retired as nodata instead. Independent of skip.nodata_skip_threshold: a
    # majority-nodata block is still meaningfully labelable by a human, but a model
    # prediction on one is closer to noise, so this defaults stricter.
    nodata_skip_threshold: float = 0.33


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
    inference: InferenceConfig

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
        inference=InferenceConfig(**data.get("inference", {})),
    )

    config.paths.resolve(config_path.parent)
    config.validate()
    return config
