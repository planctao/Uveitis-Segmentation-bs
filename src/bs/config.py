from pathlib import Path
from typing import Any

import yaml

from bs.paths import project_path


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load a YAML config file from the project tree."""
    config_path = project_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config["_config_path"] = str(config_path)
    return config


def get_dataset_root(config: dict[str, Any]) -> Path:
    return project_path(config["data"]["root"])

