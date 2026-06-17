from pathlib import Path

from bs.config import get_dataset_root, load_config


def test_default_config_loads() -> None:
    config = load_config()

    assert config["project"]["name"] == "uveitis_segmentation"
    assert config["runtime"]["device"] == "cuda"


def test_dataset_root_is_project_relative() -> None:
    config = load_config()
    dataset_root = get_dataset_root(config)

    assert isinstance(dataset_root, Path)
    assert dataset_root.name == "split_dataorigin"

