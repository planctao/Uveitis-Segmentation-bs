from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DEFAULT_MASK_EXTENSIONS = (".nii.gz", ".nii", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class FoldSummary:
    fold: str
    images: int
    masks: int
    pairs: int
    missing_masks: tuple[str, ...]
    missing_images: tuple[str, ...]


def matching_files(root: Path, extensions: Iterable[str]) -> list[Path]:
    allowed = tuple(sorted((ext.lower() for ext in extensions), key=len, reverse=True))
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_file() and path.name.lower().endswith(allowed))


def file_id(path: Path, extensions: Iterable[str]) -> str:
    """Return a comparable sample id after removing simple or compound suffixes."""
    name = path.name
    for extension in sorted(extensions, key=len, reverse=True):
        if name.lower().endswith(extension.lower()):
            return name[: -len(extension)]
    return path.stem


def summarize_fold(
    dataset_root: Path,
    fold: str,
    image_dir: str = "img",
    mask_dir: str = "mask",
    image_extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
    mask_extensions: Iterable[str] = DEFAULT_MASK_EXTENSIONS,
) -> FoldSummary:
    image_paths = matching_files(dataset_root / image_dir / fold, image_extensions)
    mask_paths = matching_files(dataset_root / mask_dir / fold, mask_extensions)

    image_stems = {file_id(path, image_extensions) for path in image_paths}
    mask_stems = {file_id(path, mask_extensions) for path in mask_paths}

    return FoldSummary(
        fold=fold,
        images=len(image_paths),
        masks=len(mask_paths),
        pairs=len(image_stems & mask_stems),
        missing_masks=tuple(sorted(image_stems - mask_stems)),
        missing_images=tuple(sorted(mask_stems - image_stems)),
    )


def summarize_dataset(
    dataset_root: Path,
    folds: Iterable[str],
    image_dir: str = "img",
    mask_dir: str = "mask",
    image_extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
    mask_extensions: Iterable[str] = DEFAULT_MASK_EXTENSIONS,
) -> list[FoldSummary]:
    return [
        summarize_fold(
            dataset_root,
            fold,
            image_dir=image_dir,
            mask_dir=mask_dir,
            image_extensions=image_extensions,
            mask_extensions=mask_extensions,
        )
        for fold in folds
    ]
