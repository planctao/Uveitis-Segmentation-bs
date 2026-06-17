from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate

