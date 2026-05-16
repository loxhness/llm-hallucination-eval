from pathlib import Path

_PKG = Path(__file__).resolve().parent


def package_dir() -> Path:
    return _PKG


def project_root() -> Path:
    """Repository root (parent of ``src/``)."""
    return _PKG.parent.parent


def default_dataset() -> Path:
    return _PKG / "data" / "mixed_v1.jsonl"
