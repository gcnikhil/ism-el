"""Configuration loading helpers for ContextTier.

All paths in ``config.yaml`` are resolved relative to the project root so the
code works on Windows, macOS and Linux.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


def project_root() -> Path:
    """Return the absolute project root directory."""
    return PROJECT_ROOT


def load_config(path: str | Path | None = None) -> dict:
    """Load ``config.yaml`` (or ``path``) as a dictionary.

    Args:
        path: Optional config path. Relative paths are resolved against the
            project root.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML root is not a mapping.
    """
    cfg_path = PROJECT_ROOT / "config.yaml" if path is None else Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {cfg_path}")
    return config


def resolve_path(value: str | Path) -> Path:
    """Resolve ``value`` to an absolute path relative to the project root."""
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
