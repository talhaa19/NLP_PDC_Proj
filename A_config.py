"""Central configuration loader.

All modules import ``get_config()`` instead of hardcoding parameters.
The config file lives at ``config.yaml`` next to this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    """Load and cache the YAML config from disk.

    Returns the full config dict. Subsequent calls return the cached copy
    without re-reading the file.
    """
    global _config
    if _config is None:
        with open(_CONFIG_PATH, encoding="utf-8") as fh:
            _config = yaml.safe_load(fh)
    return _config


def reload_config() -> dict[str, Any]:
    """Force a fresh read of config.yaml and return the new dict.

    Useful in tests that temporarily patch config values.
    """
    global _config
    _config = None
    return get_config()


def _print_config(cfg: dict, indent: int = 0) -> None:
    """Recursively pretty-print a nested config dict."""
    prefix = "  " * indent
    for key, value in cfg.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            _print_config(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")


if __name__ == "__main__":
    cfg = get_config()
    print("=" * 60)
    print("  PROJECT CONFIGURATION  (config.yaml)")
    print("=" * 60)
    _print_config(cfg)
    print("=" * 60)
    print(f"Config loaded from: {_CONFIG_PATH}")
    print("All modules read parameters from this file via get_config().")
