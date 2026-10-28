"""Loading of YAML configuration and project paths."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"

# Real and synthetic data are stored in separate trees so they can never be mixed.
DATA_MODES = ("REAL", "SYNTHETIC")


def load_yaml(name: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    with open(config_dir / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def processed_dir(mode: str, data_dir: Path = DATA_DIR) -> Path:
    if mode not in DATA_MODES:
        raise ValueError(f"data mode must be one of {DATA_MODES}, got {mode!r}")
    path = data_dir / "processed" / mode.lower()
    path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=1)
def engine_fingerprint() -> str:
    """Hash of the calculation engine (aurum/*.py outside the UI) and all config files.

    Passed into every cached dashboard computation so a change to formulas, rules or assumptions
    can never be answered from an older cached result."""
    h = hashlib.sha256()
    files = sorted((ROOT / "aurum").glob("*.py")) + sorted(CONFIG_DIR.glob("*.yaml"))
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]
