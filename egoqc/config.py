# Reads config.yaml once, so callers pass the settings on instead of every check opening the file
from pathlib import Path

import yaml


def load_config(path: Path = Path("config.yaml")) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


