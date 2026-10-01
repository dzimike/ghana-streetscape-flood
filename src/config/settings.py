"""Load project config and environment variables."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def load_config(config_name: str = "accra_pilot") -> dict:
    """Load a YAML config from configs/."""
    path = ROOT / "configs" / f"{config_name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def get_api_key(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise EnvironmentError(f"Environment variable '{name}' is not set. Check your .env file.")
    return val


def resolve_path(cfg: dict, key: str) -> Path:
    """Return an absolute path from a config paths entry."""
    return ROOT / cfg["paths"][key]
