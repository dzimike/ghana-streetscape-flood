"""Centralised logging setup using loguru."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logger(log_dir: Path | None = None, level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(log_dir / "pipeline.log", level="DEBUG", rotation="10 MB", retention="30 days")
