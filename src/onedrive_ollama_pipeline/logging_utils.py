"""Centralised logging configuration for the pipeline and admin API."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_DEFAULT_MAX_BYTES = 2 * 1024 * 1024  # 2 MB
_DEFAULT_BACKUPS = 3

_FILE_HANDLER: Optional[RotatingFileHandler] = None
_CURRENT_LEVEL: int = logging.INFO


def configure_logging(*, log_path: Path, console_level: int = logging.INFO, file_level: Optional[int] = None) -> None:
    """Configure console + rotating-file logging."""
    global _FILE_HANDLER, _CURRENT_LEVEL
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler (stream)
    _ensure_console_handler(root, console_level)

    # File handler
    if _FILE_HANDLER is None:
        handler = RotatingFileHandler(pathlib_path(log_path), maxBytes=_DEFAULT_MAX_BYTES, backupCount=_DEFAULT_BACKUPS)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler.setLevel(file_level or console_level)
        root.addHandler(handler)
        _FILE_HANDLER = handler
    else:
        if file_level:
            _FILE_HANDLER.setLevel(file_level)
        _FILE_HANDLER.baseFilename = pathlib_path(log_path)

    _CURRENT_LEVEL = console_level


def set_runtime_log_level(level_name: str) -> None:
    """Update the root logger and file handler level in-place."""
    global _CURRENT_LEVEL
    level = logging.getLevelName(level_name.upper())
    if isinstance(level, str):  # invalid level returns its own name back
        raise ValueError(f"Unknown log level: {level_name}")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in root.handlers:
        handler.setLevel(level)
    _CURRENT_LEVEL = level


def get_runtime_log_level() -> str:
    return logging.getLevelName(_CURRENT_LEVEL)


def _ensure_console_handler(root: logging.Logger, level: int) -> None:
    existing = None
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler):
            existing = handler
            break
    if existing is None:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        console.setLevel(level)
        root.addHandler(console)
    else:
        existing.setLevel(level)
        existing.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))


def pathlib_path(path: Path) -> str:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.as_posix()
