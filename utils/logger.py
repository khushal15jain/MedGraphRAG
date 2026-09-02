"""Centralized logging configuration for MedGraphRAG.

Every module in the project should obtain its logger via ``get_logger(__name__)``
rather than instantiating ``loguru`` directly. This guarantees a single,
consistently formatted sink (console + rotating file) across the whole
pipeline, which matters for debugging long-running research experiments.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _loguru_logger

_CONFIGURED = False


def _configure_root_logger(log_dir: str = "outputs/logs", level: str = "INFO") -> None:
    """Configure the global loguru sink exactly once per process.

    Args:
        log_dir: Directory where rotating log files are written.
        level: Minimum log level to emit (e.g. "DEBUG", "INFO", "WARNING").
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    _loguru_logger.remove()  # drop the default stderr sink to control format explicitly

    _loguru_logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    _loguru_logger.add(
        log_path / "medgraphrag_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    _CONFIGURED = True


def get_logger(name: str, log_dir: str = "outputs/logs", level: str = "INFO"):
    """Return a bound loguru logger tagged with the calling module's name.

    Args:
        name: Typically ``__name__`` of the calling module.
        log_dir: Directory for rotating log files (only used on first call).
        level: Console log level (only used on first call).

    Returns:
        A loguru logger instance bound with the module name for context.
    """
    _configure_root_logger(log_dir=log_dir, level=level)
    return _loguru_logger.bind(module=name)
