"""Uniform step logging: entered / emitted <Event> because <reason>."""
from __future__ import annotations

import logging
from typing import TypeVar

E = TypeVar("E")


def log_enter(logger: logging.Logger, step: str, event) -> None:
    logger.info("[%s] entered with %s", step, type(event).__name__)


def emit(logger: logging.Logger, step: str, event: E, reason: str) -> E:
    logger.info("[%s] exit -> %s (%s)", step, type(event).__name__, reason)
    return event
