"""Logging setup. Everything in this service logs under the ``stt.*`` namespace.

Mirrors my-meeting-notes/app/logging_config.py. No structlog/loguru/OTel -- stdlib
``logging`` + ``dictConfig`` is the house convention and there is nothing here that
needs more than a namespaced, non-propagating root logger.

Formatter is plain text; the code uses ``key=value`` tokens in messages for
greppability (see CLAUDE.md's canonical lifecycle-line examples) rather than
structured fields, since dictConfig gives us no structured sink to put them in.
"""

from __future__ import annotations

import logging
from logging.config import dictConfig


def configure_logging(level: str = "INFO") -> None:
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                }
            },
            "loggers": {
                "stt": {"handlers": ["console"], "level": level, "propagate": False},
            },
            "root": {"handlers": ["console"], "level": "WARNING"},
        }
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"stt.{name}")
