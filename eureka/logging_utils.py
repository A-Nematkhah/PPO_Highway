"""
logging_utils.py

Structured logging for the EUREKA pipeline. Replaces ad-hoc print(..., flush=True)
with the stdlib logging module and an optional JSON line formatter (set env
EUREKA_LOG_JSON=1) for downstream tooling (Phase 5 dashboards).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

_STD_RECORD_KEYS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


class JsonLogFormatter(logging.Formatter):
    """One JSON object per log line — easy to ingest with jq / Loki / etc."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STD_RECORD_KEYS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in _STD_RECORD_KEYS and not k.startswith("_")
        }
        if extras:
            base = f"{base} | {extras}"
        return base


_configured = False


def setup_logging(level: int | None = None) -> None:
    """Configure root eureka logger once (idempotent)."""
    global _configured
    if _configured:
        return

    log_level = level if level is not None else getattr(
        logging, os.environ.get("EUREKA_LOG_LEVEL", "INFO").upper(), logging.INFO
    )
    use_json = os.environ.get("EUREKA_LOG_JSON", "0") == "1"

    handler = logging.StreamHandler(sys.stdout)
    if use_json:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(_TextFormatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))

    root = logging.getLogger("eureka")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    if name.startswith("eureka."):
        return logging.getLogger(name)
    return logging.getLogger(f"eureka.{name}")
