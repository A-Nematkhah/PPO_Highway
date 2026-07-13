"""
telemetry.py

Append-only structured timing/metrics log (JSON Lines) for the EUREKA search.
Complements the human-readable eureka_log.json archive — each row is one event
with duration_s / metrics fields suitable for plotting (Phase 5).
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

DEFAULT_METRICS_PATH = os.path.join("eureka", "eureka_metrics.jsonl")


class Telemetry:
    def __init__(self, path: str = DEFAULT_METRICS_PATH):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def record(self, event: str, **fields: Any) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    @contextmanager
    def timed(self, event: str, **fields: Any) -> Iterator[dict[str, Any]]:
        """
        Context manager that records duration_s on exit.

        Usage:
            with telemetry.timed("smoke_test", generation=0, candidate=0) as ctx:
                ctx["passed"] = True
        """
        ctx: dict[str, Any] = dict(fields)
        start = time.perf_counter()
        try:
            yield ctx
        finally:
            ctx["duration_s"] = round(time.perf_counter() - start, 4)
            self.record(event, **ctx)
