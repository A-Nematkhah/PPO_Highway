"""
logging_utils.py

Structured logging for the EUREKA pipeline. Replaces ad-hoc print(..., flush=True)
with the stdlib logging module. Console output is always human-readable; set env
``EUREKA_LOG_JSON=1`` to also append JSON lines to ``eureka/eureka_run.jsonl``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

_STD_RECORD_KEYS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())
_SKIP_EXTRA_KEYS = frozenset({"event", "message", "asctime", "msg", "args"})


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


def _extract_extras(record: logging.LogRecord) -> dict:
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _STD_RECORD_KEYS
        and key not in _SKIP_EXTRA_KEYS
        and not key.startswith("_")
    }


def _short_logger(name: str) -> str:
    if name.startswith("eureka."):
        name = name[len("eureka.") :]
    aliases = {
        "llm_reward_designer": "llm",
        "train_candidate": "train",
        "evaluate_candidate": "eval",
        "__main__": "loop",
    }
    return aliases.get(name, name)


def _format_steps(steps: int) -> str:
    if steps >= 1_000_000:
        return f"{steps / 1_000_000:g}M"
    if steps >= 1_000:
        return f"{steps // 1_000}k"
    return str(steps)


def _format_duration(seconds) -> str:
    value = float(seconds)
    if value >= 60:
        minutes = int(value // 60)
        secs = value - minutes * 60
        return f"{minutes}m{secs:.0f}s"
    if value >= 10:
        return f"{value:.0f}s"
    return f"{value:.1f}s"


def _format_pct(value) -> str:
    return f"{float(value):.1f}%"


def _fmt_num(value, width: int, precision: int = 1) -> str:
    if value != value:  # NaN
        return "n/a".rjust(width)
    text = f"{float(value):.{precision}f}"
    return text.rjust(width)


class TrainProgressTable:
    """Print compact aligned rows for candidate PPO training progress."""

    def __init__(self, candidate: str, total_updates: int) -> None:
        self.candidate = candidate
        self.total_updates = total_updates
        self._header_printed = False

    def _print_header(self) -> None:
        if self._header_printed:
            return
        self._header_printed = True
        lines = [
            "",
            f"  [{self.candidate}] training",
            "  upd    step  fps  return  crash  speed  otk",
            "  ---- ------ ---- ------- ------ ------ ----",
        ]
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

    def add_row(
        self,
        *,
        update: int,
        global_step: int,
        fps: int,
        mean_return: float,
        crash_rate_pct: float,
        mean_speed: float,
        mean_overtakes: float,
    ) -> None:
        self._print_header()
        crash_text = f"{crash_rate_pct:5.0f}%" if crash_rate_pct == crash_rate_pct else "  n/a"
        row = (
            f"  {update:3d}/{self.total_updates:<3} "
            f"{global_step:>6} "
            f"{fps:>4} "
            f"{_fmt_num(mean_return, 7, 1)} "
            f"{crash_text:>6} "
            f"{_fmt_num(mean_speed, 6, 1)} "
            f"{_fmt_num(mean_overtakes, 4, 1)}"
        )
        sys.stdout.write(row + "\n")
        sys.stdout.flush()


def print_final_archive_table(archive: list[dict], representative_id: str | None) -> None:
    """
    Print a ranked, human-readable summary of the FULL final Pareto archive
    at the end of a run - not just the single "winner".

    Previously the only end-of-run console output was the duration on the
    "EUREKA run finished" line; every candidate's actual crash/speed/overtake
    trade-off (and even which one was picked as the representative) only
    existed in eureka_log.json, requiring the user to open and manually
    reconstruct it from JSON to answer "so which one won?". This prints the
    same information the archive already carries, sorted by the diagnostic
    legacy_fitness (see eureka/fitness.py for why that score is diagnostic-
    only, not a substitute for the Pareto trade-off itself - all three raw
    metrics are shown side by side specifically so a human isn't limited to
    that one scalar).

    Safe to call with an empty archive (prints a short "no archive" notice
    instead of an empty table).
    """
    if not archive:
        sys.stdout.write("\n  Final Pareto archive: empty (no candidate survived the whole run)\n")
        sys.stdout.flush()
        return

    ranked = sorted(archive, key=lambda c: c.get("legacy_fitness", c.get("fitness", 0.0)), reverse=True)

    lines = [
        "",
        f"  Final Pareto archive ({len(ranked)} candidates)",
        "  rank  candidate                  crash   speed   otk    fitness",
        "  ----  ------------------------- ------  ------  -----  -------",
    ]
    for candidate in ranked:
        module_path = str(candidate.get("module_path", "?"))
        short_name = module_path.rsplit(".", 1)[-1]
        is_rep = candidate.get("candidate_id") == representative_id
        label = f"{short_name} (WINNER)" if is_rep else short_name
        metrics = candidate.get("metrics", {})
        fitness = candidate.get("legacy_fitness", candidate.get("fitness", float("nan")))
        rank = candidate.get("pareto_rank", "?")
        lines.append(
            f"   {str(rank):>3}  {label:<27}"
            f"{_fmt_num(float(metrics.get('crash_rate', float('nan'))) * 100, 5, 1)}% "
            f"{_fmt_num(metrics.get('mean_speed', float('nan')), 6, 1)} "
            f"{_fmt_num(metrics.get('mean_overtakes', float('nan')), 5, 2)}  "
            f"{_fmt_num(fitness, 6, 3)}"
        )
    lines.append("")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def print_generation_table(
    generation: int,
    front_candidates: list[dict],
    winner_module_path: str | None,
    reason: str,
) -> None:
    """
    Prints the "at the end of every generation" console table: the
    generation's Pareto front (rank/candidate/crash/speed/overtakes), then
    the generation winner and why it was picked. Purely a console
    rendering of data loop.py already has by the time a generation
    finishes (front_candidates from eureka.objectives.annotate_population,
    the winner/representative id, and the same reason logic report_html.py
    uses) - no new selection or ranking logic here.
    """
    banner = "=" * 68
    lines = [banner, f"Generation {generation}", "Pareto Front", banner]

    if not front_candidates:
        lines.append("  (no candidates on the Pareto front this generation)")
    else:
        lines.append("  rank  candidate                  crash   speed   otk")
        lines.append("  ----  ------------------------- ------  ------  -----")
        for candidate in sorted(front_candidates, key=lambda c: c.get("crowding_distance", 0.0), reverse=True):
            module_path = str(candidate.get("module_path", "?"))
            short_name = module_path.rsplit(".", 1)[-1]
            is_winner = module_path == winner_module_path
            label = f"{short_name} (WINNER)" if is_winner else short_name
            metrics = candidate.get("metrics", {})
            rank = candidate.get("pareto_rank", "?")
            lines.append(
                f"   {str(rank):>3}  {label:<27}"
                f"{_fmt_num(float(metrics.get('crash_rate', float('nan'))) * 100, 5, 1)}% "
                f"{_fmt_num(metrics.get('mean_speed', float('nan')), 6, 1)} "
                f"{_fmt_num(metrics.get('mean_overtakes', float('nan')), 5, 2)}"
            )

    lines.append(banner)
    winner_short = winner_module_path.rsplit(".", 1)[-1] if winner_module_path else "(none)"
    lines.append(f"Generation Winner: {winner_short}")
    lines.append(f"Reason: {reason}")
    lines.append(banner)
    lines.append("")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def print_final_results_banner(
    archive: list[dict], representative_id: str | None, summary: str,
) -> None:
    """
    Prints the end-of-experiment "FINAL RESULTS" banner (Step 6), then
    delegates the actual archive table to print_final_archive_table so
    there's exactly one implementation of "how to render a Pareto
    archive as text" rather than two slightly-different copies.
    """
    banner = "=" * 68
    sys.stdout.write(f"\n{banner}\nFINAL RESULTS\nGlobal Pareto Archive\n{banner}\n")
    sys.stdout.flush()
    print_final_archive_table(archive, representative_id)
    winner = next((c for c in archive if c.get("candidate_id") == representative_id), None)
    winner_name = str(winner.get("module_path", "?")).rsplit(".", 1)[-1] if winner else "(none)"
    sys.stdout.write(f"Winner: {winner_name}\nSummary: {summary}\n{banner}\n\n")
    sys.stdout.flush()


def _format_context(record: logging.LogRecord) -> str:
    event = getattr(record, "event", None)
    data = _extract_extras(record)

    if event == "run_start":
        return (
            f"\n{' ' * 20}"
            f"{data.get('generations')} gen x {data.get('candidates_per_gen')} candidates "
            f"x {_format_steps(int(data.get('train_steps_per_candidate', 0)))} steps | "
            f"mode={data.get('multi_objective_mode')} | {data.get('llm_model')}"
        )

    if event == "generation_start":
        return f"  (gen {data.get('generation')})"

    if event == "llm_request":
        return f"  (gen {data.get('generation')}, k={data.get('k')})"

    if event == "llm_call_start":
        role = data.get("target_role")
        role_text = f", role={role}" if role else ""
        return (
            f"  [gen{data.get('generation')} "
            f"#{data.get('index')}/{data.get('k')}{role_text}]"
        )

    if event == "llm_call_success":
        return f"  ({data.get('lines')} lines, {_format_duration(data.get('duration_s', 0))})"

    if event == "llm_call_no_code":
        preview = data.get("preview", "")
        preview_text = f" — {preview}" if preview else ""
        return f"  ({_format_duration(data.get('duration_s', 0))}{preview_text})"

    if event == "llm_call_error":
        return f"  ({_format_duration(data.get('duration_s', 0))}: {data.get('error')})"

    if event in {"candidate_start", "candidate_saved", "candidate_rejected"}:
        parts = [f"gen{data.get('generation')} cand{data.get('candidate')}"]
        if event == "candidate_saved":
            parts.append(os.path.basename(str(data.get("path", ""))))
        if event == "candidate_rejected":
            parts.append(f"stage={data.get('stage')}")
            parts.append(str(data.get("reason", "")))
        return f"  ({', '.join(parts)})"

    if event == "train_start":
        return (
            f"  ({data.get('candidate_module')}, {data.get('updates')} updates, "
            f"{data.get('envs')} envs, seed={data.get('seed')})"
        )

    if event == "train_update":
        return ""

    if event == "train_complete":
        return (
            f"  ({data.get('candidate_module')}, "
            f"{_format_duration(data.get('duration_s', 0))}, {data.get('checkpoint')})"
        )

    if event == "eval_summary":
        return (
            f"  (crash={float(data.get('crash_rate', 0)):.2f} "
            f"speed={float(data.get('mean_speed', 0)):.1f} "
            f"overtakes={float(data.get('mean_overtakes', 0)):.2f})"
        )

    if event == "candidate_complete":
        return (
            f"  (gen{data.get('generation')} cand{data.get('candidate')}, "
            f"fitness={float(data.get('legacy_fitness', 0)):.3f}, "
            f"crash={float(data.get('crash_rate', 0)):.2f}, "
            f"speed={float(data.get('mean_speed', 0)):.1f}, "
            f"{_format_duration(data.get('duration_s', 0))})"
        )

    if event == "generation_selection":
        rep = data.get("representative")
        rep_text = f", rep={os.path.basename(str(rep))}" if rep else ""
        return (
            f"  (gen{data.get('generation')}, front={data.get('pareto_front_size')}, "
            f"archive={data.get('archive_size')}, "
            f"winner={os.path.basename(str(data.get('legacy_winner', '')))}"
            f"{rep_text}, {_format_duration(data.get('duration_s', 0))})"
        )

    if event == "new_legacy_best":
        return (
            f"  (fitness={float(data.get('legacy_fitness', 0)):.3f}, "
            f"{os.path.basename(str(data.get('module_path', '')))})"
        )

    if event == "run_complete":
        return f"  ({_format_duration(data.get('duration_s', 0))})"

    if event == "log_write":
        return f"  ({data.get('path')})"

    if event == "confirmation_failed":
        return (
            f"  ({data.get('candidate_id')}, seed={data.get('seed')}: "
            f"{data.get('reason')})"
        )

    parts = []
    for key, value in data.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.3g}")
        elif isinstance(value, (list, tuple, dict)) and len(str(value)) > 48:
            continue
        else:
            parts.append(f"{key}={value}")
    return f"  ({', '.join(parts)})" if parts else ""


class _ConsoleFormatter(logging.Formatter):
    """Human-readable single-line console output with compact structured context."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level = record.levelname[:4]
        module = _short_logger(record.name)[:10]
        message = record.getMessage()
        context = _format_context(record)
        line = f"{timestamp} {level:4s} {module:10s} {message}{context}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


_configured = False
_DEFAULT_JSON_LOG_PATH = os.path.join("eureka", "eureka_run.jsonl")


def setup_logging(level: int | None = None) -> None:
    """Configure root eureka logger once (idempotent).

    Console output is always human-readable. Set ``EUREKA_LOG_JSON=1`` to also
    append one JSON object per line to ``eureka/eureka_run.jsonl`` (override with
    ``EUREKA_LOG_JSON_PATH``).
    """
    global _configured
    if _configured:
        return

    log_level = level if level is not None else getattr(
        logging, os.environ.get("EUREKA_LOG_LEVEL", "INFO").upper(), logging.INFO
    )
    write_json_file = os.environ.get("EUREKA_LOG_JSON", "0") == "1"

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_ConsoleFormatter())

    root = logging.getLogger("eureka")
    root.handlers.clear()
    root.addHandler(console_handler)

    if write_json_file:
        json_path = os.environ.get("EUREKA_LOG_JSON_PATH", _DEFAULT_JSON_LOG_PATH)
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        file_handler = logging.FileHandler(json_path, encoding="utf-8")
        file_handler.setFormatter(JsonLogFormatter())
        root.addHandler(file_handler)

    root.setLevel(log_level)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    if name.startswith("eureka."):
        return logging.getLogger(name)
    return logging.getLogger(f"eureka.{name}")