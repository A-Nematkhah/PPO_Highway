"""
report_html.py

Generates one self-contained report.html per run (Step 5 of the
experiment-manager brief). "Self-contained" means every plot is embedded
as a base64 data URI - the file can be copied, emailed, or archived on
its own without dragging a plots/ folder along with it.

All untrusted string content (candidate source code, reflection prompt
text, metadata strings) is HTML-escaped before insertion - candidate code
in particular is LLM-generated text that this report renders verbatim
inside <pre> blocks, so it must never be trusted as literal HTML.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

_CSS = """
:root {
    --bg: #0f1117; --panel: #171a23; --border: #2a2e3a; --text: #e6e8ef;
    --muted: #9aa1b2; --accent: #5fb4ff; --good: #55c98a; --bad: #ef6a6a;
    --winner: #2c3d2b;
}
* { box-sizing: border-box; }
body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0; padding: 2.5rem 3rem 4rem; line-height: 1.5;
}
h1 { font-size: 1.9rem; margin-bottom: 0.2rem; }
h2 {
    font-size: 1.3rem; margin-top: 2.8rem; padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border); color: var(--accent);
}
h3 { font-size: 1.05rem; color: var(--text); margin-top: 1.6rem; }
.subtitle { color: var(--muted); margin-bottom: 1.8rem; }
.summary-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.9rem; margin: 1rem 0 1.6rem;
}
.card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.9rem 1.1rem;
}
.card .label { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
.card .value { font-size: 1.15rem; margin-top: 0.25rem; font-variant-numeric: tabular-nums; }
table { border-collapse: collapse; width: 100%; margin: 0.8rem 0 1.6rem; font-size: 0.92rem; }
th, td { padding: 0.5rem 0.7rem; text-align: left; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.03em; }
tr.winner-row { background: var(--winner); }
tr.winner-row td:first-child { border-left: 3px solid var(--good); }
.badge {
    display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
}
.badge-winner { background: var(--good); color: #08130d; }
.badge-front { background: rgba(95,180,255,0.18); color: var(--accent); }
.badge-fail { background: rgba(239,106,106,0.18); color: var(--bad); }
pre {
    background: #0b0d13; border: 1px solid var(--border); border-radius: 8px;
    padding: 0.9rem 1rem; overflow-x: auto; font-size: 0.83rem;
    max-height: 420px; overflow-y: auto;
}
.plot-img { width: 100%; max-width: 1100px; border-radius: 8px; border: 1px solid var(--border); margin: 0.6rem 0 1.4rem; }
.mono { font-family: "SF Mono", Menlo, Consolas, monospace; }
.muted { color: var(--muted); }
details { margin-bottom: 0.6rem; }
summary { cursor: pointer; color: var(--accent); }
footer { color: var(--muted); font-size: 0.8rem; margin-top: 3rem; border-top: 1px solid var(--border); padding-top: 1rem; }
"""


def _esc(value) -> str:
    return html.escape(str(value)) if value is not None else ""


def _fmt(value, precision: int = 3) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        try:
            return f"{float(value):.{precision}f}"
        except (TypeError, ValueError):
            return _esc(value)
    return _esc(value) if value not in (None, "") else "&mdash;"


def _summary_card(label: str, value: str) -> str:
    return f'<div class="card"><div class="label">{_esc(label)}</div><div class="value">{value}</div></div>'


def _embed_plot(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img class="plot-img" src="data:image/png;base64,{data}" alt="{_esc(path.name)}">'


def generation_reason(record: dict) -> str:
    """Human-readable explanation of why that generation's winner/
    representative was selected - derived entirely from fields loop.py
    already logs (selection_mode, legacy_scalar_winner_on_front), no new
    selection logic."""
    mode = record.get("selection_mode")
    if record.get("all_candidates_failed"):
        return "No candidates survived generation (all LLM calls failed)."
    if not record.get("results"):
        return "No candidates survived smoke test / training / evaluation."
    if mode == "shadow":
        on_front = record.get("legacy_scalar_winner_on_front")
        base = "Highest legacy scalar fitness (diagnostic score, see fitness.py)."
        if on_front is False:
            base += " Note: this scalar winner was NOT on the Pareto front this generation."
        return base
    if mode == "pareto":
        return "Unweighted Pareto knee representative (closest to the ideal point across crash_rate/speed/overtakes)."
    return "Selected by the configured multi-objective mode."


def _render_summary_section(metadata: dict, config: dict, execution_stats: dict) -> str:
    cards = [
        _summary_card("Run", _esc(metadata.get("run_name", "?"))),
        _summary_card("Date (UTC)", _esc(metadata.get("timestamp_utc", "?"))[:19].replace("T", " ")),
        _summary_card("Total runtime", _fmt(execution_stats.get("total_runtime_s"), 1) + " s"),
        _summary_card("Git commit", (_esc(metadata.get("git_commit"))[:10] or "&mdash;") + (" (dirty)" if metadata.get("git_dirty") else "")),
        _summary_card("OS", f'{_esc(metadata.get("os_name"))} / {_esc(metadata.get("python_version"))}'),
        _summary_card("CPU", f'{_esc(metadata.get("cpu_model") or "unknown")} &times; {_esc(metadata.get("cpu_cores_logical") or "?")}'),
        _summary_card("RAM", (f'{metadata.get("ram_total_gb")} GB' if metadata.get("ram_total_gb") else "unknown")),
        _summary_card("LLM model", _esc(metadata.get("llm_model", "?"))),
    ]
    return f'<div class="summary-grid">{"".join(cards)}</div>'


def _render_training_summary(config: dict) -> str:
    cards = [
        _summary_card("Generations", _esc(config.get("n_generations", "?"))),
        _summary_card("Candidates / generation", _esc(config.get("k_candidates", "?"))),
        _summary_card("Training steps / candidate", _esc(config.get("train_steps_per_candidate", "?"))),
        _summary_card("Eval episodes", _esc(config.get("n_eval_episodes", "?"))),
        _summary_card("Selection mode", _esc(config.get("multi_objective_mode", "?"))),
        _summary_card("Confirmation seeds", _esc(config.get("confirmation_seeds", []))),
    ]
    return f'<div class="summary-grid">{"".join(cards)}</div>'


def _render_generation_sections(full_log: list[dict]) -> str:
    parts = []
    for record in full_log:
        gen = record.get("generation", "?")
        reason = generation_reason(record)
        results = record.get("results") or []
        front_ids = set(record.get("generation_front_candidate_ids") or [])
        winner_id = record.get("representative_id") or record.get("legacy_scalar_winner_id")

        rows = []
        for r in sorted(results, key=lambda x: x.get("candidate_index", 0)):
            metrics = r.get("metrics", {})
            is_winner = r.get("candidate_id") == winner_id
            on_front = r.get("candidate_id") in front_ids
            badges = ""
            if is_winner:
                badges += '<span class="badge badge-winner">winner</span> '
            elif on_front:
                badges += '<span class="badge badge-front">front</span> '
            rows.append(
                f'<tr class="{"winner-row" if is_winner else ""}">'
                f'<td class="mono">{_esc(r.get("module_path", "").rsplit(".", 1)[-1])} {badges}</td>'
                f'<td>{_esc(r.get("source", ""))}</td>'
                f'<td>{_fmt(metrics.get("crash_rate"), 3)}</td>'
                f'<td>{_fmt(metrics.get("mean_speed"), 2)}</td>'
                f'<td>{_fmt(metrics.get("mean_overtakes"), 2)}</td>'
                f'<td>{_fmt(r.get("legacy_fitness", r.get("fitness")), 3)}</td>'
                f'<td>{_esc(r.get("pareto_rank", "&mdash;"))}</td>'
                "</tr>"
            )

        table = (
            '<table><thead><tr><th>Candidate</th><th>Source</th><th>Crash rate</th>'
            '<th>Mean speed</th><th>Mean overtakes</th><th>Fitness</th><th>Pareto rank</th>'
            f"</tr></thead><tbody>{''.join(rows) or '<tr><td colspan=7 class=muted>No surviving candidates.</td></tr>'}</tbody></table>"
        )

        parts.append(
            f"<h3>Generation {_esc(gen)}</h3>"
            f'<p class="muted">{_esc(reason)}</p>'
            f"{table}"
        )
    return "".join(parts) or '<p class="muted">No generations recorded.</p>'


def _render_pareto_archive_table(archive: list[dict], representative_id: Optional[str]) -> str:
    if not archive:
        return '<p class="muted">Final archive is empty.</p>'

    ranked = sorted(archive, key=lambda c: (c.get("pareto_rank", 999), -(c.get("crowding_distance") or 0)))
    rows = []
    for c in ranked:
        metrics = c.get("metrics", {})
        is_winner = c.get("candidate_id") == representative_id
        rows.append(
            f'<tr class="{"winner-row" if is_winner else ""}">'
            f'<td>{_esc(c.get("pareto_rank", "&mdash;"))}</td>'
            f'<td class="mono">{_esc(str(c.get("module_path", "")).rsplit(".", 1)[-1])}'
            f'{" <span class=\'badge badge-winner\'>winner</span>" if is_winner else ""}</td>'
            f'<td>{_esc(c.get("generation", "&mdash;"))}</td>'
            f'<td>{_fmt(metrics.get("crash_rate"), 3)}</td>'
            f'<td>{_fmt(metrics.get("mean_speed"), 2)}</td>'
            f'<td>{_fmt(metrics.get("mean_overtakes"), 2)}</td>'
            f'<td>{_fmt(metrics.get("smoothness"), 3) if "smoothness" in metrics else "&mdash;"}</td>'
            f'<td>{_fmt(metrics.get("mean_raw_return"), 2)}</td>'
            "<td>&#10003;</td>"
            f'<td>{"&#10003;" if is_winner else ""}</td>'
            "</tr>"
        )

    return (
        "<table><thead><tr><th>Rank</th><th>Candidate</th><th>Generation</th>"
        "<th>Crash rate</th><th>Mean speed</th><th>Mean overtakes</th><th>Smoothness</th>"
        "<th>Raw return</th><th>Archive member</th><th>Winner</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _render_reflection_section(reflection_dir: Optional[Path]) -> str:
    if not reflection_dir or not reflection_dir.is_dir():
        return '<p class="muted">No reflection prompts were archived for this run.</p>'
    files = sorted(reflection_dir.glob("*.txt"))
    if not files:
        return '<p class="muted">No reflection prompts were archived for this run.</p>'
    parts = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        parts.append(
            f"<details><summary>{_esc(path.stem)}</summary><pre>{_esc(text)}</pre></details>"
        )
    return "".join(parts)


def _render_plots_section(plots_dir: Optional[Path]) -> str:
    if not plots_dir or not plots_dir.is_dir():
        return '<p class="muted">No plots were generated for this run.</p>'
    images = sorted(plots_dir.glob("*.png"))
    if not images:
        return '<p class="muted">No plots were generated for this run.</p>'
    return "".join(_embed_plot(p) for p in images)


def _render_execution_stats(execution_stats: dict) -> str:
    cards = [
        _summary_card("Training time", _fmt(execution_stats.get("train_time_s"), 1) + " s"),
        _summary_card("Evaluation time", _fmt(execution_stats.get("eval_time_s"), 1) + " s"),
        _summary_card("LLM time", _fmt(execution_stats.get("llm_time_s"), 1) + " s"),
        _summary_card("Total runtime", _fmt(execution_stats.get("total_runtime_s"), 1) + " s"),
    ]
    return f'<div class="summary-grid">{"".join(cards)}</div>'


def compute_execution_stats(telemetry_path: PathLike, total_runtime_s: Optional[float] = None) -> dict:
    """
    Aggregates train/eval/llm_generation durations straight out of the
    already-written telemetry.jsonl - pure post-processing over
    Telemetry's existing event stream, no new instrumentation.
    """
    stats = {"train_time_s": 0.0, "eval_time_s": 0.0, "llm_time_s": 0.0}
    path = Path(telemetry_path)
    if not path.is_file():
        if total_runtime_s is not None:
            stats["total_runtime_s"] = total_runtime_s
        return stats

    key_by_event = {"train": "train_time_s", "eval": "eval_time_s", "llm_generation": "llm_time_s"}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = key_by_event.get(row.get("event"))
            if key and isinstance(row.get("duration_s"), (int, float)):
                stats[key] += row["duration_s"]

    if total_runtime_s is not None:
        stats["total_runtime_s"] = total_runtime_s
    return stats


def generate_html_report(
    *,
    run_dir: PathLike,
    full_log: list[dict],
    archive: list[dict],
    representative_id: Optional[str],
    metadata: dict,
    config: dict,
    execution_stats: dict,
    plots_dir: Optional[PathLike] = None,
    reflection_dir: Optional[PathLike] = None,
    out_path: Optional[PathLike] = None,
) -> Path:
    """Builds the full self-contained report.html and writes it to
    run_dir/report.html (or out_path if given). Returns the path."""
    run_dir = Path(run_dir)
    out_path = Path(out_path) if out_path else run_dir / "report.html"

    body = f"""
<h1>EUREKA Experiment Report</h1>
<p class="subtitle">{_esc(metadata.get("run_name", "?"))} &middot; generated automatically after run completion</p>

<h2>Experiment Summary</h2>
{_render_summary_section(metadata, config, execution_stats)}

<h2>Training Summary</h2>
{_render_training_summary(config)}

<h2>Generation Summary</h2>
{_render_generation_sections(full_log)}

<h2>Pareto Archive</h2>
{_render_pareto_archive_table(archive, representative_id)}

<h2>Reflection</h2>
{_render_reflection_section(Path(reflection_dir) if reflection_dir else None)}

<h2>Plots</h2>
{_render_plots_section(Path(plots_dir) if plots_dir else None)}

<h2>Execution Statistics</h2>
{_render_execution_stats(execution_stats)}

<footer>Generated by eureka/report_html.py &middot; config/metadata JSON alongside this file has the full detail.</footer>
"""

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EUREKA report - {_esc(metadata.get("run_name", ""))}</title>
<style>{_CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
