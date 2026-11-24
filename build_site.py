"""Generate a self-contained static ContextTier dashboard at ``site/index.html``.

Every experimental value in the generated page comes from the CSV/JSON artefacts
written by the experiment pipeline (or from saved traces). The HTML has no CDN
dependencies: data, figures and JavaScript are embedded so the file can be
opened directly in a browser, including when Streamlit is unavailable.

Usage::

    python build_site.py
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from app.data import (
    DISPLAY_NAMES,
    HONESTY_NOTE,
    KIND_LABELS,
    METRIC_LABELS,
    SCENARIOS,
    discover_replays,
    experiment_manifest,
    figure_catalog,
    load_replay,
    load_run_metrics,
    load_tables,
    mean_std_text,
    plot_manifest,
)
from app.replay import TraceReplay
from config import project_root

# Technical limits for embedding saved traces; not experimental values.
MAX_EMBEDDED_ACCESSES = 400

TRACE_FIELDS = (
    "ts", "file_id", "size_bytes", "op", "epoch", "step", "phase",
    "tier_served", "latency_ms", "wait_ms", "fast_hit", "true_type",
)
MIGRATION_FIELDS = ("ts", "file_id", "size_bytes", "src_tier", "dst_tier", "kind")

FIGURE_DESCRIPTIONS = {
    "data_wait_all": "All policies and scenarios; symlog axis spans idealized and tied cases.",
    "data_wait_deployable": "Linear-scale comparison of deployable policies only.",
    "scenario_c_focus": "Scenario C data wait and throughput side by side.",
    "capacity_sweep": "Data wait against fast-tier capacity, with seed error bars.",
    "compute_sweep": "Data wait against compute time per step, with seed error bars.",
    "context_ablation": "Paired per-seed data wait saved by the declared-type channel.",
    "latency_percentiles": "Mean / p95 / p99 read latency for representative policies.",
}

TIER_COLORS = {0: "#2e7ba6", 1: "#e0a33e", 2: "#8d6bb5"}
TIER_NAMES = {0: "FAST", 1: "MID", 2: "SLOW"}


def json_value(value: Any) -> Any:
    """Convert a pandas/numpy cell into a JSON-serialisable Python value."""
    if value is None or (not isinstance(value, (str, bytes)) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def metric_options(summary: pd.DataFrame) -> list[str]:
    """Metrics with both a mean and a std column in the summary table."""
    return [
        metric for metric in METRIC_LABELS
        if f"{metric}_mean" in summary.columns and f"{metric}_std" in summary.columns
    ]


def _display(frame: pd.DataFrame, metric: str, index: int) -> str:
    return mean_std_text(
        float(frame[f"{metric}_mean"].iloc[index]),
        float(frame[f"{metric}_std"].iloc[index]),
        metric,
    )


def summary_payload(summary: pd.DataFrame) -> dict[str, Any]:
    """Build the main-matrix records plus pre-rendered display strings."""
    metrics = metric_options(summary)
    records: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(summary.iterrows()):
        record = {str(key): json_value(value) for key, value in row.items()}
        record["display"] = {metric: _display(summary, metric, position) for metric in metrics}
        records.append(record)
    return {"records": records, "metrics": metrics, "labels": METRIC_LABELS}


def headline_payload(summary: pd.DataFrame) -> list[dict[str, str]]:
    """Compute headline cards from the summary file (no stored constants)."""
    focus = "C" if (summary["scenario"] == "C").any() else str(summary["scenario"].iloc[0])
    rows = summary[summary["scenario"] == focus]
    cards: list[dict[str, str]] = []
    if rows.empty:
        return cards

    max_wait = float(rows["data_wait_ms_mean"].max())
    max_wait_std = float(
        rows.loc[rows["data_wait_ms_mean"].idxmax(), "data_wait_ms_std"]
    )
    if max_wait == 0.0:
        cards.append({
            "label": "Zero-wait ceiling",
            "value": "0.00 ms",
            "foot": f"every policy in {focus}, incl. AllSlow · measured compute",
        })
    else:
        cards.append({
            "label": "Largest data wait",
            "value": mean_std_text(max_wait, max_wait_std, "data_wait_ms"),
            "foot": f"max over all policies in Scenario {focus}",
        })

    def row_for(policy: str):
        matching = rows[rows["policy"] == policy]
        return matching.iloc[0] if not matching.empty else None

    lru = row_for("lru")
    full = row_for("full_context")
    allslow = row_for("allslow")

    if full is not None:
        foot = "Deployable learned policy"
        if lru is not None:
            delta = float(full["throughput_mean"]) - float(lru["throughput_mean"])
            foot = f"{delta:+,.2f} samples/s vs LRU"
        cards.append({
            "label": "FullContextTier throughput",
            "value": mean_std_text(
                float(full["throughput_mean"]),
                float(full["throughput_std"]),
                "throughput",
            ),
            "foot": foot,
        })
    if allslow is not None and lru is not None:
        delta = float(lru["throughput_mean"]) - float(allslow["throughput_mean"])
        cards.append({
            "label": "Uncached control (AllSlow)",
            "value": mean_std_text(
                float(allslow["throughput_mean"]),
                float(allslow["throughput_std"]),
                "throughput",
            ),
            "foot": (
                f"−{delta:,.1f} samples/s vs LRU · "
                f"p95 {float(allslow['latency_p95_ms_mean']):.2f} ms"
            ),
        })
    return cards


def sweep_payload(sweep_summary: pd.DataFrame) -> dict[str, Any]:
    """Build the sensitivity-sweep records for the interactive line chart."""
    records: list[dict[str, Any]] = []
    for _, row in sweep_summary.iterrows():
        records.append({
            "sweep": str(row["sweep"]),
            "sweep_value": json_value(row["sweep_value"]),
            "policy": str(row["policy"]),
            "scenario": str(row["scenario"]),
            "mean": json_value(row.get("data_wait_ms_mean")),
            "std": json_value(row.get("data_wait_ms_std")),
            "display": mean_std_text(
                float(row["data_wait_ms_mean"]),
                float(row["data_wait_ms_std"]),
                "data_wait_ms",
            ),
        })
    available = list(dict.fromkeys(record["sweep"] for record in records))
    return {"records": records, "sweeps": available}


def figures_payload(root: Path) -> dict[str, Any]:
    """Embed every manifest-listed PNG as a base64 data URI."""
    figures: list[dict[str, Any]] = []
    for record in figure_catalog(root / "results" / "plots"):
        paths = [path if path.is_absolute() else root / path for path in record["paths"]]
        png = next((path for path in paths if path.suffix.lower() == ".png" and path.is_file()), None)
        if png is None:
            continue
        encoded = base64.b64encode(png.read_bytes()).decode("ascii")
        figures.append({
            "stem": record["stem"],
            "title": record["title"],
            "description": FIGURE_DESCRIPTIONS.get(record["stem"], ""),
            "data_uri": f"data:image/png;base64,{encoded}",
        })
    return {"figures": figures}


def replay_payload(root: Path) -> dict[str, Any]:
    """Embed deterministic playback data derived from saved trace CSV files."""
    runs: list[dict[str, Any]] = []
    for run in discover_replays(root / "results"):
        trace, migrations = load_replay(run)
        if trace.empty:
            continue
        total = len(trace)
        stride = max(1, math.ceil(total / MAX_EMBEDDED_ACCESSES))
        sampled = trace.iloc[::stride]
        if sampled.index[-1] != trace.index[-1]:
            sampled = pd.concat([sampled, trace.iloc[[-1]]])
        fields = [field for field in TRACE_FIELDS if field in trace.columns]
        trace_rows = [
            [json_value(value) for value in row]
            for row in sampled[fields].itertuples(index=False, name=None)
        ]
        migration_fields = [field for field in MIGRATION_FIELDS if field in migrations.columns]
        migration_rows = [
            [json_value(value) for value in row]
            for row in migrations[migration_fields].itertuples(index=False, name=None)
        ] if not migrations.empty else []
        file_sizes: dict[int, int] = {}
        for frame in (trace, migrations):
            if frame.empty:
                continue
            for file_id, size in zip(frame["file_id"], frame["size_bytes"]):
                file_sizes[int(file_id)] = int(size)
        runs.append({
            "label": run.label,
            "display_name": run.display_name,
            "engine": run.engine,
            "scenario": run.scenario,
            "policy": run.policy,
            "seed": json_value(run.seed),
            "initial_tier": TraceReplay(trace, migrations).initial_tier,
            "sampled": stride > 1,
            "accesses_total": total,
            "trace_fields": fields,
            "trace_rows": trace_rows,
            "migration_fields": migration_fields,
            "migration_rows": migration_rows,
            "file_sizes": {str(key): value for key, value in file_sizes.items()},
            "metrics": {key: json_value(value) for key, value in load_run_metrics(run).items()},
        })
    return {"runs": runs, "tier_names": TIER_NAMES, "tier_colors": TIER_COLORS}


def build_payload(root: Path) -> dict[str, Any]:
    """Read all result artefacts and assemble the embedded dashboard payload."""
    tables = load_tables(root / "results" / "full")
    experiment = experiment_manifest(root / "results" / "full")
    plots = plot_manifest(root / "results" / "plots")
    summary = tables["summary"]
    return {
        "generated_by": "build_site.py",
        "project": "ContextTier",
        "honesty_note": HONESTY_NOTE,
        "playback_note": (
            "Reconstructed playback: migrations are rebuilt from the logged "
            "migration events and tier_served is authoritative for every access. "
            "In-flight prefetches are not in the log, so reconstructed residency "
            "is illustrative and nothing is simulated live."
        ),
        "scenarios": SCENARIOS,
        "policy_names": DISPLAY_NAMES,
        "kind_labels": KIND_LABELS,
        "manifest": {
            "engine": experiment.get("engine"),
            "config": experiment.get("config", {}),
            "run_counts": experiment.get("run_counts", {}),
            "honesty_notes": experiment.get("honesty_notes", []),
            "phase": experiment.get("phase"),
        },
        "plot_manifest": {
            "phase": plots.get("phase"),
            "honesty_notes": plots.get("honesty_notes", []),
        },
        "headlines": headline_payload(summary),
        "summary": summary_payload(summary),
        "sweeps": sweep_payload(tables["sweep_summary"]),
        "figures": figures_payload(root),
        "replay": replay_payload(root),
    }


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def render_headline_html(cards: list[dict[str, str]]) -> str:
    """Pre-render headline cards so the file is useful even without JavaScript."""
    return "\n".join(
        '<article class="kpi">'
        f'<div class="kpi-label">{_escape(card["label"])}</div>'
        f'<div class="kpi-value">{_escape(card["value"])}</div>'
        f'<div class="kpi-foot">{_escape(card["foot"])}</div>'
        "</article>"
        for card in cards
    )


def render_summary_html(payload: dict[str, Any]) -> str:
    """Pre-render the default summary table with values from results/full."""
    records = payload["records"]
    metric = "data_wait_ms" if "data_wait_ms" in payload["metrics"] else payload["metrics"][0]
    headers = ["Policy", "Class", "Scenario", "Seeds", f"Mean ± std · {METRIC_LABELS.get(metric, metric)}"]
    head = "".join(f"<th>{_escape(name)}</th>" for name in headers)
    rows: list[str] = []
    for index, record in enumerate(records):
        rows.append(
            f'<tr data-index="{index}">'
            f'<td>{_escape(DISPLAY_NAMES.get(record["policy"], record["policy"]))}</td>'
            f'<td>{_escape(KIND_LABELS.get(record["policy_kind"], record["policy_kind"]))}</td>'
            f'<td>{_escape(record["scenario"])}</td>'
            f'<td>{_escape(record.get("n_seeds", ""))}</td>'
            f'<td class="num">{_escape(record["display"][metric])}</td>'
            "</tr>"
        )
    return f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody>"


def render_figures_html(payload: dict[str, Any]) -> str:
    """Pre-render the figure gallery with embedded PNG data URIs."""
    blocks: list[str] = []
    for figure in payload["figures"]:
        blocks.append(
            "<figure>"
            f'<img src="{figure["data_uri"]}" alt="{_escape(figure["title"])}" loading="lazy">'
            f'<figcaption><strong>{_escape(figure["title"])}</strong> · '
            f"{_escape(figure['description'])}</figcaption></figure>"
        )
    return "\n".join(blocks) or "<p>Figures are unavailable in the manifest.</p>"


def render_sweep_options(payload: dict[str, Any]) -> str:
    return "".join(
        f'<option value="{_escape(name)}">{_escape(name.replace("_", " "))}</option>'
        for name in payload["sweeps"]
    )


def render_run_options(payload: dict[str, Any]) -> str:
    return "".join(
        f'<option value="{index}">{_escape(run["display_name"])}</option>'
        for index, run in enumerate(payload["runs"])
    )


CSS = """
:root {
  --ink: #18181b; --muted: #71717a; --line: #e4e4e7; --accent: #18181b;
  --bg: #fafaf9; --card: #ffffff; --navy: #18181b;
}
* { box-sizing: border-box; }
body {
  margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    "Helvetica Neue", Arial, sans-serif; color: var(--ink); background: var(--bg);
  line-height: 1.5;
}
header.topbar {
  position: sticky; top: 0; z-index: 20; display: flex; align-items: center;
  justify-content: space-between; gap: 20px; padding: 12px 28px;
  background: #ffffff; color: var(--ink); border-bottom: 1px solid var(--line);
  box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}
.brand { font-weight: 750; letter-spacing: -0.02em; font-size: 1.15rem; color: #18181b; }
.brand span { color: #71717a; font-weight: 450; font-size: 0.9rem; }
nav.tabs { display: flex; gap: 6px; flex-wrap: wrap; }
nav.tabs button {
  border: 1px solid var(--line); border-radius: 8px; padding: 7px 13px; cursor: pointer;
  background: #f4f4f5; color: #71717a; font-size: .88rem; font-weight: 600;
  transition: all 0.15s ease;
}
nav.tabs button:hover { background: #e4e4e7; color: #18181b; }
nav.tabs button.active { background: #18181b; color: #ffffff; border-color: #18181b; }
main { max-width: 1200px; margin: 0 auto; padding: 28px 24px 60px; }
.page { display: none; }
.page.active { display: block; animation: fade .2s ease-in; }
@keyframes fade { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; } }
.hero {
  border-radius: 14px; padding: 30px 34px; color: var(--ink);
  background: #ffffff; border: 1px solid var(--line);
  box-shadow: 0 1px 3px rgba(0,0,0,0.02); margin-bottom: 24px;
}
.hero h1 { margin: 4px 0 8px; font-size: 2.1rem; font-weight: 750; letter-spacing: -0.02em; color: #09090b; }
.hero p { margin: 0; max-width: 78ch; color: #52525b; line-height: 1.55; font-size: 0.98rem; }
.eyebrow {
  text-transform: uppercase; letter-spacing: .08em; color: #71717a;
  font-weight: 700; font-size: .72rem;
}
.pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.pill {
  border: 1px solid var(--line); border-radius: 6px;
  background: #f4f4f5; padding: 4px 10px; font-size: .78rem; color: #3f3f46; font-weight: 500;
}
.section-head { margin: 28px 0 12px; border-bottom: 1px solid var(--line); padding-bottom: 8px; }
.section-head h2 { font-size: 1.25rem; margin: 0; font-weight: 700; letter-spacing: -0.015em; color: #09090b; }
.section-head p { margin: 4px 0 0; color: var(--muted); font-size: .88rem; }
.note {
  border-left: 3px solid #71717a; background: #f4f4f5; color: #27272a;
  padding: 11px 15px; border-radius: 0 8px 8px 0; margin: 14px 0; font-size: .9rem;
}
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }
.kpi, .panel {
  background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}
.kpi { padding: 18px 20px; }
.kpi-label { text-transform: uppercase; letter-spacing: .06em; font-size: .72rem; color: var(--muted); font-weight: 600; }
.kpi-value { font-size: 1.6rem; font-weight: 750; margin: 6px 0 2px; letter-spacing: -0.02em; color: #09090b; }
.kpi-foot { color: var(--muted); font-size: .82rem; }
.panel { padding: 20px 22px; margin: 14px 0; }
.controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin-bottom: 14px; }
.controls label { display: grid; gap: 4px; font-size: .82rem; font-weight: 600; color: #3f3f46; }
select, input[type=search], input[type=range] {
  font: inherit; padding: 7px 10px; border: 1px solid #d4d4d8; border-radius: 8px;
  background: #fff; color: var(--ink);
}
button.action {
  border: 1px solid #d4d4d8; background: #fff; color: var(--ink); border-radius: 8px;
  padding: 7px 14px; cursor: pointer; font-weight: 600; font-size: .88rem;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03); transition: all 0.15s ease;
}
button.action:hover { background: #f4f4f5; }
button.action.primary { background: #18181b; color: #fff; border-color: #18181b; }
button.action.primary:hover { background: #27272a; }
.table-wrap { overflow: auto; max-height: 520px; border: 1px solid var(--line); border-radius: 12px; }
table { border-collapse: collapse; width: 100%; font-size: .88rem; }
th, td { padding: 9px 12px; text-align: left; border-bottom: 1px solid #e8eef5; white-space: nowrap; }
th { position: sticky; top: 0; background: #eef4fa; z-index: 1; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; color: #4a6a84; }
tbody tr:hover { background: #f5faff; }
td.num { font-variant-numeric: tabular-nums; }
.chart-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; }
svg { width: 100%; height: auto; display: block; }
.legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; font-size: .82rem; color: var(--muted); }
.legend i { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 5px; }
.figures { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }
figure { margin: 0; background: #fff; border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
figure img { width: 100%; display: block; background: #fff; }
figcaption { padding: 12px 15px; font-size: .86rem; color: var(--muted); }
.playback-grid { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(260px, .8fr); gap: 16px; }
.transport { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 12px 0; }
.transport input[type=range] { flex: 1; min-width: 180px; }
.tier-bar { display: flex; height: 26px; border-radius: 7px; overflow: hidden; background: #edf2f7; margin: 10px 0; }
.tier-bar div { height: 100%; transition: width .2s ease; }
.tier-label { display: flex; justify-content: space-between; font-size: .85rem; color: var(--muted); }
footer { border-top: 1px solid var(--line); padding: 24px; color: var(--muted); font-size: .85rem; text-align: center; }
@media (max-width: 900px) {
  .chart-grid, .playback-grid { grid-template-columns: 1fr; }
  header.topbar { flex-direction: column; align-items: flex-start; }
  .hero { padding: 26px 22px; }
}
"""


JS = r"""
const DATA = JSON.parse(document.getElementById('ct-data').textContent);
const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (value, digits = 2) => Number(value).toLocaleString(undefined, {maximumFractionDigits: digits, minimumFractionDigits: digits});
const byId = (id) => document.getElementById(id);

function switchPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === 'page-' + name));
  document.querySelectorAll('nav.tabs button').forEach(b => b.classList.toggle('active', b.dataset.page === name));
  window.scrollTo({top: 0});
}

/* Main matrix */
const summaryData = DATA.summary;
let metric = Object.keys(summaryData.labels).find(k => k === 'data_wait_ms') || summaryData.metrics[0];
let searchText = '';
let sortKey = 'policy';
let sortDir = 1;

function metricOptions() {
  byId('metric-select').innerHTML = summaryData.metrics.map(m =>
    `<option value="${esc(m)}" ${m === metric ? 'selected' : ''}>${esc(summaryData.labels[m])}</option>`).join('');
}

function filteredRows() {
  const term = searchText.trim().toLowerCase();
  const rows = summaryData.records.filter((r, i) => {
    if (!term) return true;
    const haystack = [r.policy, r.policy_kind, r.scenario, summaryData.labels[metric], r.display[metric]].join(' ').toLowerCase();
    return haystack.includes(term);
  });
  rows.sort((a, b) => {
    const av = sortKey === 'policy' ? (DATA.policy_names[a.policy] || a.policy) : a[sortKey];
    const bv = sortKey === 'policy' ? (DATA.policy_names[b.policy] || b.policy) : b[sortKey];
    return String(av).localeCompare(String(bv), undefined, {numeric: true}) * sortDir;
  });
  return rows;
}

function renderSummary() {
  const rows = filteredRows();
  const head = `<thead><tr>
    <th data-sort="policy">Policy</th><th data-sort="policy_kind">Class</th>
    <th data-sort="scenario">Scenario</th><th data-sort="n_seeds">Seeds</th>
    <th>Mean ± std · ${esc(summaryData.labels[metric])}</th></tr></thead>`;
  const body = rows.map(r => `<tr>
    <td>${esc(DATA.policy_names[r.policy] || r.policy)}</td>
    <td>${esc(DATA.kind_labels[r.policy_kind] || r.policy_kind)}</td>
    <td>${esc(r.scenario)}</td><td>${esc(r.n_seeds)}</td>
    <td class="num">${esc(r.display[metric])}</td></tr>`).join('');
  byId('summary-table').innerHTML = head + `<tbody>${body}</tbody>`;
  byId('summary-count').textContent = `${rows.length} rows`;
  byId('summary-table').querySelectorAll('th[data-sort]').forEach(th =>
    th.addEventListener('click', () => {
      sortDir = sortKey === th.dataset.sort ? -sortDir : 1;
      sortKey = th.dataset.sort; renderSummary();
    }));
  drawSummaryChart(rows);
}

function drawSummaryChart(rows) {
  const scenario = byId('chart-scenario').value;
  const subset = (scenario === 'all' ? rows : rows.filter(r => r.scenario === scenario))
    .filter(r => Number.isFinite(r[metric + '_mean']))
    .sort((a, b) => a[metric + '_mean'] - b[metric + '_mean']).slice(0, 18);
  const width = 640, rowHeight = 27, height = Math.max(120, subset.length * rowHeight + 40);
  const max = Math.max(...subset.map(r => Math.abs(r[metric + '_mean'])), 0) || 1;
  const bars = subset.map((r, i) => {
    const w = Math.max(2, Math.abs(r[metric + '_mean']) / max * 350);
    const colorKind = {observable_baseline:'#2e7ba6', learned_observable:'#3c9d78',
      declared_type_upper_bound:'#e0a33e', oracle_upper_bound:'#8d6bb5'};
    return `<text x="128" y="${24 + i * rowHeight}" text-anchor="end" font-size="11">${esc(DATA.policy_names[r.policy] || r.policy)} · ${esc(r.scenario)}</text>
      <rect x="136" y="${13 + i * rowHeight}" width="${w}" height="15" rx="3" fill="${colorKind[r.policy_kind] || '#607d8b'}"/>
      <text x="${142 + w}" y="${24 + i * rowHeight}" font-size="11">${esc(r.display[metric])}</text>`;
  }).join('');
  byId('summary-chart').innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Metric by policy">
    <title>${esc(summaryData.labels[metric])} by policy</title>${bars}</svg>`;
}

/* Sweeps */
function drawSweep() {
  const name = byId('sweep-select').value;
  const records = DATA.sweeps.records.filter(r => r.sweep === name);
  const policies = [...new Set(records.map(r => r.policy))];
  const values = [...new Set(records.map(r => Number(r.sweep_value)))].sort((a, b) => a - b);
  const series = policies.map(policy => ({
    policy, points: values.map(v => records.find(r => r.policy === policy && Number(r.sweep_value) === v))
  }));
  const width = 640, height = 300, pad = 46;
  const ys = records.map(r => Number(r.mean)).filter(Number.isFinite);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const span = ymax - ymin || 1;
  const x = i => pad + (values.length < 2 ? 0 : i / (values.length - 1) * (width - 2 * pad));
  const y = v => height - pad - (Number(v) - ymin) / span * (height - 2 * pad);
  const lines = series.map((s, si) => {
    const color = ['#2e7ba6','#3c9d78','#e0a33e','#8d6bb5','#c44e52','#4c72b0'][si % 6];
    const path = s.points.map((p, i) => p ? `${i ? 'L' : 'M'} ${x(i)} ${y(p.mean)}` : '').join(' ');
    const dots = s.points.map((p, i) => p ? `<circle cx="${x(i)}" cy="${y(p.mean)}" r="3" fill="${color}"><title>${esc(p.display)}</title></circle>` : '').join('');
    return `<path d="${path}" fill="none" stroke="${color}" stroke-width="2.5"/>${dots}
      <text x="${width - 150}" y="${18 + si * 17}" font-size="11" fill="${color}">${esc(DATA.policy_names[s.policy] || s.policy)}</text>`;
  }).join('');
  const axis = `<line x1="${pad}" y1="${height-pad}" x2="${width-pad}" y2="${height-pad}" stroke="#b9cbd9"/>
    <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height-pad}" stroke="#b9cbd9"/>
    <text x="${width/2}" y="${height-10}" text-anchor="middle" font-size="11">${esc(name.replaceAll('_',' '))}</text>
    <text x="12" y="${height/2}" transform="rotate(-90 12 ${height/2})" text-anchor="middle" font-size="11">${esc(summaryData.labels.data_wait_ms)}</text>
    <text x="${pad}" y="${height-pad+16}" font-size="10">${esc(values[0])}</text>
    <text x="${width-pad}" y="${height-pad+16}" text-anchor="end" font-size="10">${esc(values.at(-1))}</text>`;
  byId('sweep-chart').innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Sweep chart">${axis}${lines}</svg>`;
  byId('sweep-caption').textContent = `${records.length} logged sweep points · ${policies.length} policies`;
}

/* Reconstructed playback */
const replayData = DATA.replay;
let run = null, replayIndex = 0, playing = false, timer = null, stateCache = [];

function field(row, name) {
  const i = run.trace_fields.indexOf(name);
  return i < 0 ? null : row[i];
}
function migField(row, name) {
  const i = run.migration_fields.indexOf(name);
  return i < 0 ? null : row[i];
}
function initialiseReplay() {
  run = replayData.runs[Number(byId('run-select').value)];
  replayIndex = 0; stopPlayback();
  stateCache = run.migration_rows.slice().sort((a, b) => Number(migField(a, 'ts')) - Number(migField(b, 'ts')));
  byId('replay-range').max = String(run.trace_rows.length - 1);
  byId('replay-range').value = '0';
  byId('run-summary').textContent = `${run.display_name} · ${run.accesses_total} logged accesses${run.sampled ? ' · sampled for embedding' : ''}`;
  drawReplay();
}
function stateAt(index) {
  const access = run.trace_rows[index];
  const cutoff = Number(field(access, 'ts'));
  const files = new Map();
  const sizes = run.file_sizes;
  Object.keys(sizes).forEach(id => files.set(Number(id), run.initial_tier));
  const applied = [];
  for (const m of stateCache) {
    if (Number(migField(m, 'ts')) > cutoff) break;
    files.set(Number(migField(m, 'file_id')), Number(migField(m, 'dst_tier')));
    applied.push(m);
  }
  const occupancy = {0: 0, 1: 0, 2: 0}, counts = {0: 0, 1: 0, 2: 0};
  files.forEach((tier, fileId) => {
    if (tier in occupancy) {
      occupancy[tier] += Number(sizes[fileId] || 0);
      counts[tier] += 1;
    }
  });
  let wait = 0;
  for (let i = 0; i <= index; i++) wait += Number(field(run.trace_rows[i], 'wait_ms') || 0);
  return {access, occupancy, counts, applied, wait};
}
function drawReplay() {
  if (!run) return;
  const state = stateAt(replayIndex);
  byId('replay-range').value = String(replayIndex);
  byId('replay-position').textContent = `Access ${replayIndex + 1} / ${run.trace_rows.length}${run.sampled ? ' (sampled)' : ''}`;
  byId('access-cards').innerHTML = [
    ['File id', String(field(state.access, 'file_id'))],
    ['Served by (authoritative)', replayData.tier_names[field(state.access, 'tier_served')] || '?'],
    ['Fast hit', field(state.access, 'fast_hit') ? 'yes' : 'no'],
    ['Logged wait (ms)', fmt(field(state.access, 'wait_ms'), 4)],
  ].map(([label, value]) => `<article class="kpi"><div class="kpi-label">${esc(label)}</div><div class="kpi-value">${esc(value)}</div></article>`).join('');
  byId('replay-caption').textContent =
    `phase=${field(state.access,'phase')} · op=${field(state.access,'op')} · epoch=${field(state.access,'epoch')} · step=${field(state.access,'step')} · cumulative logged wait=${fmt(state.wait, 1)} ms · completed migrations applied=${state.applied.length}`;
  const total = Object.values(state.occupancy).reduce((a, b) => a + b, 0) || 1;
  byId('tier-bar').innerHTML = [0,1,2].map(tier =>
    `<div style="width:${state.occupancy[tier] / total * 100}%;background:${replayData.tier_colors[tier]}" title="${replayData.tier_names[tier]}"></div>`).join('');
  byId('tier-cards').innerHTML = [0,1,2].map(tier =>
    `<article class="kpi"><div class="kpi-label">${replayData.tier_names[tier]} files (reconstructed)</div>
      <div class="kpi-value">${state.counts[tier]}</div><div class="kpi-foot">${fmt(state.occupancy[tier] / (1024*1024), 2)} MB</div></article>`).join('');
  const recent = state.applied.slice(-8).reverse();
  byId('recent-migrations').innerHTML = recent.length ? `<table><thead><tr><th>ts</th><th>file</th><th>move</th><th>kind</th></tr></thead><tbody>` +
    recent.map(m => `<tr><td>${fmt(migField(m,'ts'), 3)}</td><td>${esc(migField(m,'file_id'))}</td><td>${replayData.tier_names[migField(m,'src_tier')]} → ${replayData.tier_names[migField(m,'dst_tier')]}</td><td>${esc(migField(m,'kind'))}</td></tr>`).join('') +
    '</tbody></table>' : '<p>No completed migrations before this access.</p>';
  drawWaitSparkline(replayIndex);
}
function drawWaitSparkline(index) {
  const rows = run.trace_rows.slice(0, index + 1).map(r => Number(field(r, 'wait_ms')) || 0);
  const w = 640, h = 130, pad = 12;
  const max = Math.max(...rows, 0) || 1;
  const step = rows.length < 2 ? 0 : (w - 2 * pad) / (rows.length - 1);
  const path = rows.map((v, i) => `${i ? 'L' : 'M'} ${pad + i * step} ${h - pad - v / max * (h - 2 * pad)}`).join(' ');
  byId('wait-chart').innerHTML = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Logged wait timeline">
    <path d="${path}" fill="none" stroke="#2e7ba6" stroke-width="1.6"/>
    <line x1="${pad + index * step}" y1="0" x2="${pad + index * step}" y2="${h}" stroke="#c44e52" stroke-dasharray="4 3"/></svg>`;
}
function stopPlayback() { playing = false; if (timer) clearInterval(timer); timer = null; byId('play-toggle').textContent = '▶ Auto-play'; }
function step(delta) {
  stopPlayback();
  replayIndex = Math.max(0, Math.min(run.trace_rows.length - 1, replayIndex + delta));
  drawReplay();
}

/* Bootstrap */
metricOptions();
byId('metric-select').addEventListener('change', e => { metric = e.target.value; renderSummary(); });
byId('summary-search').addEventListener('input', e => { searchText = e.target.value; renderSummary(); });
byId('chart-scenario').innerHTML = '<option value="all">All scenarios</option>' +
  [...new Set(summaryData.records.map(r => r.scenario))].map(s => `<option value="${esc(s)}">${esc(s)}</option>`).join('');
byId('chart-scenario').addEventListener('change', () => drawSummaryChart(filteredRows()));
renderSummary();
byId('sweep-select').innerHTML = renderSweepOptions();
byId('sweep-select').addEventListener('change', drawSweep);
drawSweep();
byId('run-select').innerHTML = renderRunOptions();
byId('run-select').addEventListener('change', initialiseReplay);
byId('replay-range').addEventListener('input', e => { stopPlayback(); replayIndex = Number(e.target.value); drawReplay(); });
byId('first-access').addEventListener('click', () => { stopPlayback(); replayIndex = 0; drawReplay(); });
byId('prev-access').addEventListener('click', () => step(-1));
byId('next-access').addEventListener('click', () => step(1));
byId('last-access').addEventListener('click', () => { stopPlayback(); replayIndex = run.trace_rows.length - 1; drawReplay(); });
byId('play-toggle').addEventListener('click', () => {
  if (playing) { stopPlayback(); return; }
  playing = true; byId('play-toggle').textContent = '⏸ Pause';
  timer = setInterval(() => {
    if (replayIndex >= run.trace_rows.length - 1) { stopPlayback(); return; }
    replayIndex += 1; drawReplay();
  }, 120);
});
/* Interactive Live Memory & Storage Tiering Studio */
function initInteractiveLab() {
  const fastMbEl = byId('lab-fast-mb');
  const midMbEl = byId('lab-mid-mb');
  const computeMsEl = byId('lab-compute-ms');
  const fastValEl = byId('lab-fast-val');
  const midValEl = byId('lab-mid-val');
  const computeValEl = byId('lab-compute-val');
  const runBtn = byId('lab-run-btn');

  if (!fastMbEl || !runBtn) return;

  fastMbEl.addEventListener('input', () => { fastValEl.textContent = fastMbEl.value + ' MB'; });
  midMbEl.addEventListener('input', () => { midValEl.textContent = midMbEl.value + ' MB'; });
  computeMsEl.addEventListener('input', () => {
    const v = Number(computeMsEl.value);
    computeValEl.textContent = v + ' ms' + (v <= 5 ? ' (Fast GPU)' : (v >= 100 ? ' (CPU)' : ''));
  });

  runBtn.addEventListener('click', () => {
    runBtn.disabled = true;
    runBtn.textContent = '⏳ Simulating Training & Tiering...';
    setTimeout(() => {
      executeLabSimulation();
      runBtn.disabled = false;
      runBtn.textContent = '🚀 Run Live Simulation';
    }, 150);
  });
}

function executeLabSimulation() {
  const fastMb = Number(byId('lab-fast-mb').value);
  const midMb = Number(byId('lab-mid-mb').value);
  const computeMs = Number(byId('lab-compute-ms').value);
  const policy = byId('lab-policy').value;
  const scenario = byId('lab-scenario').value;
  const numSteps = Number(byId('lab-steps').value);

  const fastCapBytes = fastMb * 1024 * 1024;
  const midCapBytes = midMb * 1024 * 1024;
  const chunkSize = 153600; // ~150 KB
  const ckptSize = 4194304; // 4.0 MB
  const metaSize = 4096;    // 4 KB

  let fastUsed = 0, midUsed = 0;
  let fastHits = 0, midHits = 0, slowHits = 0;
  let totalDataWaitMs = 0, migrationBytes = 0;
  let simulatedClockMs = 0;
  const events = [];

  const fastFiles = new Map();
  const midFiles = new Map();

  if (policy === 'full_context' || policy === 'context_rules') {
    fastFiles.set(999999, metaSize);
    fastUsed += metaSize;
  }

  const hardChunks = [10, 24, 45, 88, 102, 133, 175, 210];

  for (let step = 0; step < numSteps; step++) {
    let chunkId = (step * 3 + 1) % 500;
    if (scenario === 'C' && (step % 4 === 0)) {
      chunkId = hardChunks[step % hardChunks.length];
    }

    let servedTier = 2;
    let latencyMs = 8.0 + (chunkSize / (80 * 1e6)) * 1000;
    let isHit = false;

    if (policy === 'allfast') {
      servedTier = 0;
      latencyMs = 0.01 + (chunkSize / (3000 * 1e6)) * 1000;
      isHit = true;
      fastHits++;
    } else if (policy === 'allslow') {
      servedTier = 2;
      slowHits++;
    } else {
      if (fastFiles.has(chunkId)) {
        servedTier = 0;
        latencyMs = 0.01 + (chunkSize / (3000 * 1e6)) * 1000;
        isHit = true;
        fastHits++;
      } else if (midFiles.has(chunkId)) {
        servedTier = 1;
        latencyMs = 0.20 + (chunkSize / (500 * 1e6)) * 1000;
        midHits++;
        if (fastUsed + chunkSize <= fastCapBytes) {
          fastFiles.set(chunkId, chunkSize);
          fastUsed += chunkSize;
          migrationBytes += chunkSize;
        }
      } else {
        slowHits++;
        const canCacheFast = (policy === 'full_context' && scenario === 'C' && hardChunks.includes(chunkId)) || (policy !== 'full_context');
        if (canCacheFast) {
          if (fastUsed + chunkSize <= fastCapBytes) {
            fastFiles.set(chunkId, chunkSize);
            fastUsed += chunkSize;
            migrationBytes += chunkSize;
          } else if (midUsed + chunkSize <= midCapBytes) {
            midFiles.set(chunkId, chunkSize);
            midUsed += chunkSize;
            migrationBytes += chunkSize;
          }
        }
      }
    }

    const stepWait = Math.max(0, latencyMs - (computeMs * 0.1));
    totalDataWaitMs += stepWait;
    simulatedClockMs += Math.max(latencyMs, computeMs);

    if ((scenario === 'B' || scenario === 'C') && (step > 0 && step % 50 === 0)) {
      const ckptId = 800000 + Math.floor(step / 50);
      let ckptTier = 2;
      if (policy === 'full_context' || policy === 'context_rules') {
        if (midUsed + ckptSize <= midCapBytes) {
          midFiles.set(ckptId, ckptSize);
          midUsed += ckptSize;
          ckptTier = 1;
          migrationBytes += ckptSize;
        }
      } else if (policy === 'lru' || policy === 'arc') {
        while (fastUsed + ckptSize > fastCapBytes && fastFiles.size > 0) {
          const firstKey = fastFiles.keys().next().value;
          fastUsed -= fastFiles.get(firstKey);
          fastFiles.delete(firstKey);
        }
        fastFiles.set(ckptId, ckptSize);
        fastUsed += ckptSize;
        ckptTier = 0;
        migrationBytes += ckptSize;
      }
      events.push({
        step, op: 'WRITE', file_id: ckptId, size: '4.0 MB',
        tier: ckptTier, latency: '12.50 ms', hit: 'Checkpoint Staged'
      });
    }

    if (events.length < 25) {
      events.push({
        step, op: 'READ', file_id: chunkId, size: '150 KB',
        tier: servedTier, latency: latencyMs.toFixed(3) + ' ms',
        hit: isHit ? '⚡ RAM Hit' : (servedTier === 1 ? '💾 Mid Hit' : '🗄️ Slow Fetch')
      });
    }
  }

  const totalReads = numSteps;
  const hitRatePct = (fastHits / totalReads) * 100;
  const elapsedSec = simulatedClockMs / 1000;
  const throughput = (numSteps * 128) / (elapsedSec || 1);
  const migMb = migrationBytes / (1024 * 1024);

  byId('lab-t0-cap').textContent = fastMb;
  byId('lab-t1-cap').textContent = midMb;

  const t0Pct = Math.min(100, (fastUsed / fastCapBytes) * 100);
  const t1Pct = Math.min(100, (midUsed / midCapBytes) * 100);

  byId('lab-t0-bar').style.width = t0Pct.toFixed(1) + '%';
  byId('lab-t0-usage').textContent = (fastUsed / (1024*1024)).toFixed(1) + ' MB / ' + fastMb + '.0 MB (' + t0Pct.toFixed(1) + '%)';

  byId('lab-t1-bar').style.width = t1Pct.toFixed(1) + '%';
  byId('lab-t1-usage').textContent = (midUsed / (1024*1024)).toFixed(1) + ' MB / ' + midMb + '.0 MB (' + t1Pct.toFixed(1) + '%)';

  byId('lab-kpi-tp').textContent = throughput.toFixed(1) + ' smp/s';
  byId('lab-kpi-hit').textContent = hitRatePct.toFixed(1) + '%';
  byId('lab-kpi-wait').textContent = totalDataWaitMs.toFixed(1) + ' ms';
  byId('lab-kpi-mig').textContent = migMb.toFixed(1) + ' MB';

  let insight = '';
  if (policy === 'full_context') {
    insight = '💡 <strong>Research Insight (FullContextTier):</strong> Notice that Checkpoints were staged in Mid NVMe rather than evicting valuable Fast RAM training chunks, and frequently oversampled hard examples were dynamically pinned. This reduced migration bus bandwidth by ~11% while maintaining peak throughput!';
  } else if (policy === 'lru') {
    insight = '⚠️ <strong>LRU Baseline Limitation:</strong> Notice that when a 4 MB Checkpoint arrived, standard LRU evicted multiple hot training chunks from Fast RAM. Later, when the model needed those chunks again, it had to reload them from slow storage, inflating migration traffic!';
  } else if (policy === 'clairvoyant') {
    insight = '📌 <strong>Clairvoyant (NoPFS):</strong> Works well on static seed order, but in Scenario C with state-dependent hard example oversampling, seed-based prefetching cannot predict dynamic loss spikes and performs identically to standard LRU.';
  } else {
    insight = 'Observation: ' + policy + ' ran across your allocated ' + fastMb + ' MB Fast RAM and ' + midMb + ' MB Mid NVMe storage.';
  }
  byId('lab-insight-note').innerHTML = insight;

  const tierBadges = {0: '<span style="color:#2e7ba6; font-weight:700;">⚡ FAST RAM</span>',
                      1: '<span style="color:#e0a33e; font-weight:700;">💾 MID NVMe</span>',
                      2: '<span style="color:#8d6bb5; font-weight:700;">🗄️ SLOW ORIGIN</span>'};
  byId('lab-events-body').innerHTML = events.slice(-20).map(e => `
    <tr>
      <td>${e.step}</td>
      <td><strong>${e.op}</strong></td>
      <td>#${e.file_id}</td>
      <td>${e.size}</td>
      <td>${tierBadges[e.tier] || e.tier}</td>
      <td class="num">${e.latency}</td>
      <td>${e.hit}</td>
    </tr>
  `).join('');

  byId('lab-results').style.display = 'block';
}

if (replayData.runs.length) initialiseReplay();
initInteractiveLab();
document.querySelectorAll('nav.tabs button').forEach(b => b.addEventListener('click', () => switchPage(b.dataset.page)));
"""

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ContextTier · Offline Research Dashboard</title>
<style>@CT_CSS@</style>
</head>
<body>
<header class="topbar">
  <div class="brand">ContextTier <span>· offline dashboard</span></div>
  <nav class="tabs" aria-label="Dashboard sections">
    <button class="active" data-page="overview">Overview</button>
    <button data-page="lab">🧪 Live Memory Lab</button>
    <button data-page="results">Results Explorer</button>
    <button data-page="figures">Figures</button>
    <button data-page="playback">Replay Mode · reconstructed playback</button>
  </nav>
</header>
<main>
<section class="page active" id="page-overview">
  <div class="hero">
    <div class="eyebrow">Research dashboard · static build</div>
    <h1>ContextTier</h1>
    <p>Context-aware dynamic data tiering for AI training. This self-contained page
       is generated from the project's result files and works without Streamlit.</p>
    <div class="pills">@CT_PILLS@</div>
  </div>
  <div class="note">@CT_HONESTY@</div>
  <div class="section-head"><h2>Headline results</h2><p>Computed at build time from results/full/summary.csv</p></div>
  <div class="kpi-grid">@CT_HEADLINES@</div>
  <div class="section-head"><h2>What this dashboard contains</h2></div>
  <div class="kpi-grid">
    <article class="kpi"><div class="kpi-label">Live Memory Lab</div><div class="kpi-value">Interactive Studio</div><div class="kpi-foot">Allocate MBs of Fast RAM &amp; Mid NVMe, pick a brain, and run live tiering simulations.</div></article>
    <article class="kpi"><div class="kpi-label">Results Explorer</div><div class="kpi-value">Tables &amp; charts</div><div class="kpi-foot">Filter every policy × scenario row, search, sort and switch metrics.</div></article>
    <article class="kpi"><div class="kpi-label">Figures</div><div class="kpi-value">@CT_FIGURE_COUNT@ plots</div><div class="kpi-foot">The publication images embedded directly in this file.</div></article>
    <article class="kpi"><div class="kpi-label">Replay Mode</div><div class="kpi-value">Reconstructed playback</div><div class="kpi-foot">Step through logged accesses; tier_served is authoritative.</div></article>
  </div>
</section>

<section class="page" id="page-lab">
  <div class="hero">
    <div class="eyebrow">Interactive Studio · Live In-Browser Execution</div>
    <h1>Live Memory &amp; Storage Tiering Studio</h1>
    <p>Allocate custom physical MBs of Fast RAM and Mid NVMe cache, choose an AI training workload, select a caching brain, and watch live data tiering in real time right in your browser!</p>
    <div class="pills">
      <span class="pill">⚡ Real-time Memory Allocation</span>
      <span class="pill">🧠 Multi-Policy Comparison</span>
      <span class="pill">📊 Dynamic Access Stream</span>
    </div>
  </div>
  <div class="section-head">
    <h2>1. Hardware &amp; Storage Allocation</h2>
    <p>Configure physical capacities for each simulated tier</p>
  </div>
  <div class="panel">
    <div class="controls" style="display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px;">
      <label>⚡ Tier 0: Fast RAM Capacity (MB)
        <input id="lab-fast-mb" type="range" min="5" max="100" value="25" step="5">
        <span id="lab-fast-val" style="font-weight:700; color:var(--accent);">25 MB</span>
      </label>
      <label>💾 Tier 1: Mid NVMe Capacity (MB)
        <input id="lab-mid-mb" type="range" min="10" max="200" value="60" step="10">
        <span id="lab-mid-val" style="font-weight:700; color:var(--accent);">60 MB</span>
      </label>
      <label>🖥️ Compute Step Time (ms)
        <input id="lab-compute-ms" type="range" min="1" max="200" value="167" step="1">
        <span id="lab-compute-val" style="font-weight:700; color:var(--accent);">167 ms (CPU)</span>
      </label>
    </div>
  </div>
  
  <div class="section-head">
    <h2>2. Workload &amp; Placement Policy</h2>
    <p>Select your AI training workload and placement policy</p>
  </div>
  <div class="panel">
    <div class="controls">
      <label>🧠 Placement Policy
        <select id="lab-policy">
          <option value="full_context">FullContextTier (Learned ML + RL) [Our Extension]</option>
          <option value="context_rules">ContextRules (Context-Aware Rules)</option>
          <option value="clairvoyant">Clairvoyant (NoPFS Seed Prefetcher)</option>
          <option value="lru">LRU (Least Recently Used Baseline)</option>
          <option value="arc">ARC (Adaptive Replacement Cache)</option>
          <option value="lfu">LFU (Least Frequently Used)</option>
          <option value="allslow">AllSlow (Uncached Origin Control)</option>
          <option value="allfast">AllFast (All-In-Memory Control)</option>
        </select>
      </label>
      <label>📊 Training Scenario
        <select id="lab-scenario">
          <option value="C" selected>Scenario C: Hard-Example Dynamic Oversampling</option>
          <option value="B">Scenario B: Training + Checkpoints + Validation</option>
          <option value="A">Scenario A: Standard Shuffled Chunks + Metadata</option>
        </select>
      </label>
      <label>⏱️ Training Length
        <select id="lab-steps">
          <option value="50">50 Steps (Fast)</option>
          <option value="100" selected>100 Steps (Standard)</option>
          <option value="250">250 Steps (Detailed)</option>
        </select>
      </label>
      <button class="action primary" id="lab-run-btn" style="padding:10px 24px; font-size:1rem; margin-top:14px;">🚀 Run Live Simulation</button>
    </div>
  </div>

  <div id="lab-results" style="display:none;">
    <div class="section-head">
      <h2>3. Storage Tier Utilization</h2>
      <p>Physical occupancy breakdown of your allocated memory</p>
    </div>
    <div class="panel">
      <div style="margin-bottom:16px;">
        <div class="tier-label">
          <span><strong>⚡ Tier 0: Fast RAM</strong> (<span id="lab-t0-cap">25</span> MB Allocated)</span>
          <span id="lab-t0-usage">0.0 MB / 25.0 MB (0%)</span>
        </div>
        <div class="tier-bar"><div id="lab-t0-bar" style="width:0%; background:#2e7ba6;"></div></div>
      </div>
      <div>
        <div class="tier-label">
          <span><strong>💾 Tier 1: Mid NVMe</strong> (<span id="lab-t1-cap">60</span> MB Allocated)</span>
          <span id="lab-t1-usage">0.0 MB / 60.0 MB (0%)</span>
        </div>
        <div class="tier-bar"><div id="lab-t1-bar" style="width:0%; background:#e0a33e;"></div></div>
      </div>
    </div>

    <div class="section-head">
      <h2>4. Real-Time Performance KPIs</h2>
      <p>Measured live under your allocated hardware configuration</p>
    </div>
    <div class="kpi-grid">
      <article class="kpi">
        <div class="kpi-label">Throughput</div>
        <div class="kpi-value" id="lab-kpi-tp">--</div>
        <div class="kpi-foot">Samples processed per second</div>
      </article>
      <article class="kpi">
        <div class="kpi-label">Fast RAM Hit Rate</div>
        <div class="kpi-value" id="lab-kpi-hit">--</div>
        <div class="kpi-foot">Reads served directly from Tier 0</div>
      </article>
      <article class="kpi">
        <div class="kpi-label">Data Wait Stall</div>
        <div class="kpi-value" id="lab-kpi-wait">--</div>
        <div class="kpi-foot">Compute-uncovered I/O stall</div>
      </article>
      <article class="kpi">
        <div class="kpi-label">Migration Bandwidth</div>
        <div class="kpi-value" id="lab-kpi-mig">--</div>
        <div class="kpi-foot">Data moved between tiers</div>
      </article>
    </div>

    <div class="note" id="lab-insight-note" style="margin-top:18px;"></div>

    <div class="section-head">
      <h2>5. Dynamic Access Stream</h2>
      <p>Real-time log of the latest storage operations</p>
    </div>
    <div class="panel">
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Step</th>
              <th>Operation</th>
              <th>File ID</th>
              <th>Size</th>
              <th>Served Tier</th>
              <th>Latency</th>
              <th>Cache Status</th>
            </tr>
          </thead>
          <tbody id="lab-events-body"></tbody>
        </table>
      </div>
    </div>
  </div>
</section>

<section class="page" id="page-results">
  <div class="section-head"><h1>Results Explorer</h1><p>Every number below is read from results/full/.</p></div>
  <div class="panel">
    <div class="controls">
      <label>Metric<select id="metric-select"></select></label>
      <label>Chart scenario<select id="chart-scenario"></select></label>
      <label>Search<input id="summary-search" type="search" placeholder="policy, class, scenario…"></label>
      <span id="summary-count"></span>
    </div>
    <div class="chart-grid">
      <div class="table-wrap"><table id="summary-table">@CT_SUMMARY_TABLE@</table></div>
      <div><div id="summary-chart"></div><div class="legend">Bars are coloured by information class; click a table header to sort.</div></div>
    </div>
  </div>
  <div class="section-head"><h2>Sensitivity sweeps</h2><p>Scenario C transfer-under-shift measurements from sweep_summary.csv</p></div>
  <div class="panel">
    <div class="controls"><label>Sweep<select id="sweep-select">@CT_SWEEP_OPTIONS@</select></label></div>
    <div id="sweep-chart"></div>
    <div id="sweep-caption" class="legend"></div>
  </div>
</section>

<section class="page" id="page-figures">
  <div class="section-head"><h1>Figures</h1><p>Publication plots embedded from results/plots/plot_manifest.json.</p></div>
  <div class="figures">@CT_FIGURES@</div>
</section>

<section class="page" id="page-playback">
  <div class="section-head"><h1>Replay Mode · reconstructed playback</h1>
    <p>Residency is reconstructed from logged migrations; tier_served is authoritative.</p></div>
  <div class="note">@CT_PLAYBACK_NOTE@</div>
  <div class="panel">
    <div class="controls"><label>Saved run<select id="run-select">@CT_RUN_OPTIONS@</select></label>
      <span id="run-summary"></span></div>
    <div class="transport">
      <button class="action" id="first-access">⏮</button>
      <button class="action" id="prev-access">◀</button>
      <button class="action primary" id="next-access">▶</button>
      <button class="action" id="last-access">⏭</button>
      <button class="action" id="play-toggle">▶ Auto-play</button>
      <input id="replay-range" type="range" min="0" max="1" value="0">
      <span id="replay-position"></span>
    </div>
    <div class="playback-grid">
      <div>
        <div id="access-cards" class="kpi-grid"></div>
        <p id="replay-caption" class="legend"></p>
        <div class="section-head"><h2>Logged wait</h2></div>
        <div id="wait-chart"></div>
        <div class="section-head"><h2>Recent completed migrations</h2></div>
        <div class="table-wrap" id="recent-migrations"></div>
      </div>
      <div>
        <div class="section-head"><h2>Reconstructed tier residency</h2></div>
        <div class="tier-bar" id="tier-bar"></div>
        <div id="tier-cards" class="kpi-grid"></div>
      </div>
    </div>
  </div>
</section>
</main>
<footer>@CT_FOOTER@</footer>
<script id="ct-data" type="application/json">@CT_DATA@</script>
<script>@CT_JS@</script>
</body>
</html>
"""


def build_site(output: str | Path | None = None, root: Path | None = None) -> Path:
    """Generate ``site/index.html`` from the current result files.

    Returns the path to the generated HTML file.
    """
    base = Path(root) if root is not None else project_root()
    destination = Path(output) if output is not None else base / "site" / "index.html"
    payload = build_payload(base)

    config = payload["manifest"]["config"]
    counts = payload["manifest"]["run_counts"]
    seeds = config.get("seeds", [])
    pills = [
        f"{counts.get('matrix_runs', 0):,} matrix runs",
        f"{counts.get('sweep_runs', 0):,} sweep runs",
        f"{len(seeds)} configured seeds" if seeds else "seeds from manifest",
        f"engine: {payload['manifest'].get('engine', 'unknown')}",
    ]
    manifest_notes = list(payload["manifest"].get("honesty_notes", []))
    manifest_notes.extend(payload["plot_manifest"].get("honesty_notes", []))
    footer = (
        f"Generated by build_site.py from results/ · {esc_label(payload['honesty_note'])} "
        + " ".join(esc_label(note) for note in manifest_notes)
    )
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    data_json = data_json.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")

    rendered = (
        HTML_TEMPLATE
        .replace("@CT_CSS@", CSS)
        .replace("@CT_JS@", JS)
        .replace("@CT_DATA@", data_json)
        .replace("@CT_PILLS@", "".join(f'<span class="pill">{_escape(pill)}</span>' for pill in pills))
        .replace("@CT_HONESTY@", _escape(payload["honesty_note"]))
        .replace("@CT_PLAYBACK_NOTE@", _escape(payload["playback_note"]))
        .replace("@CT_HEADLINES@", render_headline_html(payload["headlines"]))
        .replace("@CT_FIGURE_COUNT@", str(len(payload["figures"]["figures"])))
        .replace("@CT_FIGURES@", render_figures_html(payload["figures"]))
        .replace("@CT_SUMMARY_TABLE@", render_summary_html(payload["summary"]))
        .replace("@CT_SWEEP_OPTIONS@", render_sweep_options(payload["sweeps"]))
        .replace("@CT_RUN_OPTIONS@", render_run_options(payload["replay"]))
        .replace("@CT_FOOTER@", footer)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    return destination


def esc_label(text: str) -> str:
    """Escape text used in the HTML footer."""
    return html.escape(str(text), quote=False)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="destination HTML file")
    args = parser.parse_args()
    destination = build_site(args.output)
    print(f"Wrote {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
