"""Streamlit-free loaders for Phase 8 result tables, plots and saved traces.

Every number shown by the website comes from a file written by
``experiments.py`` / ``plots.py`` / ``run_experiments.py``. This module contains
no hard-coded experimental values and no Streamlit imports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from config import project_root

RESULTS_DIR: Path = project_root() / "results"
FULL_DIR: Path = RESULTS_DIR / "full"
PLOTS_DIR: Path = RESULTS_DIR / "plots"

RESULT_TABLES: tuple[str, ...] = (
    "raw_runs.csv",
    "summary.csv",
    "sweep_runs.csv",
    "sweep_summary.csv",
)

SCENARIOS: dict[str, str] = {
    "A": "Shuffled training chunks + metadata.",
    "B": "Scenario A + checkpoints + validation reads.",
    "C": "Scenario C = B + hard-example oversampling (access order depends on training state; no seed clairvoyance).",
    "D": "Scenario B with every file pre-resident in Tier 0 (designed tie).",
    "E": "Synthetic non-uniform hot/warm/cold sampling (state-dependent stress test).",
    "F": "Scenario E with a rotating hot set (temperature-drift stress test).",
}

KIND_LABELS: dict[str, str] = {
    "observable_baseline": "Observable baseline",
    "learned_observable": "Learned, observable inputs",
    "declared_type_upper_bound": "Declared-type upper bound",
    "oracle_upper_bound": "Oracle upper bound",
}

TIER_NAMES: dict[int, str] = {0: "FAST", 1: "MID", 2: "SLOW"}

DISPLAY_NAMES: dict[str, str] = {
    "allslow": "AllSlow",
    "allfast": "AllFast",
    "belady": "Belady (oracle)",
    "clairvoyant": "Clairvoyant",
    "lru": "LRU",
    "lfu": "LFU",
    "arc": "ARC",
    "context_rules_noctx": "ContextRules (no ctx)",
    "context_rules": "ContextRules",
    "classifier": "Classifier",
    "embedding": "Embedding",
    "full_context": "FullContextTier",
}

METRIC_LABELS: dict[str, str] = {
    "data_wait_ms": "Data wait (ms)",
    "data_wait_pct": "Data wait (% of elapsed)",
    "throughput": "Throughput (samples/s)",
    "fast_hit_rate": "FAST hit rate",
    "mid_hit_rate": "MID hit rate",
    "slow_hit_rate": "SLOW hit rate",
    "migration_bytes": "Migration bytes",
    "wasted_prefetch_bytes": "Wasted prefetch bytes",
    "latency_mean_ms": "Mean read latency (ms)",
    "latency_p95_ms": "p95 read latency (ms)",
    "latency_p99_ms": "p99 read latency (ms)",
}

HONESTY_NOTE: str = (
    "This is a laptop-scale CPU emulation with scaled-down tiers, not a "
    "production storage stack. We re-implement seed-based exact prefetching "
    "and call it NoPFS-style; we do not run the original NoPFS code. All "
    "numbers on this site are read from the project's own result files."
)


def load_tables(directory: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load the four Phase 8 CSV tables from ``results/full``."""
    base = Path(directory) if directory is not None else FULL_DIR
    missing = [name for name in RESULT_TABLES if not (base / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing result tables in {base}: {', '.join(missing)}. "
            "Run `python experiments.py` first."
        )
    return {name[:-4]: pd.read_csv(base / name) for name in RESULT_TABLES}


def load_bootstrap(directory: str | Path | None = None) -> pd.DataFrame:
    """Load the paired-bootstrap comparison table, or an empty frame.

    The table is optional: it exists only after ``paper_stats.py`` has run.
    """
    base = Path(directory) if directory is not None else FULL_DIR
    path = base / "bootstrap_comparison.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def compute_benchmark(results_dir: str | Path | None = None) -> dict[str, Any]:
    """Load the measured CPU step-time benchmark, or an empty mapping."""
    base = Path(results_dir) if results_dir is not None else RESULTS_DIR
    path = base / "real" / "step_time_benchmark.json"
    if not path.is_file():
        return {}
    return load_json(path)


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON manifest and require a mapping at the root."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def experiment_manifest(directory: str | Path | None = None) -> dict[str, Any]:
    """Load ``experiment_manifest.json`` from the Phase 8 results folder."""
    base = Path(directory) if directory is not None else FULL_DIR
    return load_json(base / "experiment_manifest.json")


def plot_manifest(directory: str | Path | None = None) -> dict[str, Any]:
    """Load ``plot_manifest.json`` from the Phase 9 figures folder."""
    base = Path(directory) if directory is not None else PLOTS_DIR
    return load_json(base / "plot_manifest.json")


def figure_catalog(directory: str | Path | None = None) -> list[dict[str, Any]]:
    """Return one record per Phase 9 figure with its PNG/PDF paths.

    Records are ordered by the plot manifest, which is the order used in the
    paper. Paths are returned as absolute :class:`~pathlib.Path` objects.
    """
    base = Path(directory) if directory is not None else PLOTS_DIR
    manifest = plot_manifest(base)
    records: list[dict[str, Any]] = []
    for stem, paths in manifest.get("figures", {}).items():
        available = [Path(item) for item in paths]
        png = next((path for path in available if path.suffix == ".png"), None)
        pdf = next((path for path in available if path.suffix == ".pdf"), None)
        records.append(
            {
                "stem": stem,
                "title": stem.replace("_", " ").title(),
                "png": png,
                "pdf": pdf,
                "paths": available,
            }
        )
    return records


@dataclass(frozen=True)
class ReplayRun:
    """A saved virtual- or real-mode run that can be replayed in the UI."""

    label: str
    trace_path: Path
    migrations_path: Path | None
    metrics_path: Path | None
    engine: str
    scenario: str
    policy: str
    seed: int

    @property
    def display_name(self) -> str:
        """Short human-readable run identifier."""
        return f"{self.engine} / {self.scenario} / {self.policy} / seed {self.seed}"


def discover_replays(results_dir: str | Path | None = None) -> list[ReplayRun]:
    """Find every ``*_trace.csv`` and pair it with its auxiliary files.

    Sorting puts virtual-mode runs first, then orders by scenario, policy and
    seed, so the default selection is deterministic across machines.
    """
    base = Path(results_dir) if results_dir is not None else RESULTS_DIR
    runs: list[ReplayRun] = []
    for trace_path in sorted(base.rglob("*_trace.csv")):
        stem = trace_path.name[: -len("_trace.csv")]
        migrations_path = trace_path.with_name(f"{stem}_migrations.csv")
        metrics_path = trace_path.with_name(f"{stem}_metrics.csv")
        parts = trace_path.relative_to(base).parts
        engine = "real" if "real" in parts else "virtual"

        # Saved runs live in `results/<scenario>/<policy>_seed<N>_trace.csv`
        # (or `results/real/<scenario>/...`). The scenario is a directory name,
        # while the policy id may itself contain underscores (full_context).
        parent_name = trace_path.parent.name
        scenario = parent_name if parent_name in SCENARIOS else "?"
        policy = stem
        seed = -1
        if "_seed" in stem:
            policy, seed_text = stem.rsplit("_seed", 1)
            seed = int(seed_text) if seed_text.isdigit() else -1
        runs.append(
            ReplayRun(
                label=trace_path.relative_to(base).as_posix(),
                trace_path=trace_path,
                migrations_path=migrations_path if migrations_path.is_file() else None,
                metrics_path=metrics_path if metrics_path.is_file() else None,
                engine=engine,
                scenario=scenario,
                policy=policy,
                seed=seed,
            )
        )
    runs.sort(key=lambda run: (run.engine != "virtual", run.scenario, run.policy, run.seed))
    return runs


def load_replay(run: ReplayRun) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a run's access trace and migration log (empty frame if absent)."""
    trace = pd.read_csv(run.trace_path)
    # Phase-13 traces add evaluation-only true_temp. Preserve playback of saved
    # A-D artefacts written before that additive schema change.
    if "true_temp" not in trace.columns:
        trace["true_temp"] = "unknown"
    if run.migrations_path is not None:
        migrations = pd.read_csv(run.migrations_path)
    else:
        migrations = pd.DataFrame(
            columns=[
                "ts",
                "run_id",
                "seed",
                "scenario",
                "policy",
                "file_id",
                "size_bytes",
                "src_tier",
                "dst_tier",
                "kind",
            ]
        )
    return trace, migrations


def load_run_metrics(run: ReplayRun) -> dict[str, float]:
    """Load the scalar metrics recorded for one run, if available."""
    if run.metrics_path is None:
        return {}
    frame = pd.read_csv(run.metrics_path)
    return {
        str(row["name"]): float(row["value"])
        for _, row in frame.iterrows()
    }


def format_metric(metric: str, value: float) -> str:
    """Format a metric for display without changing its underlying value."""
    if metric.endswith("_bytes") or metric in {"migration_bytes", "wasted_prefetch_bytes"}:
        return f"{value / (1024 * 1024):,.2f} MB"
    if metric in {"fast_hit_rate", "mid_hit_rate", "slow_hit_rate"}:
        return f"{value:.3f}"
    if metric == "data_wait_pct":
        return f"{value:.2f}%"
    return f"{value:,.2f}"


def mean_std_text(mean: float, std: float, metric: str) -> str:
    """Render a mean ± std cell using the metric's display unit."""
    if metric in {"fast_hit_rate", "mid_hit_rate", "slow_hit_rate"}:
        return f"{mean:.3f} ± {std:.3f}"
    if metric in {"migration_bytes", "wasted_prefetch_bytes"}:
        return f"{mean / (1024 * 1024):,.1f} ± {std / (1024 * 1024):,.1f} MB"
    return f"{mean:,.2f} ± {std:,.2f}"


def scenario_note(scenario: str) -> str:
    """Return the fixed scenario description from the design contract."""
    return SCENARIOS.get(scenario, "Unknown scenario.")


def kind_label(kind: str) -> str:
    """Return the human-readable information class of a policy."""
    return KIND_LABELS.get(kind, kind)


def policy_name(policy: str) -> str:
    """Return the display name of a policy id."""
    return DISPLAY_NAMES.get(policy, policy)


def finite_values(values: Iterable[Any]) -> list[float]:
    """Coerce values to floats, dropping NaN and infinite entries."""
    result: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number and number not in (float("inf"), float("-inf")):
            result.append(number)
    return result


def tier_bytes(occupancy: Mapping[int, int]) -> str:
    """Format a tier occupancy mapping as a compact string."""
    return ", ".join(
        f"{TIER_NAMES.get(tier, str(tier))}: {bytes_ / (1024 * 1024):.2f} MB"
        for tier, bytes_ in sorted(occupancy.items())
    )
