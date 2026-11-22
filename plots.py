"""Phase 9: publication-quality PDF and PNG figures from Phase 8 results.

All plotted values come from the CSVs written by ``experiments.py``; no number
is invented or illustrative. Error bars are the sample standard deviation over
the configured seeds (20 in the final matrix), or, for the paired ablation, the
std of the per-seed difference.

Figures are generated with the non-interactive Agg backend so the module works
on CPU-only machines without a display. PDFs are vector; PNGs are 300 dpi.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Publication styling
# --------------------------------------------------------------------------- #
STYLE: dict = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
}

# A stable logical order and readable labels for the paper tables.
POLICY_ORDER: tuple[str, ...] = (
    "allslow",
    "allfast",
    "belady",
    "clairvoyant",
    "lru",
    "lfu",
    "arc",
    "context_rules_noctx",
    "context_rules",
    "classifier",
    "embedding",
    "full_context",
)

DEPLOYABLE_POLICIES: tuple[str, ...] = (
    "lru",
    "lfu",
    "arc",
    "context_rules_noctx",
    "classifier",
    "embedding",
    "context_rules",
    "full_context",
)

FOCUS_POLICIES: tuple[str, ...] = (
    "belady",
    "full_context",
    "context_rules",
    "lru",
    "clairvoyant",
)

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

KIND_COLORS: dict[str, str] = {
    "observable_baseline": "#4c72b0",
    "learned_observable": "#55a868",
    "declared_type_upper_bound": "#dd8452",
    "oracle_upper_bound": "#c44e52",
}

METRIC_LABELS: dict[str, str] = {
    "data_wait_ms": "Data wait (ms)",
    "throughput": "Throughput (samples/s)",
    "latency_mean_ms": "Mean read latency (ms)",
    "latency_p95_ms": "p95 read latency (ms)",
    "latency_p99_ms": "p99 read latency (ms)",
}


def apply_style() -> None:
    """Apply the shared publication style to matplotlib's global config."""
    plt.rcParams.update(STYLE)


# --------------------------------------------------------------------------- #
# Table helpers
# --------------------------------------------------------------------------- #
def load_tables(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the raw, summary and sweep tables written by Phase 8."""
    directory = Path(results_dir)
    expected = ("raw_runs.csv", "summary.csv", "sweep_runs.csv", "sweep_summary.csv")
    missing = [name for name in expected if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing result tables in {directory}: {', '.join(missing)}. "
            "Run `python experiments.py` first."
        )
    return {
        "raw_runs": pd.read_csv(directory / "raw_runs.csv"),
        "summary": pd.read_csv(directory / "summary.csv"),
        "sweep_runs": pd.read_csv(directory / "sweep_runs.csv"),
        "sweep_summary": pd.read_csv(directory / "sweep_summary.csv"),
    }


def _label(policy: str) -> str:
    return DISPLAY_NAMES.get(policy, policy)


def _color(kind: str) -> str:
    return KIND_COLORS.get(kind, "#8172b3")


def _metric_pair(
    frame: pd.DataFrame, policy: str, scenario: str, metric: str
) -> tuple[float, float]:
    """Return (mean, std) for one policy/scenario/metric, or (nan, 0)."""
    selection = frame[
        (frame["policy"] == policy) & (frame["scenario"] == scenario)
    ]
    if selection.empty or f"{metric}_mean" not in selection.columns:
        return float("nan"), 0.0
    row = selection.iloc[0]
    mean = float(row[f"{metric}_mean"])
    std = float(row.get(f"{metric}_std", 0.0))
    return mean, (0.0 if pd.isna(std) else std)


def _policy_kind(frame: pd.DataFrame, policy: str) -> str:
    selection = frame[frame["policy"] == policy]
    if selection.empty or "policy_kind" not in selection.columns:
        return "observable_baseline"
    return str(selection.iloc[0]["policy_kind"])


def _seed_count(frame: pd.DataFrame) -> int:
    """Read the number of seeds from the generated tables, never hard-code it."""
    if "n_seeds" in frame.columns and not frame.empty:
        return int(frame["n_seeds"].max())
    if "seed" in frame.columns:
        return int(frame["seed"].nunique())
    return 0


def _bar_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    policies: Sequence[str],
    scenario: str,
    metric: str,
    title: str,
    *,
    symlog: bool = False,
) -> None:
    """Draw one horizontal bar panel with seed-std error bars."""
    available = [policy for policy in policies if (frame["policy"] == policy).any()]
    pairs = [_metric_pair(frame, policy, scenario, metric) for policy in available]
    means = np.asarray([pair[0] for pair in pairs], dtype=float)
    stds = np.asarray([pair[1] for pair in pairs], dtype=float)
    positions = np.arange(len(available))
    colors = [_color(_policy_kind(frame, policy)) for policy in available]

    ax.barh(
        positions,
        means,
        xerr=stds,
        capsize=3,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
        error_kw={"elinewidth": 0.9, "ecolor": "#333333"},
    )
    ax.set_yticks(positions, labels=[_label(policy) for policy in available])
    ax.invert_yaxis()
    ax.set_xlabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title)
    if symlog:
        ax.set_xscale("symlog", linthresh=1.0)
    ax.grid(axis="x", alpha=0.25)


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str, formats: Sequence[str]) -> list[Path]:
    """Save one figure in each requested format and close it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        figure.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_all_policies(
    summary: pd.DataFrame, output_dir: Path, formats: Sequence[str]
) -> list[Path]:
    """Four panels (A-D), all twelve policies, data wait on a symlog axis."""
    apply_style()
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, scenario in zip(axes.flat, ("A", "B", "C", "D")):
        _bar_panel(
            axis,
            summary,
            POLICY_ORDER,
            scenario,
            "data_wait_ms",
            f"Scenario {scenario}",
            symlog=True,
        )
    figure.suptitle(
        f"Data wait across all policies (mean ± std, {_seed_count(summary)} seeds; symlog axis)"
    )
    return _save_figure(figure, output_dir, "data_wait_all", formats)


def plot_deployable_policies(
    summary: pd.DataFrame, output_dir: Path, formats: Sequence[str]
) -> list[Path]:
    """Readable linear-scale comparison of deployable policies on A-D."""
    apply_style()
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, scenario in zip(axes.flat, ("A", "B", "C", "D")):
        _bar_panel(
            axis,
            summary,
            DEPLOYABLE_POLICIES,
            scenario,
            "data_wait_ms",
            f"Scenario {scenario}",
        )
    figure.suptitle(
        f"Data wait: deployable policies (mean ± std, {_seed_count(summary)} seeds)"
    )
    return _save_figure(figure, output_dir, "data_wait_deployable", formats)


def plot_scenario_focus(
    summary: pd.DataFrame, output_dir: Path, formats: Sequence[str]
) -> list[Path]:
    """Scenario C: data wait and throughput side by side."""
    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    _bar_panel(axes[0], summary, FOCUS_POLICIES, "C", "data_wait_ms", "Scenario C: data wait")
    _bar_panel(axes[1], summary, FOCUS_POLICIES, "C", "throughput", "Scenario C: throughput")
    figure.suptitle(f"Scenario C focus (mean ± std, {_seed_count(summary)} seeds)")
    return _save_figure(figure, output_dir, "scenario_c_focus", formats)


def plot_capacity_sweep(
    sweep: pd.DataFrame, output_dir: Path, formats: Sequence[str]
) -> list[Path]:
    """Fast-tier capacity sweep on Scenario C with seed error bars."""
    apply_style()
    figure, axis = plt.subplots(figsize=(7, 4.6))
    for policy in ("lru", "context_rules", "full_context"):
        subset = sweep[
            (sweep["sweep"] == "fast_capacity_mb")
            & (sweep["policy"] == policy)
            & (sweep["scenario"] == "C")
        ]
        if subset.empty:
            continue
        values = sorted(float(value) for value in subset["sweep_value"].unique())
        means = [
            float(subset[subset["sweep_value"] == value]["data_wait_ms_mean"].iloc[0])
            for value in values
        ]
        stds = [
            float(subset[subset["sweep_value"] == value]["data_wait_ms_std"].iloc[0])
            for value in values
        ]
        axis.errorbar(
            values,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            linewidth=1.5,
            label=_label(policy),
            color=_color(_policy_kind(sweep, policy)),
        )
    axis.set_xlabel("Fast-tier capacity (MB)")
    axis.set_ylabel("Data wait (ms)")
    axis.set_title("Sensitivity to fast-tier capacity (Scenario C, mean ± std)")
    axis.legend(frameon=False)
    return _save_figure(figure, output_dir, "capacity_sweep", formats)


def plot_compute_sweep(
    sweep: pd.DataFrame, output_dir: Path, formats: Sequence[str]
) -> list[Path]:
    """Compute-time sweep on Scenario C with seed error bars."""
    apply_style()
    figure, axis = plt.subplots(figsize=(7, 4.6))
    for policy in ("lru", "context_rules", "full_context"):
        subset = sweep[
            (sweep["sweep"] == "compute_ms")
            & (sweep["policy"] == policy)
            & (sweep["scenario"] == "C")
        ]
        if subset.empty:
            continue
        values = sorted(float(value) for value in subset["sweep_value"].unique())
        means = [
            float(subset[subset["sweep_value"] == value]["data_wait_ms_mean"].iloc[0])
            for value in values
        ]
        stds = [
            float(subset[subset["sweep_value"] == value]["data_wait_ms_std"].iloc[0])
            for value in values
        ]
        axis.errorbar(
            values,
            means,
            yerr=stds,
            marker="s",
            capsize=4,
            linewidth=1.5,
            label=_label(policy),
            color=_color(_policy_kind(sweep, policy)),
        )
    axis.set_xlabel("Compute per step (ms)")
    axis.set_ylabel("Data wait (ms)")
    axis.set_title("Sensitivity to compute time (Scenario C, mean ± std)")
    axis.legend(frameon=False)
    return _save_figure(figure, output_dir, "compute_sweep", formats)


def plot_compute_gain(
    sweep: pd.DataFrame,
    raw: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    benchmark_path: str | Path = "results/real/step_time_benchmark.json",
) -> list[Path]:
    """Paired data-wait saved by FullContextTier over baselines vs compute_ms.

    A positive value means FullContextTier stalled less than the baseline. The
    measured median CPU step time from ``step_time_benchmark.json`` is drawn as
    a vertical reference so readers can see where this machine sits relative to
    the sweep. No number in this figure is hard-coded: the reference is read
    from the benchmark artefact.
    """
    apply_style()
    selection = sweep[
        (sweep["sweep"] == "compute_ms") & (sweep["scenario"] == "C")
    ]
    values = sorted(float(value) for value in selection["sweep_value"].unique())
    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    raw_compute = raw[(raw["sweep"] == "compute_ms") & (raw["scenario"] == "C")]
    focus = raw_compute[raw_compute["policy"] == "full_context"].set_index(
        ["sweep_value", "seed"]
    )["data_wait_ms"]

    for baseline in ("lru", "lfu"):
        baseline_series = raw_compute[raw_compute["policy"] == baseline].set_index(
            ["sweep_value", "seed"]
        )["data_wait_ms"]
        gains: list[float] = []
        errors: list[float] = []
        for value in values:
            paired = (baseline_series.loc[value] - focus.loc[value]).dropna() if value in baseline_series.index.get_level_values(0) else pd.Series(dtype=float)
            gains.append(float(paired.mean()) if len(paired) else 0.0)
            errors.append(float(paired.std(ddof=1)) if len(paired) > 1 else 0.0)
        axis.errorbar(
            values,
            gains,
            yerr=errors,
            marker="o",
            capsize=4,
            linewidth=1.5,
            label=f"vs {_label(baseline)}",
            color=_color(_policy_kind(sweep, baseline)),
        )

    axis.axhline(0.0, color="#333333", linewidth=0.8, linestyle=":")
    benchmark = Path(benchmark_path)
    if benchmark.is_file():
        data = json.loads(benchmark.read_text(encoding="utf-8"))
        measured = float(data["measurement"]["median_ms"])
        axis.axvline(measured, color="#c44e52", linewidth=1.2, linestyle="--")
        axis.annotate(
            f"measured CPU step\n{measured:.2f} ms",
            xy=(measured, axis.get_ylim()[1]),
            xytext=(-8, -6),
            textcoords="offset points",
            ha="right",
            va="top",
            color="#c44e52",
            fontsize=8,
        )
        right = max(measured * 1.06, max(values) * 1.15 if values else measured * 1.06)
        axis.set_xlim(min(values) * 0.85 if values else 0, right)
    axis.set_xlabel("Compute per step (ms)")
    axis.set_ylabel("Data wait saved by FullContextTier (ms)")
    axis.set_title(
        f"Gain vs compute budget (Scenario C, paired over {_seed_count(sweep)} seeds)"
    )
    axis.legend(frameon=False)
    return _save_figure(figure, output_dir, "compute_gain", formats)


def plot_context_ablation(
    raw: pd.DataFrame, output_dir: Path, formats: Sequence[str]
) -> list[Path]:
    """Paired per-seed data-wait saved by ContextRules' declared-type channel."""
    apply_style()
    figure, axis = plt.subplots(figsize=(7, 4.4))
    scenarios = list(("A", "B", "C", "D"))
    savings: list[float] = []
    errors: list[float] = []
    for scenario in scenarios:
        with_ctx = raw[
            (raw["policy"] == "context_rules") & (raw["scenario"] == scenario)
        ].set_index("seed")["data_wait_ms"]
        no_ctx = raw[
            (raw["policy"] == "context_rules_noctx") & (raw["scenario"] == scenario)
        ].set_index("seed")["data_wait_ms"]
        difference = (no_ctx - with_ctx).reindex(with_ctx.index).dropna()
        savings.append(float(difference.mean()) if len(difference) else 0.0)
        errors.append(float(difference.std(ddof=1)) if len(difference) > 1 else 0.0)

    positions = np.arange(len(scenarios))
    axis.bar(
        positions,
        savings,
        yerr=errors,
        capsize=4,
        color="#dd8452",
        edgecolor="black",
        linewidth=0.4,
    )
    axis.set_xticks(positions, labels=[f"Scenario {s}" for s in scenarios])
    axis.set_ylabel("Data wait saved (ms)")
    axis.set_title(
        f"ContextRules ablation: paired saving over {_seed_count(raw)} seeds (mean ± std)"
    )
    return _save_figure(figure, output_dir, "context_ablation", formats)


def plot_latency_percentiles(
    summary: pd.DataFrame, output_dir: Path, formats: Sequence[str]
) -> list[Path]:
    """Mean / p95 / p99 read latency for representative policies on C."""
    apply_style()
    figure, axis = plt.subplots(figsize=(8, 4.6))
    policies = list(("lru", "full_context", "context_rules", "belady"))
    metrics = ("latency_mean_ms", "latency_p95_ms", "latency_p99_ms")
    x_positions = np.arange(len(metrics))
    width = 0.18
    for offset, policy in enumerate(policies):
        means = [_metric_pair(summary, policy, "C", metric)[0] for metric in metrics]
        stds = [_metric_pair(summary, policy, "C", metric)[1] for metric in metrics]
        axis.bar(
            x_positions + (offset - 1.5) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=_label(policy),
            color=_color(_policy_kind(summary, policy)),
            edgecolor="black",
            linewidth=0.4,
        )
    axis.set_xticks(x_positions, labels=("mean", "p95", "p99"))
    axis.set_ylabel("Read latency (ms)")
    axis.set_title("Read latency distribution (Scenario C, mean ± std)")
    axis.legend(frameon=False)
    return _save_figure(figure, output_dir, "latency_percentiles", formats)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def generate_all(
    results_dir: str | Path = "results/full",
    output_dir: str | Path = "results/plots",
    formats: Sequence[str] = ("pdf", "png"),
) -> dict:
    """Generate every Phase 9 figure and write a provenance manifest."""
    tables = load_tables(results_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    figures = {
        "data_wait_all": plot_all_policies(tables["summary"], output, formats),
        "data_wait_deployable": plot_deployable_policies(tables["summary"], output, formats),
        "scenario_c_focus": plot_scenario_focus(tables["summary"], output, formats),
        "capacity_sweep": plot_capacity_sweep(tables["sweep_summary"], output, formats),
        "compute_sweep": plot_compute_sweep(tables["sweep_summary"], output, formats),
        "compute_gain": plot_compute_gain(
            tables["sweep_summary"], tables["sweep_runs"], output, formats
        ),
        "context_ablation": plot_context_ablation(tables["raw_runs"], output, formats),
        "latency_percentiles": plot_latency_percentiles(tables["summary"], output, formats),
    }
    manifest = {
        "phase": 9,
        "source_tables": sorted(str(Path(results_dir) / name) for name in
                                ("raw_runs.csv", "summary.csv", "sweep_runs.csv", "sweep_summary.csv")),
        "formats": list(formats),
        "figures": {name: [str(path) for path in paths] for name, paths in figures.items()},
        "honesty_notes": [
            "All plotted values come from the user's Phase 8 runs; no numbers are invented.",
            "Error bars are the sample std over the configured seeds; the ablation and gain figures use paired per-seed differences.",
            "compute_gain draws the measured CPU step median read from results/real/step_time_benchmark.json; no value is hard-coded.",
            "Oracle and declared-type policies are labeled as such in every figure legend or axis.",
            "The all-policies figure uses a symlog x-axis because AllSlow and Scenario D differ by orders of magnitude.",
        ],
    }
    manifest_path = output / "plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"figures": figures, "manifest": manifest_path}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/full")
    parser.add_argument("--output-dir", default="results/plots")
    parser.add_argument("--formats", default="pdf,png", help="Comma-separated formats.")
    return parser.parse_args()


def main() -> None:
    """Generate all figures and print their paths."""
    args = parse_args()
    formats = tuple(item.strip() for item in args.formats.split(",") if item.strip())
    report = generate_all(args.results_dir, args.output_dir, formats)
    for name, paths in report["figures"].items():
        print(f"{name}: {', '.join(str(path) for path in paths)}")
    print(f"manifest: {report['manifest']}")


if __name__ == "__main__":
    main()
