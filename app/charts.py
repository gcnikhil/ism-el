"""Small matplotlib charts for the Streamlit pages (no Streamlit imports)."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.data import DISPLAY_NAMES, KIND_LABELS, METRIC_LABELS
from plots import KIND_COLORS
from app.replay import TraceReplay


def _label(policy: str) -> str:
    return DISPLAY_NAMES.get(policy, policy)


def _color(kind: str) -> str:
    return KIND_COLORS.get(kind, "#8172b3")


def summary_bar_chart(
    summary: pd.DataFrame,
    metric: str,
    policies: Sequence[str],
    scenarios: Sequence[str],
    *,
    show_std: bool = True,
    log_scale: bool = False,
) -> plt.Figure:
    """Grouped horizontal bars of mean metric with optional seed-std bars.

    Each scenario is one panel; policies are grouped by information class via
    the same colours used in the paper figures.
    """
    n_scenarios = max(1, len(scenarios))
    figure, axes = plt.subplots(
        n_scenarios, 1, figsize=(9, 2.4 * n_scenarios), squeeze=False
    )
    available = [policy for policy in policies if (summary["policy"] == policy).any()]
    for axis, scenario in zip(axes[:, 0], scenarios):
        positions = np.arange(len(available))
        means: list[float] = []
        stds: list[float] = []
        kinds: list[str] = []
        for policy in available:
            row = summary[
                (summary["policy"] == policy) & (summary["scenario"] == scenario)
            ]
            if row.empty:
                means.append(float("nan"))
                stds.append(0.0)
            else:
                means.append(float(row[f"{metric}_mean"].iloc[0]))
                stds.append(float(row.get(f"{metric}_std", pd.Series([0.0])).iloc[0]))
            kinds.append(str(row["policy_kind"].iloc[0]) if not row.empty else "observable_baseline")

        axis.barh(
            positions,
            means,
            xerr=stds if show_std else None,
            capsize=3,
            color=[_color(kind) for kind in kinds],
            edgecolor="black",
            linewidth=0.4,
        )
        axis.set_yticks(positions, labels=[_label(policy) for policy in available])
        axis.invert_yaxis()
        axis.set_title(f"Scenario {scenario}")
        axis.set_xlabel(METRIC_LABELS.get(metric, metric))
        if log_scale:
            axis.set_xscale("symlog", linthresh=1.0)
        axis.grid(axis="x", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.tight_layout()
    return figure


def raw_scatter_chart(
    raw: pd.DataFrame,
    metric: str,
    policies: Sequence[str],
    scenarios: Sequence[str],
) -> plt.Figure:
    """Show one dot per seed so spread is not hidden by mean±std."""
    figure, axis = plt.subplots(figsize=(9, 4.5))
    colors = {kind: _color(kind) for kind in KIND_LABELS}
    offset = 0.0
    ticks: list[float] = []
    tick_labels: list[str] = []
    for policy in policies:
        for scenario in scenarios:
            rows = raw[
                (raw["policy"] == policy) & (raw["scenario"] == scenario)
            ]
            if rows.empty:
                continue
            kind = str(rows["policy_kind"].iloc[0])
            x_values = np.full(len(rows), offset)
            axis.scatter(
                x_values,
                rows[metric].to_numpy(dtype=float),
                color=colors.get(kind, "#8172b3"),
                alpha=0.75,
                s=45,
                edgecolor="black",
                linewidth=0.3,
                label=kind if kind not in axis.get_legend_handles_labels()[1] else None,
            )
            ticks.append(offset)
            tick_labels.append(f"{_label(policy)}\n{scenario}")
            offset += 1
    axis.set_xticks(ticks, labels=tick_labels, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel(METRIC_LABELS.get(metric, metric))
    axis.grid(axis="y", alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, frameon=False, fontsize=8)
    figure.tight_layout()
    return figure


def wait_timeline(replay: TraceReplay, index: int) -> plt.Figure:
    """Plot per-read wait up to the current access with a cursor line."""
    series = replay.wait_series(index)
    figure, axis = plt.subplots(figsize=(9, 2.6))
    axis.plot(series["access"], series["wait_ms"], linewidth=0.9, color="#4c72b0")
    if len(series):
        axis.axvline(index, color="#c44e52", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Access index")
    axis.set_ylabel("wait_ms")
    axis.set_title("Logged per-read wait (trace values, not the paper's per-step metric)")
    axis.grid(alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    return figure
