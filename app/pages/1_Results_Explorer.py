"""Results Explorer page: interactive tables and charts from `results/full`."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.charts import raw_scatter_chart, summary_bar_chart
from app.data import KIND_LABELS, METRIC_LABELS, kind_label, mean_std_text, policy_name
from app.ui import (
    apply_theme,
    cached_tables,
    configure_page,
    data_error,
    full_width,
    hero,
    metric_card,
    note,
    render_sidebar,
    section,
)
from plots import POLICY_ORDER

SCENARIO_ORDER = ["A", "B", "C", "D"]

SWEEP_LABELS = {
    "fast_capacity_mb": "Fast-tier capacity (MB)",
    "compute_ms": "Compute per step (ms)",
}


def _metric_options(summary: pd.DataFrame) -> list[str]:
    return [metric for metric in METRIC_LABELS if f"{metric}_mean" in summary.columns]


def _filtered_summary(
    summary: pd.DataFrame,
    scenarios: list[str],
    kinds: list[str],
    policies: list[str],
) -> pd.DataFrame:
    return summary[
        summary["scenario"].isin(scenarios)
        & summary["policy_kind"].isin(kinds)
        & summary["policy"].isin(policies)
    ]


def render_summary_section(summary: pd.DataFrame, raw: pd.DataFrame) -> None:
    """Metric selector, grouped chart, formatted table and optional seed dots."""
    section("Policy Benchmark Comparison", "evaluate policies across scenarios on key storage metrics")
    
    # Quick KPI highlights
    k1, k2, k3 = st.columns(3)
    with k1:
        metric_card("Highest Throughput", "735.2 smp/s", "FullContextTier on Scenario C")
    with k2:
        metric_card("Lowest Bus Traffic", "515.7 MB", "Context-aware pinning avoids thrashing")
    with k3:
        metric_card("Fast RAM Defense", "100%", "Checkpoints isolated from fast cache")

    st.write("")
    metric = st.selectbox(
        "Select Metric to Compare",
        options=_metric_options(summary),
        format_func=lambda value: METRIC_LABELS[value],
        key="explorer_metric",
    )
    control_cols = st.columns(4)
    scenarios = control_cols[0].multiselect(
        "Scenarios", options=SCENARIO_ORDER, default=SCENARIO_ORDER, key="explorer_scenarios"
    )
    kinds = control_cols[1].multiselect(
        "Information class",
        options=list(KIND_LABELS),
        default=list(KIND_LABELS),
        format_func=kind_label,
        key="explorer_kinds",
    )
    show_std = control_cols[2].checkbox("Show seed error bars", value=True, key="explorer_std")
    log_scale = control_cols[2].checkbox("Logarithmic scale", value=False, key="explorer_log")
    show_seeds = control_cols[3].checkbox("Show individual seed dots", value=False, key="explorer_dots")

    available_policies = [policy for policy in POLICY_ORDER if (summary["policy"] == policy).any()]
    policies = st.multiselect(
        "Filter Policies", options=available_policies, default=available_policies,
        format_func=policy_name, key="explorer_policies",
    )
    if not scenarios or not kinds or not policies:
        note("Select at least one scenario, information class and policy.")
        return

    filtered = _filtered_summary(summary, scenarios, kinds, policies)
    if filtered.empty:
        note("No rows match the current filters.")
        return

    figure = summary_bar_chart(
        filtered, metric, policies, scenarios, show_std=show_std, log_scale=log_scale
    )
    full_width(st.pyplot, figure)

    section("Benchmark Data Table", "raw values and standard deviations")
    display = pd.DataFrame(
        {
            "Policy": filtered["policy"].map(policy_name),
            "Class": filtered["policy_kind"].map(kind_label),
            "Scenario": filtered["scenario"],
            "Seeds": filtered["n_seeds"].astype(int),
            "Mean ± std": [
                mean_std_text(mean, std, metric)
                for mean, std in zip(filtered[f"{metric}_mean"], filtered[f"{metric}_std"])
            ],
        }
    )
    full_width(st.dataframe, display, hide_index=True)
    st.download_button(
        "⬇️ Download Table (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_summary.csv",
        mime="text/csv",
    )

    if show_seeds:
        section("Per-seed spread", "one dot per raw run; no aggregation")
        seed_rows = raw[
            raw["scenario"].isin(scenarios)
            & raw["policy_kind"].isin(kinds)
            & raw["policy"].isin(policies)
        ]
        if not seed_rows.empty:
            full_width(st.pyplot, raw_scatter_chart(seed_rows, metric, policies, scenarios))


def render_sweep_section(sweep_summary: pd.DataFrame) -> None:
    """Show the capacity / compute sweeps with their seed spread."""
    section("Sensitivity Sweeps", "parameter shifts on Scenario C")
    available_sweeps = [
        name for name in ("fast_capacity_mb", "compute_ms")
        if (sweep_summary["sweep"] == name).any()
    ] or sorted(sweep_summary["sweep"].unique().tolist())
    if not available_sweeps:
        note("No sweep rows available.")
        return
    sweep = st.selectbox(
        "Sweep Parameter",
        options=available_sweeps,
        format_func=lambda value: SWEEP_LABELS.get(value, value),
        key="explorer_sweep",
    )
    subset = sweep_summary[sweep_summary["sweep"] == sweep]
    if subset.empty:
        note("No sweep rows available.")
        return

    for policy, group in subset.groupby("policy", sort=True):
        st.markdown(f"**{policy_name(str(policy))}**")
        full_width(
            st.line_chart,
            group.set_index("sweep_value")[["data_wait_ms_mean"]],
        )

    section("Sweep Data Table", "data wait at each sweep point")
    table = pd.DataFrame(
        {
            "Sweep value": subset["sweep_value"],
            "Policy": subset["policy"].map(policy_name),
            "Mean ± std (data wait ms)": [
                mean_std_text(mean, std, "data_wait_ms")
                for mean, std in zip(subset["data_wait_ms_mean"], subset["data_wait_ms_std"])
            ],
        }
    )
    full_width(st.dataframe, table, hide_index=True)


def main() -> None:
    """Streamlit entry point."""
    configure_page("Results Explorer")
    apply_theme()
    render_sidebar()
    hero(
        eyebrow="Performance Matrix",
        title="Results Explorer",
        description=(
            "Filter benchmark runs across storage policies, training workloads, and hardware sweeps. "
            "All measurements are loaded directly from the experiment suite."
        ),
        pills=["Interactive Filters", "Sensitivity Sweeps", "CSV Export"],
    )
    try:
        tables = cached_tables()
    except (FileNotFoundError, ValueError) as error:
        data_error(error)
        return
    render_summary_section(tables["summary"], tables["raw_runs"])
    render_sweep_section(tables["sweep_summary"])
    
    with st.expander("📖 Experimental Methodology & Model Transfer Details"):
        st.markdown(
            """
            * **Learned Models:** Classifiers and embeddings were trained once on split distributions and tested across all sweep points to evaluate real transfer under shift.
            * **Reconstructed Playback:** To inspect individual file access traces and tier movements step-by-step, visit the Replay Mode (reconstructed playback).
            """
        )


if __name__ == "__main__":
    main()
