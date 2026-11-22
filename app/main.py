"""Overview page of the ContextTier research dashboard.

Run with::

    streamlit run app/main.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.data import (
    SCENARIOS,
    mean_std_text,
    policy_name,
    scenario_note,
)
from app.ui import (
    DEPLOYABLE_KINDS,
    IDEALIZED_POLICIES,
    apply_theme,
    cached_bootstrap,
    cached_compute_benchmark,
    cached_experiment_manifest,
    cached_tables,
    configure_page,
    data_error,
    full_width,
    hero,
    html_escape,
    metric_card,
    note,
    render_sidebar,
    section,
    show_honesty_note,
)


def _deployable_rows(summary: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Rows for policies that are neither oracle, privileged nor idealized."""
    return summary[
        (summary["scenario"] == scenario)
        & (summary["policy_kind"].isin(DEPLOYABLE_KINDS))
        & (~summary["policy"].isin(IDEALIZED_POLICIES))
    ]


def render_overview(manifest: dict) -> None:
    """Render the project pitch and the three research extensions."""
    config = manifest.get("config", {})
    counts = manifest.get("run_counts", {})
    seeds = config.get("seeds", [])
    seed_text = f"{len(seeds)} configured seeds" if seeds else "seeds from manifest"
    hero(
        eyebrow="Research dashboard · CPU emulation",
        title="ContextTier",
        description=(
            "Context-aware dynamic data tiering for AI training. Explore the exact "
            "measurements behind the paper, browse every figure, and inspect "
            "reconstructed playback of saved runs."
        ),
        pills=[
            f"{counts.get('matrix_runs', 0)} matrix runs",
            f"{counts.get('sweep_runs', 0)} sweep runs",
            seed_text,
        ],
    )
    show_honesty_note()

    section("The question", "what changes when access context is observable")
    st.markdown(
        """
        How should a training-data store place and migrate files across fast / mid /
        slow tiers when the *context* of each access differs — training chunks,
        metadata, checkpoints, validation reads and hot-example caches? The same
        result files power this dashboard and the paper's tables.
        """
    )

    section("Three extensions", "beyond seed-based exact prefetching")
    extension_cols = st.columns(3)
    extension_cols[0].markdown(
        "**1 · Context awareness**\n\n"
        "Observed context is used for *all* data types, not only training samples."
    )
    extension_cols[1].markdown(
        "**2 · Dynamic migration**\n\n"
        "Files move between tiers as the workload evolves, with logged migrations."
    )
    extension_cols[2].markdown(
        "**3 · State-dependent workloads**\n\n"
        "Scenario C's order depends on training state, so seed clairvoyance does "
        "not apply; learned classifiers, embeddings and tabular RL are evaluated "
        "alongside observable baselines."
    )


def _bootstrap_row(
    bootstrap: pd.DataFrame | None,
    scenario: str,
    metric: str,
    *,
    baseline: str = "lru",
    focus: str = "full_context",
) -> pd.Series | None:
    """Return the paired-bootstrap row for one focus/baseline/metric pair."""
    if bootstrap is None or bootstrap.empty:
        return None
    match = bootstrap[
        (bootstrap["sweep"] == "matrix")
        & (bootstrap["scenario"] == scenario)
        & (bootstrap["metric"] == metric)
        & (bootstrap["baseline"] == baseline)
        & (bootstrap["focus"] == focus)
    ]
    return match.iloc[0] if not match.empty else None


def _render_compute_strip(
    sweep_summary: pd.DataFrame | None, scenario: str
) -> None:
    """Show where policies separate once compute stops hiding the I/O path."""
    if sweep_summary is None or sweep_summary.empty:
        return
    compute = sweep_summary[
        (sweep_summary["sweep"] == "compute_ms")
        & (sweep_summary["scenario"] == scenario)
    ]
    if compute.empty:
        return
    separating: list[float] = []
    for value, group in compute.groupby("sweep_value"):
        if int((group["data_wait_ms_mean"] > 0).sum()) >= 2:
            separating.append(float(value))
    if not separating:
        return
    value = min(separating)
    group = compute[compute["sweep_value"] == value].set_index("policy")
    order = [
        policy
        for policy in ("lru", "context_rules", "full_context")
        if policy in group.index
    ] or list(group.index)
    section(
        "Where policies separate",
        f"mean data wait at {value:g} ms compute per step · Scenario {scenario}",
    )
    columns = st.columns(len(order))
    for column, policy in zip(columns, order):
        row = group.loc[policy]
        with column:
            metric_card(
                policy_name(str(policy)),
                f"{float(row['data_wait_ms_mean']):,.1f} ms",
                f"± {float(row['data_wait_ms_std']):,.1f} ms across seeds",
            )
    note(
        "Reducing the compute budget exposes the I/O path: data wait becomes "
        "non-zero and policies rank. At the measured step time it is hidden by "
        "compute overlap, which is why the headline matrix ties on data wait."
    )


def render_key_numbers(
    summary: pd.DataFrame,
    sweep_summary: pd.DataFrame | None = None,
    bootstrap: pd.DataFrame | None = None,
) -> None:
    """Show headline numbers computed live from the Phase 8 tables."""
    section("Headline results", "read at runtime from results/full/")
    scenario = st.selectbox(
        "Focus scenario",
        options=list(SCENARIOS),
        format_func=lambda letter: f"{letter} — {scenario_note(letter)}",
        index=list(SCENARIOS).index("C") if "C" in SCENARIOS else 0,
        key="overview_scenario",
    )
    rows = summary[summary["scenario"] == scenario]
    deployable = _deployable_rows(summary, scenario)
    if rows.empty or deployable.empty:
        note("No deployable summary rows for this scenario yet.")
        return

    def row_for(policy: str):
        matching = rows[rows["policy"] == policy]
        return matching.iloc[0] if not matching.empty else None

    lru = row_for("lru")
    full = row_for("full_context")
    allslow = row_for("allslow")
    max_wait = float(rows["data_wait_ms_mean"].max())
    max_wait_std = float(
        rows.loc[rows["data_wait_ms_mean"].idxmax(), "data_wait_ms_std"]
    )

    cards = st.columns(3)
    with cards[0]:
        if max_wait == 0.0:
            metric_card(
                "Zero-wait ceiling",
                "0.00 ms",
                f"every policy in {scenario}, incl. AllSlow · measured compute",
            )
        else:
            metric_card(
                "Largest data wait",
                mean_std_text(max_wait, max_wait_std, "data_wait_ms"),
                f"max over all policies in Scenario {scenario}",
            )
    if full is not None:
        foot = "Deployable learned policy"
        boot = _bootstrap_row(bootstrap, scenario, "throughput")
        if boot is not None and pd.notna(boot["mean_difference"]):
            foot = (
                f"+{float(boot['mean_difference']):,.2f} samples/s vs LRU · "
                f"{int(boot['seeds_won'])}/{int(boot['n_seeds'])} seeds"
            )
        with cards[1]:
            metric_card(
                "FullContextTier throughput",
                mean_std_text(
                    float(full["throughput_mean"]),
                    float(full["throughput_std"]),
                    "throughput",
                ),
                foot,
            )
    if allslow is not None and lru is not None:
        with cards[2]:
            metric_card(
                "Uncached control (AllSlow)",
                mean_std_text(
                    float(allslow["throughput_mean"]),
                    float(allslow["throughput_std"]),
                    "throughput",
                ),
                f"−{float(lru['throughput_mean']) - float(allslow['throughput_mean']):,.1f} "
                f"samples/s vs LRU · p95 {float(allslow['latency_p95_ms_mean']):.2f} ms",
            )

    note(
        "At the measured CPU step time, data wait is zero for every policy, so "
        "this matrix does not rank placement on that metric. Caching still pays "
        "in throughput (uncached reads), and FullContextTier's small but "
        "seed-consistent edge over LRU comes from less synchronous promotion. "
        "Ranking on data wait returns when compute is reduced."
    )
    _render_compute_strip(sweep_summary, scenario)


def render_scenarios() -> None:
    """Explain the four fixed scenarios from the design contract."""
    section("Scenarios", "the fixed workload matrix")
    display = pd.DataFrame(
        [
            {"Scenario": letter, "Workload": description}
            for letter, description in SCENARIOS.items()
        ]
    )
    full_width(st.dataframe, display, hide_index=True)


def render_provenance(manifest: dict, benchmark: dict | None = None) -> None:
    """Show the manifest's run counts and fixed configuration knobs."""
    section("Provenance", "configuration recorded with the experiment run")
    config = manifest.get("config", {})
    counts = manifest.get("run_counts", {})
    compute_ms = config.get("compute_ms")
    measurement = (benchmark or {}).get("measurement", {})
    facts = [
        ("Engine", str(manifest.get("engine", "unknown"))),
        ("Matrix runs", f"{counts.get('matrix_runs', 0):,}"),
        ("Sweep runs", f"{counts.get('sweep_runs', 0):,}"),
        ("Epochs", str(config.get("epochs", "?"))),
    ]
    if measurement and compute_ms is not None:
        facts.append(
            (
                "Compute per step (measured)",
                f"{float(measurement.get('median_ms', compute_ms)):.3f} ms",
            )
        )
    else:
        facts.append(
            ("Compute per step", f"{compute_ms} ms" if compute_ms is not None else "?")
        )
    cols = st.columns(len(facts))
    for column, (label, value) in zip(cols, facts):
        with column:
            metric_card(label, value)
    if measurement:
        st.caption(
            "Compute proxy = median of "
            f"{measurement.get('steps_measured', '?')} measured CPU forward+backward "
            "steps (results/real/step_time_benchmark.json)."
        )
    seeds = config.get("seeds", [])
    st.caption(
        f"Seeds: {html_escape(', '.join(str(seed) for seed in seeds)) or 'from manifest'}"
    )
    for item in manifest.get("honesty_notes", []):
        st.caption(f"• {item}")


def main() -> None:
    """Streamlit entry point."""
    configure_page("Overview")
    apply_theme()
    render_sidebar()
    try:
        tables = cached_tables()
        manifest = cached_experiment_manifest()
    except (FileNotFoundError, ValueError) as error:
        hero("Research dashboard", "ContextTier", "Result artefacts are unavailable.")
        data_error(error)
        return
    render_overview(manifest)
    render_key_numbers(
        tables["summary"],
        tables.get("sweep_summary"),
        cached_bootstrap(),
    )
    render_scenarios()
    render_provenance(manifest, cached_compute_benchmark())
    note(
        "Next: Results Explorer for every metric, Figures for publication plots, "
        "or Replay Mode for reconstructed playback of saved traces."
    )


if __name__ == "__main__":
    main()
