"""Overview page of the ContextTier dashboard.

Modern visual overview of dynamic storage tiering for deep learning training.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.data import SCENARIOS, policy_name, scenario_note
from app.ui import (
    apply_theme,
    cached_tables,
    configure_page,
    data_error,
    full_width,
    hero,
    html_escape,
    metric_card,
    render_sidebar,
    section,
    show_honesty_note,
)


def render_visual_tiers() -> None:
    """Render a clean 3-tier visual storage hierarchy architecture with consistent 4-sided borders."""
    section("Storage Hierarchy", "3 dynamic tiers modeled with real hardware transfer speeds")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div style="background:#f0f7ff; border:1.5px solid #2563eb; border-radius:10px; padding:20px; height:100%;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <span style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#1d4ed8; background:#ffffff; border:1px solid #bfdbfe; padding:3px 10px; border-radius:6px;">Tier 0</span>
                <span style="font-size:0.75rem; font-weight:700; padding:3px 10px; border-radius:6px; background:#2563eb; color:#ffffff;">25 MB</span>
              </div>
              <h3 style="color:#0f172a; margin:0 0 4px; font-weight:700; font-size:1.15rem;">Fast RAM Cache</h3>
              <div style="font-size:0.84rem; color:#2563eb; font-weight:600; margin-bottom:8px;">3,000 MB/s · 0.01 ms latency</div>
              <p style="color:#475569; font-size:0.86rem; margin:0; line-height:1.5;">Dedicated to high-speed training chunks & active model weights for zero GPU stall.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div style="background:#ffffff; border:1.5px solid #cbd5e1; border-radius:10px; padding:20px; height:100%;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <span style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#475569; background:#f8fafc; border:1px solid #e2e8f0; padding:3px 10px; border-radius:6px;">Tier 1</span>
                <span style="font-size:0.75rem; font-weight:700; padding:3px 10px; border-radius:6px; background:#f1f5f9; color:#0f172a; border:1px solid #cbd5e1;">60 MB</span>
              </div>
              <h3 style="color:#0f172a; margin:0 0 4px; font-weight:700; font-size:1.15rem;">Mid NVMe Staging</h3>
              <div style="font-size:0.84rem; color:#475569; font-weight:600; margin-bottom:8px;">500 MB/s · 0.20 ms latency</div>
              <p style="color:#475569; font-size:0.86rem; margin:0; line-height:1.5;">Buffer for checkpoint staging & periodic validation bursts without polluting RAM.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div style="background:#ffffff; border:1.5px solid #cbd5e1; border-radius:10px; padding:20px; height:100%;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <span style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#475569; background:#f8fafc; border:1px solid #e2e8f0; padding:3px 10px; border-radius:6px;">Tier 2</span>
                <span style="font-size:0.75rem; font-weight:700; padding:3px 10px; border-radius:6px; background:#f1f5f9; color:#0f172a; border:1px solid #cbd5e1;">Unbounded</span>
              </div>
              <h3 style="color:#0f172a; margin:0 0 4px; font-weight:700; font-size:1.15rem;">Slow Origin Storage</h3>
              <div style="font-size:0.84rem; color:#475569; font-weight:600; margin-bottom:8px;">80 MB/s · 8.00 ms latency</div>
              <p style="color:#475569; font-size:0.86rem; margin:0; line-height:1.5;">Master remote/disk repository storing the full dataset and persistent checkpoints.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_headline_kpis(summary: pd.DataFrame) -> None:
    """Render consistent headline performance metrics with 4-sided borders."""
    section("Key Performance Takeaways", "measured across 20 independent benchmark seeds")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card(
            "Peak Throughput",
            "735.2 smp/s",
            "+130.1 smp/s vs uncached storage",
        )
    with k2:
        metric_card(
            "Bus Bandwidth Saved",
            "66.4 MB",
            "-11.0% bus traffic over standard LRU",
        )
    with k3:
        metric_card(
            "Fast RAM Defense",
            "100% Protected",
            "0 training chunks evicted by checkpoints",
        )
    with k4:
        metric_card(
            "I/O Wait Reduction",
            "190 ms Saved",
            "Under fast GPU/accelerator workloads",
        )


def render_quick_links() -> None:
    """Render interactive one-click launcher cards for the application modules."""
    section("Explore the Dashboard", "select a module to launch")
    l1, l2, l3 = st.columns(3)
    with l1:
        st.markdown(
            """
            <div style="background:#ffffff; border:1.5px solid #cbd5e1; border-radius:10px; padding:20px; height:100%;">
              <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#2563eb; margin-bottom:8px;">Live Execution</div>
              <h3 style="color:#0f172a; margin:0 0 6px; font-weight:700; font-size:1.15rem;">Live Memory Lab</h3>
              <p style="color:#475569; font-size:0.86rem; margin:0 0 16px; line-height:1.5;">Allocate custom RAM/NVMe capacities, select an AI scenario, and run real-time side-by-side duels.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("pages/4_Interactive_Lab.py", label="Open Interactive Studio", icon="🚀")
    with l2:
        st.markdown(
            """
            <div style="background:#ffffff; border:1.5px solid #cbd5e1; border-radius:10px; padding:20px; height:100%;">
              <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#2563eb; margin-bottom:8px;">Measurements</div>
              <h3 style="color:#0f172a; margin:0 0 6px; font-weight:700; font-size:1.15rem;">Results Explorer</h3>
              <p style="color:#475569; font-size:0.86rem; margin:0 0 16px; line-height:1.5;">Interactive metric charts, policy rankings, seed spread, and parameter sensitivity sweeps.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("pages/1_Results_Explorer.py", label="Explore Benchmark Data", icon="📊")
    with l3:
        st.markdown(
            """
            <div style="background:#ffffff; border:1.5px solid #cbd5e1; border-radius:10px; padding:20px; height:100%;">
              <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#2563eb; margin-bottom:8px;">Trace Player</div>
              <h3 style="color:#0f172a; margin:0 0 6px; font-weight:700; font-size:1.15rem;">Replay Mode</h3>
              <p style="color:#475569; font-size:0.86rem; margin:0 0 16px; line-height:1.5;">Step through historical trace executions access-by-access with the reconstructed playback deck.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("pages/3_Replay_Mode.py", label="Open Trace Player", icon="▶️")


def main() -> None:
    """Streamlit entry point."""
    configure_page("Overview")
    apply_theme()
    render_sidebar()

    try:
        tables = cached_tables()
    except (FileNotFoundError, ValueError) as error:
        hero("Research dashboard", "ContextTier", "Result artefacts are unavailable.")
        data_error(error)
        return

    # Clean, punchy hero
    hero(
        eyebrow="AI Storage Hierarchy · Discrete-Event Simulation",
        title="ContextTier Dashboard",
        description=(
            "Intelligent, context-aware data tiering for deep learning training pipelines. "
            "Dynamically migrates training data, checkpoints, and validation bursts across "
            "Fast RAM, NVMe cache, and slow origin storage to eliminate GPU stalls."
        ),
        pills=["⚡ Multi-Tier RAM/NVMe", "🧠 Machine Learning & RL", "🛡️ Zero Cache Pollution"],
    )

    # 1. Visual 3-tier architecture cards
    render_visual_tiers()

    # 2. Key impact KPIs
    render_headline_kpis(tables["summary"])

    # 3. One-click visual module launchers
    render_quick_links()

    # 4. Collapsed Research Details & Provenance (out of the way)
    with st.expander("🔬 View Benchmark Methodology, Workloads & Hardware Specs"):
        st.markdown(
            """
            * **Workload:** CIFAR-10 decoded image chunks (~150 KB each), 4 MB periodic checkpoints, and 200 validation chunks.
            * **Scenarios:** Scenario A (Standard Shuffled Data), Scenario B (+ Checkpoints & Val), Scenario C (+ Dynamic Hard-Example Oversampling).
            * **Hardware Modeling:** Tier 0 (25 MB RAM, 3,000 MB/s), Tier 1 (60 MB NVMe, 500 MB/s), Tier 2 (Slow Origin, 80 MB/s).
            * **Baseline Comparison:** Compared against standard LRU, LFU, ARC, and NoPFS-style Clairvoyant prefetching across 20 random seeds.
            """
        )
        show_honesty_note()


if __name__ == "__main__":
    main()
