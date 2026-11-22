"""Reconstructed Playback: deterministic step-through of saved access traces.

This page is intentionally labelled *reconstructed playback* everywhere: tier
residency is rebuilt from the logged migration events, while the trace's
``tier_served`` column remains the authoritative served-tier measurement.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from app.charts import wait_timeline
from app.data import (
    TIER_NAMES,
    format_metric,
    load_replay,
    load_run_metrics,
    scenario_note,
)
from app.replay import TraceReplay
from app.ui import (
    apply_theme,
    cached_replays,
    configure_page,
    full_width,
    hero,
    metric_card,
    note,
    render_sidebar,
    section,
)

SLIDER_KEY = "replay_slider"

PLAYBACK_NOTE = (
    "This is reconstructed playback, not a live simulation: tier residency is "
    "rebuilt from completed migration events in the log, and the trace's "
    "`tier_served` is authoritative for every access."
)


def _clamp(value: int, total: int) -> int:
    return max(0, min(int(value), total - 1))


def _set_index(value: int, total: int) -> None:
    """Set the index BEFORE the slider widget is instantiated.

    Streamlit forbids writing a widget's session-state key after that widget
    exists in the current run, so all synchronous updates must happen above
    the slider in :func:`_render_controls`.
    """
    clamped = _clamp(value, total)
    st.session_state["replay_index"] = clamped
    st.session_state[SLIDER_KEY] = clamped


def _queue_index(value: int, total: int) -> None:
    """Defer a post-slider update (autoplay) to the next script run."""
    st.session_state["replay_pending"] = _clamp(value, total)


def _render_controls(total: int) -> tuple[int, bool, float]:
    """Render transport buttons first, then the slider."""
    current = st.session_state["replay_index"]
    left, back, forward, end, play_col = st.columns([0.6, 0.6, 0.6, 0.6, 1.4])
    if left.button("⏮", help="First access", key="replay_first"):
        st.session_state["replay_playing"] = False
        _set_index(0, total)
        st.rerun()
    if back.button("◀", help="Previous access", key="replay_prev"):
        st.session_state["replay_playing"] = False
        _set_index(current - 1, total)
        st.rerun()
    if forward.button("▶", help="Next access", key="replay_next"):
        st.session_state["replay_playing"] = False
        _set_index(current + 1, total)
        st.rerun()
    if end.button("⏭", help="Last access", key="replay_last"):
        st.session_state["replay_playing"] = False
        _set_index(total - 1, total)
        st.rerun()
    playing = st.session_state.get("replay_playing", False)
    if play_col.button("⏸ Pause" if playing else "▶ Auto-play", key="replay_toggle"):
        st.session_state["replay_playing"] = not playing
        st.rerun()

    slider_col, speed_col = st.columns([3.0, 1.2])
    if SLIDER_KEY not in st.session_state:
        st.session_state[SLIDER_KEY] = current
    index = slider_col.slider(
        "Access index", min_value=0, max_value=total - 1, key=SLIDER_KEY,
    )
    st.session_state["replay_index"] = int(index)  # non-widget key: safe
    speed = speed_col.slider(
        "Delay (s)", min_value=0.01, max_value=0.50, value=0.05, step=0.01,
        key="replay_speed",
    )
    return int(index), st.session_state.get("replay_playing", False), float(speed)


def _render_access_detail(access: dict, show_truth: bool) -> None:
    """Show the observable fields of the current access (truth is opt-in)."""
    cols = st.columns(4)
    cols[0].metric("File id", str(int(access["file_id"])))
    cols[1].metric("Size", f"{int(access['size_bytes']) / 1024:,.1f} KB")
    cols[2].metric("Served by", TIER_NAMES.get(int(access["tier_served"]), "?"))
    cols[3].metric("FAST hit", "yes" if bool(access["fast_hit"]) else "no")
    st.caption(
        f"op=`{access['op']}` · phase=`{access['phase']}` · "
        f"epoch={int(access['epoch'])} · step={int(access['step'])} · "
        f"latency={float(access['latency_ms']):.4f} ms · "
        f"wait={float(access['wait_ms']):.4f} ms · ts={float(access['ts']):.3f}"
    )
    if show_truth:
        st.caption(f"Evaluation-only true_type = `{access['true_type']}`")


def _render_tiers(replay: TraceReplay, index: int) -> None:
    """Show reconstructed tier residency and the authoritative served tier."""
    state = replay.state_at(index)
    st.markdown("**Reconstructed tier residency** (completed migrations only)")
    cols = st.columns(3)
    for column, tier in zip(cols, (0, 1, 2)):
        name = TIER_NAMES.get(tier, str(tier))
        files = sum(1 for value in state.tier_by_file.values() if value == tier)
        size_mb = state.occupancy_bytes.get(tier, 0) / (1024 * 1024)
        with column:
            metric_card(f"{name} files", str(files), f"{size_mb:.2f} MB reconstructed")
    st.caption(
        f"Migrations applied: {state.migrations_applied} · "
        f"migrated bytes: {state.migration_bytes / (1024 * 1024):,.2f} MB · "
        f"cumulative logged wait: {state.cumulative_wait_ms:,.1f} ms · "
        f"cumulative FAST hits: {state.fast_hits}"
    )


def _render_recent_migrations(replay: TraceReplay, index: int) -> None:
    """Show the last completed migrations before the current access."""
    recent = replay.recent_migrations(index, count=8)
    if recent.empty:
        st.caption("No completed migrations before this access in the log.")
        return
    table = pd.DataFrame(
        {
            "ts": recent["ts"].map(lambda value: f"{float(value):.3f}"),
            "file_id": recent["file_id"].astype(int),
            "size (KB)": recent["size_bytes"].map(lambda value: f"{int(value) / 1024:,.1f}"),
            "tier move": [
                f"{TIER_NAMES.get(int(src), src)} → {TIER_NAMES.get(int(dst), dst)}"
                for src, dst in zip(recent["src_tier"], recent["dst_tier"])
            ],
            "kind": recent["kind"],
        }
    )
    full_width(st.dataframe, table, hide_index=True)


def main() -> None:
    """Streamlit entry point."""
    configure_page("Reconstructed Playback")
    apply_theme()
    render_sidebar()
    hero(
        eyebrow="Replay Mode",
        title="Reconstructed Playback",
        description=(
            "Step through a saved access trace. Residency is reconstructed from "
            "logged migrations; the trace's tier_served is authoritative."
        ),
        pills=["Deterministic", "Log-driven", "No live simulation"],
    )
    note(PLAYBACK_NOTE)

    runs = cached_replays()
    if not runs:
        st.error("No saved traces found. Run `python run_experiments.py …` first.")
        return
    labels = [run.label for run in runs]
    selected = st.selectbox(
        "Saved run",
        options=labels,
        format_func=lambda label: next(run.display_name for run in runs if run.label == label),
    )
    run = next(item for item in runs if item.label == selected)
    if st.session_state.get("replay_run") != selected:
        st.session_state["replay_run"] = selected
        _set_index(0, 1)
        st.session_state["replay_playing"] = False

    trace, migrations = load_replay(run)
    replay = TraceReplay(trace, migrations)
    total = len(replay)
    if total == 0:
        st.error("The selected trace contains no accesses.")
        return
    pending = st.session_state.pop("replay_pending", None)
    if pending is not None:
        _set_index(pending, total)
    else:
        _set_index(st.session_state.get("replay_index", 0), total)

    section("Saved run", f"{scenario_note(run.scenario)} · {run.display_name}")
    st.caption(
        f"Initial placement in the reconstruction: **{replay.initial_placement}**. "
        "Served tiers always come from the trace, never from the reconstruction."
    )
    metrics = load_run_metrics(run)
    if metrics:
        metric_cols = st.columns(3)
        for column, name in zip(metric_cols, ("data_wait_ms", "throughput", "fast_hit_rate")):
            if name in metrics:
                with column:
                    metric_card(name, format_metric(name, metrics[name]), "Logged run metric")

    section("Transport", "choose an access; auto-play advances the reconstruction")
    index, playing, speed = _render_controls(total)
    if playing and index < total - 1:
        time.sleep(speed)
        _queue_index(index + 1, total)
        st.rerun()
    if playing and index == total - 1:
        st.session_state["replay_playing"] = False

    section("Current access", f"access {index + 1} of {total} · reconstructed playback")
    state = replay.state_at(index)
    _render_access_detail(
        dict(state.access),
        st.checkbox("Show evaluation-only true_type", value=False),
    )
    _render_tiers(replay, index)

    section("Logged wait timeline", "trace values up to the selected access")
    full_width(st.pyplot, wait_timeline(replay, index))
    with st.expander("Recent completed migrations used by the reconstruction"):
        _render_recent_migrations(replay, index)
    note(
        PLAYBACK_NOTE + " In-flight prefetches are absent from the migration log, so "
        "reconstructed occupancy is illustrative. Nothing here is simulated live "
        "and no value is invented."
    )


if __name__ == "__main__":
    main()
