"""Phase 10 tests: website data helpers, replay engine and Streamlit smoke."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from app.charts import summary_bar_chart, wait_timeline
from app.data import (
    RESULTS_DIR,
    discover_replays,
    experiment_manifest,
    figure_catalog,
    format_metric,
    kind_label,
    load_replay,
    load_run_metrics,
    mean_std_text,
    plot_manifest,
    scenario_note,
    tier_bytes,
)
from app.replay import TraceReplay
from config import load_config, project_root
from logger import Logger
from tiers import FAST, MID, SLOW


def _synthetic_trace(scenario: str = "A", policy: str = "lru") -> pd.DataFrame:
    rows = []
    for index, (ts, file_id, served, hit, wait) in enumerate(
        ((1.0, 10, SLOW, 0, 8.0), (2.0, 20, FAST, 1, 0.0), (3.0, 10, MID, 0, 2.0))
    ):
        rows.append(
            {
                "ts": ts,
                "run_id": f"{scenario}-{policy}-seed0",
                "seed": 0,
                "scenario": scenario,
                "policy": policy,
                "file_id": file_id,
                "size_bytes": 1024,
                "op": "read",
                "epoch": 0,
                "step": index,
                "phase": "train",
                "tier_served": served,
                "latency_ms": 1.0,
                "wait_ms": wait,
                "fast_hit": bool(hit),
                "true_type": "chunk",
            }
        )
    return pd.DataFrame(rows, columns=list(Logger.TRACE_COLUMNS))


def _synthetic_migrations() -> pd.DataFrame:
    rows = [
        {"ts": 1.5, "file_id": 10, "size_bytes": 1024, "src_tier": SLOW, "dst_tier": FAST, "kind": "promote"},
        {"ts": 2.5, "file_id": 10, "size_bytes": 1024, "src_tier": FAST, "dst_tier": MID, "kind": "promote"},
        {"ts": 9.0, "file_id": 20, "size_bytes": 1024, "src_tier": SLOW, "dst_tier": FAST, "kind": "promote"},
    ]
    frame = pd.DataFrame(rows)
    for column in Logger.MIGRATION_COLUMNS:
        if column not in frame.columns:
            frame[column] = "seed0"
    return frame[list(Logger.MIGRATION_COLUMNS)]


def test_discover_replays_finds_all_saved_runs() -> None:
    runs = discover_replays()
    # Model-specific runs are deliberately added under results/models/ during
    # Phase 12, so discovery must remain open-ended rather than fixed at the
    # original nine saved traces.
    assert len(runs) >= 9
    assert all(run.trace_path.is_file() for run in runs)
    assert all(run.label.endswith("_trace.csv") for run in runs)
    assert runs[0].engine == "virtual"
    assert any(run.engine == "real" for run in runs)
    expected = {
        ("A", "lru", 0),
        ("C", "classifier", 3),
        ("C", "embedding", 3),
        ("C", "full_context", 3),
    }
    found = {(run.scenario, run.policy, run.seed) for run in runs if run.engine == "virtual"}
    assert expected.issubset(found)


def test_load_replay_and_metrics_from_saved_run() -> None:
    run = next(
        item for item in discover_replays()
        if item.label == "A/lru_seed0_trace.csv"
    )
    trace, migrations = load_replay(run)
    # Phase 13 adds an evaluation-only temperature column during replay loading;
    # the original A-D logging contract remains the required prefix.
    assert list(trace.columns[: len(Logger.TRACE_COLUMNS)]) == list(Logger.TRACE_COLUMNS)
    assert set(trace.columns[len(Logger.TRACE_COLUMNS) :]) <= {Logger.TEMPERATURE_COLUMN}
    assert list(migrations.columns) == list(Logger.MIGRATION_COLUMNS)
    assert len(trace) > 1000
    metrics = load_run_metrics(run)
    assert "data_wait_ms" in metrics
    assert metrics["read_count"] == float(len(trace))


def test_replay_applies_migrations_in_timestamp_order() -> None:
    replay = TraceReplay(_synthetic_trace(), _synthetic_migrations())
    first = replay.state_at(0)
    assert first.served_tier == SLOW
    assert first.tier_by_file[10] == SLOW
    assert first.migrations_applied == 0
    assert first.cumulative_wait_ms == 8.0
    assert first.fast_hits == 0
    assert first.occupancy_bytes[SLOW] == 2048

    second = replay.state_at(1)
    assert second.tier_by_file[10] == FAST
    assert second.migrations_applied == 1
    assert second.migration_bytes == 1024
    assert second.served_tier == FAST
    assert second.fast_hit is True
    assert second.cumulative_wait_ms == 8.0
    assert second.cumulative_latency_ms == 2.0
    assert second.fast_hits == 1

    third = replay.state_at(2)
    assert third.tier_by_file[10] == MID
    assert third.migrations_applied == 2
    assert third.served_tier_name == "MID"
    assert third.cumulative_wait_ms == 10.0
    assert third.occupancy_bytes[MID] == 1024
    assert third.occupancy_bytes[SLOW] == 1024


def test_replay_initial_placement_rules() -> None:
    scenario_d = TraceReplay(_synthetic_trace(scenario="D"))
    allfast = TraceReplay(_synthetic_trace(policy="allfast"))
    assert scenario_d.initial_tier == FAST
    assert allfast.initial_tier == FAST
    assert scenario_d.state_at(0).tier_by_file[10] == FAST
    assert TraceReplay(_synthetic_trace()).initial_tier == SLOW


def test_replay_without_migration_log() -> None:
    replay = TraceReplay(_synthetic_trace(), None)
    state = replay.state_at(2)
    assert state.migrations_applied == 0
    assert state.migration_bytes == 0
    assert state.tier_by_file[10] == SLOW
    assert replay.recent_migrations(2).empty
    with pytest.raises(IndexError):
        replay.state_at(len(replay))


def test_replay_wait_series_and_recent_migrations() -> None:
    replay = TraceReplay(_synthetic_trace(), _synthetic_migrations())
    series = replay.wait_series(1)
    assert list(series["wait_ms"]) == [8.0, 0.0]
    recent = replay.recent_migrations(2, count=1)
    assert len(recent) == 1
    assert int(recent.iloc[0]["file_id"]) == 10


def test_charts_render_from_tables() -> None:
    summary = pd.read_csv(project_root() / "results" / "full" / "summary.csv")
    figure = summary_bar_chart(
        summary, "data_wait_ms", ["lru", "full_context", "belady"], ["C", "D"],
        show_std=True, log_scale=True,
    )
    assert len(figure.axes) == 2
    assert figure.axes[0].get_xscale() == "symlog"
    replay = TraceReplay(_synthetic_trace(), _synthetic_migrations())
    timeline = wait_timeline(replay, 2)
    assert timeline.axes[0].get_xlabel() == "Access index"


def test_data_display_helpers() -> None:
    assert format_metric("migration_bytes", 2 * 1024 * 1024) == "2.00 MB"
    assert format_metric("fast_hit_rate", 0.91234) == "0.912"
    assert "±" in mean_std_text(10.0, 2.0, "data_wait_ms")
    assert "C" in scenario_note("C")
    assert kind_label("oracle_upper_bound") == "Oracle upper bound"
    assert "FAST" in tier_bytes({FAST: 1024, SLOW: 2048})


def test_manifests_and_figure_catalog() -> None:
    manifest = experiment_manifest()
    assert manifest["phase"] == 8
    # Run count scales with the configured seeds; derive it from the config so
    # a 20-seed regeneration is accepted without hard-coding a magic number.
    config = load_config()
    n_runs = (
        len(config["seeds"])
        * len(manifest["policies"])
        * len(manifest["scenarios"])
    )
    assert manifest["run_counts"]["matrix_runs"] == n_runs
    plots = plot_manifest()
    assert plots["phase"] == 9
    figures = figure_catalog()
    assert len(figures) == 8
    for figure in figures:
        assert figure["png"] is not None and figure["png"].is_file()
        assert figure["pdf"] is not None and figure["pdf"].is_file()


def test_app_sources_contain_no_hardcoded_results() -> None:
    """The UI must read every experimental value from result files."""
    forbidden = ("755.97", "814.69", "708.63", "38297.43")
    sources = sorted((project_root() / "app").rglob("*.py"))
    sources.append(project_root() / "build_site.py")
    assert sources
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for number in forbidden:
            assert number not in text, f"{number} hard-coded in {path}"
        assert "use_container_width" not in text, f"deprecated width argument in {path}"


def test_pages_label_replay_as_reconstructed_playback() -> None:
    """Replay must be labelled reconstructed playback in every page source."""
    app_root = project_root() / "app"
    for path in [app_root / "main.py", *app_root.glob("pages/*.py"), app_root / "README.md"]:
        text = path.read_text(encoding="utf-8").lower()
        assert "reconstructed playback" in text, f"missing label in {path}"


def test_full_width_helper_falls_back_on_older_streamlit() -> None:
    """The width helper must support modern and legacy Streamlit signatures."""
    from app.ui import full_width

    calls: list[dict] = []

    def modern(**kwargs):
        calls.append(kwargs)
        return "modern"

    def legacy(**kwargs):
        if "width" in kwargs:
            raise TypeError("unexpected keyword argument 'width'")
        calls.append(kwargs)
        return "legacy"

    assert full_width(modern) == "modern"
    assert calls[-1] == {"width": "stretch"}
    assert full_width(legacy) == "legacy"
    assert calls[-1] == {"use_container_width": True}


PAGES = (
    project_root() / "app" / "main.py",
    project_root() / "app" / "pages" / "1_Results_Explorer.py",
    project_root() / "app" / "pages" / "2_Figures.py",
    project_root() / "app" / "pages" / "3_Replay_Mode.py",
)


def test_four_pages_exist() -> None:
    assert len(PAGES) == 4
    for path in PAGES:
        assert path.is_file(), path


@pytest.mark.parametrize("path", PAGES, ids=lambda path: Path(path).name)
def test_streamlit_pages_render_without_exceptions(path: Path) -> None:
    """AppTest smoke: each page renders against the real result artefacts."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file(str(path), default_timeout=60).run()
    assert not at.exception, [str(error.value) for error in at.exception]
    assert at.title, "page did not render a title"


def test_streamlit_replay_transport_buttons_advance_index() -> None:
    """Regression: transport buttons must not write a widget key after creation."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    path = project_root() / "app" / "pages" / "3_Replay_Mode.py"
    at = AppTest.from_file(str(path), default_timeout=60).run()
    assert not at.exception
    assert at.session_state["replay_index"] == 0

    at.button(key="replay_next").click().run()
    assert not at.exception, [str(error.value) for error in at.exception]
    assert at.session_state["replay_index"] == 1

    at.button(key="replay_last").click().run()
    assert not at.exception
    assert at.session_state["replay_index"] > 1

    at.button(key="replay_first").click().run()
    assert not at.exception
    assert at.session_state["replay_index"] == 0


def test_build_site_generates_self_contained_html(tmp_path: Path) -> None:
    """The static dashboard must embed values read from the result files."""
    builder = importlib.import_module("build_site")
    destination = tmp_path / "site" / "index.html"
    result = builder.build_site(destination)
    assert result == destination
    assert destination.is_file()
    page = destination.read_text(encoding="utf-8")
    assert "<!doctype html>" in page.lower()
    assert "reconstructed playback" in page.lower()
    assert "tier_served" in page
    assert "data:image/png;base64," in page
    assert "@CT_" not in page
    assert "<script src=" not in page  # no CDN dependency

    summary = pd.read_csv(project_root() / "results" / "full" / "summary.csv")
    first = summary.iloc[0]
    expected = mean_std_text(
        float(first["data_wait_ms_mean"]),
        float(first["data_wait_ms_std"]),
        "data_wait_ms",
    )
    assert expected in page

    sweep = pd.read_csv(project_root() / "results" / "full" / "sweep_summary.csv")
    sweep_expected = mean_std_text(
        float(sweep.iloc[0]["data_wait_ms_mean"]),
        float(sweep.iloc[0]["data_wait_ms_std"]),
        "data_wait_ms",
    )
    assert sweep_expected in page

    match = re.search(
        r'<script id="ct-data" type="application/json">(.*?)</script>', page, re.DOTALL
    )
    assert match is not None
    payload = json.loads(match.group(1))
    assert len(payload["summary"]["records"]) == len(summary)
    assert payload["summary"]["records"][0]["display"]["data_wait_ms"] == expected
    assert payload["replay"]["runs"]
    assert any("reconstructed playback" in note.lower() for note in [payload["playback_note"]])
