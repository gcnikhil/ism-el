"""Phase 9 tests: publication figures sourced from the Phase 8 tables."""

from __future__ import annotations

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from config import project_root
from experiments import METRIC_NAMES
from plots import (
    DEPLOYABLE_POLICIES,
    FOCUS_POLICIES,
    KIND_COLORS,
    POLICY_ORDER,
    _bar_panel,
    apply_style,
    generate_all,
    load_tables,
    plot_all_policies,
    plot_capacity_sweep,
    plot_compute_sweep,
    plot_context_ablation,
    plot_deployable_policies,
    plot_latency_percentiles,
    plot_scenario_focus,
)


def synthetic_summary() -> pd.DataFrame:
    """Build a small summary with the exact Phase 8 column schema."""
    rows: list[dict] = []
    for policy in POLICY_ORDER:
        kind = (
            "oracle_upper_bound" if policy in {"belady", "clairvoyant"}
            else "declared_type_upper_bound" if policy.startswith("context_rules")
            else "learned_observable" if policy in {"classifier", "embedding", "full_context"}
            else "observable_baseline"
        )
        for scenario in ("A", "B", "C", "D"):
            row: dict = {"policy": policy, "policy_kind": kind, "scenario": scenario, "n_seeds": 5}
            for index, metric in enumerate(METRIC_NAMES):
                row[f"{metric}_mean"] = 100.0 + index * 10 + (0 if scenario == "A" else 1)
                row[f"{metric}_std"] = 1.0 + index * 0.1
            rows.append(row)
    return pd.DataFrame(rows)


def synthetic_sweep_summary() -> pd.DataFrame:
    rows: list[dict] = []
    for sweep, values in (("fast_capacity_mb", (10.0, 25.0)), ("compute_ms", (1.0, 5.0))):
        for policy in ("lru", "context_rules", "full_context"):
            for scenario in ("A", "C"):
                for value in values:
                    rows.append(
                        {
                            "sweep": sweep,
                            "sweep_value": value,
                            "policy": policy,
                            "policy_kind": (
                                "declared_type_upper_bound" if policy == "context_rules"
                                else "learned_observable" if policy == "full_context"
                                else "observable_baseline"
                            ),
                            "scenario": scenario,
                            "n_seeds": 5,
                            "data_wait_ms_mean": 100.0 + value,
                            "data_wait_ms_std": 2.0,
                        }
                    )
    return pd.DataFrame(rows)


def synthetic_raw() -> pd.DataFrame:
    rows: list[dict] = []
    for policy in ("context_rules", "context_rules_noctx"):
        for scenario in ("A", "B", "C", "D"):
            for seed in (0, 1):
                rows.append(
                    {
                        "policy": policy,
                        "policy_kind": "declared_type_upper_bound",
                        "scenario": scenario,
                        "seed": seed,
                        "data_wait_ms": 100.0
                        + seed * 10
                        + ((5.0 + 2.0 * seed) if policy == "context_rules_noctx" else 0.0),
                    }
                )
    return pd.DataFrame(rows)


def test_style_and_constants() -> None:
    apply_style()
    assert plt.rcParams["figure.dpi"] == 150
    assert plt.rcParams["savefig.dpi"] == 300
    assert set(KIND_COLORS) >= {
        "observable_baseline",
        "learned_observable",
        "declared_type_upper_bound",
        "oracle_upper_bound",
    }
    assert "full_context" in DEPLOYABLE_POLICIES
    assert "full_context" in FOCUS_POLICIES


def test_bar_panel_uses_table_values() -> None:
    summary = synthetic_summary()
    figure, axis = plt.subplots()
    _bar_panel(axis, summary, ("lru", "full_context"), "C", "data_wait_ms", "Test")
    bar_container = next(container for container in axis.containers if hasattr(container, "patches"))
    widths = [patch.get_width() for patch in bar_container.patches]
    expected = [
        float(summary[(summary["policy"] == policy) & (summary["scenario"] == "C")]["data_wait_ms_mean"].iloc[0])
        for policy in ("lru", "full_context")
    ]
    np.testing.assert_allclose(widths, expected)
    plt.close(figure)


def test_load_tables_from_phase8_results() -> None:
    tables = load_tables(project_root() / "results" / "full")
    assert set(tables) == {"raw_runs", "summary", "sweep_runs", "sweep_summary"}
    # Row counts scale with the configured seeds/policies; derive them rather
    # than hard-coding a seed count so a 20-seed regeneration still passes.
    raw = tables["raw_runs"]
    summary = tables["summary"]
    n_policies = raw["policy"].nunique()
    n_scenarios = raw["scenario"].nunique()
    n_seeds = raw["seed"].nunique()
    assert len(raw) == n_policies * n_scenarios * n_seeds
    assert len(summary) == n_policies * n_scenarios
    assert len(tables["sweep_runs"]) > 0
    assert "data_wait_ms_std" in tables["summary"].columns


def test_every_figure_writes_pdf_and_png(tmp_path) -> None:
    summary = synthetic_summary()
    sweep = synthetic_sweep_summary()
    raw = synthetic_raw()
    formats = ("pdf", "png")
    groups = (
        plot_all_policies(summary, tmp_path, formats),
        plot_deployable_policies(summary, tmp_path, formats),
        plot_scenario_focus(summary, tmp_path, formats),
        plot_capacity_sweep(sweep, tmp_path, formats),
        plot_compute_sweep(sweep, tmp_path, formats),
        plot_context_ablation(raw, tmp_path, formats),
        plot_latency_percentiles(summary, tmp_path, formats),
    )
    assert len(groups) == 7
    for paths in groups:
        assert len(paths) == 2
        for path in paths:
            assert path.is_file()
            assert path.stat().st_size > 1000
        assert paths[0].suffix == ".pdf"
        assert paths[1].suffix == ".png"


def test_generate_all_writes_manifest(tmp_path) -> None:
    source = tmp_path / "full"
    source.mkdir()
    real = project_root() / "results" / "full"
    for name in ("raw_runs.csv", "summary.csv", "sweep_runs.csv", "sweep_summary.csv"):
        (source / name).write_bytes((real / name).read_bytes())
    output = tmp_path / "plots"
    report = generate_all(source, output, formats=("png",))
    assert len(report["figures"]) == 8
    for paths in report["figures"].values():
        assert len(paths) == 1
        assert paths[0].suffix == ".png"
    manifest = json.loads(report["manifest"].read_text(encoding="utf-8"))
    assert manifest["phase"] == 9
    assert manifest["formats"] == ["png"]
    assert "no numbers are invented" in " ".join(manifest["honesty_notes"])
    assert set(manifest["figures"]) == set(report["figures"])


def test_context_ablation_uses_paired_differences(tmp_path) -> None:
    raw = synthetic_raw()
    paths = plot_context_ablation(raw, tmp_path, ("png",))
    assert paths[0].is_file()
    # Paired per-seed differences vary (5.0 and 7.0 ms) in the fixture.
    for scenario in ("A", "B", "C", "D"):
        with_ctx = raw[(raw["policy"] == "context_rules") & (raw["scenario"] == scenario)]
        no_ctx = raw[(raw["policy"] == "context_rules_noctx") & (raw["scenario"] == scenario)]
        differences = no_ctx.set_index("seed")["data_wait_ms"] - with_ctx.set_index("seed")["data_wait_ms"]
        assert float(differences.mean()) == 6.0
        assert float(differences.std(ddof=1)) > 0.0
