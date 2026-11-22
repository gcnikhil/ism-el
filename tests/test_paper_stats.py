"""Tests for the paper's post-processing statistics (all inputs synthetic)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from config import load_config, project_root

from paper_stats import (
    DECOMPOSITION_COLUMNS,
    best_baseline_table,
    bootstrap_tables,
    decomposition_table,
    paired_bootstrap,
)


def test_compute_ms_is_measured_not_placeholder() -> None:
    config = load_config()
    benchmark_path = project_root() / "results" / "real" / "step_time_benchmark.json"
    data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    assert float(config["compute_ms"]) == pytest.approx(
        float(data["measurement"]["median_ms"]), rel=0.0, abs=1e-12
    )
    assert int(data["measurement"]["steps_measured"]) == 200
    assert len(config["seeds"]) == 20


def test_paired_bootstrap_recovers_known_interval() -> None:
    differences = np.array([10.0, 11.0, 9.0, 10.0, 12.0, 8.0])
    mean, low, high = paired_bootstrap(differences, resamples=2000, seed=7)
    assert mean == pytest.approx(10.0)
    assert low < mean < high
    assert low >= differences.min()
    assert high <= differences.max()


def test_bootstrap_tables_report_wins_and_direction() -> None:
    raw = pd.DataFrame(
        {
            "scenario": ["C"] * 4,
            "sweep": ["matrix"] * 4,
            "sweep_value": [None] * 4,
            "policy": ["lru", "lfu", "full_context", "full_context"],
            "seed": [0, 0, 0, 1],
            "data_wait_ms": [100.0, 110.0, 90.0, 95.0],
            "throughput": [10.0, 9.0, 11.0, 10.5],
        }
    )
    # Give LFU a second paired seed so the frame has two complete pairs.
    raw = pd.concat(
        [
            raw,
            pd.DataFrame(
                {
                    "scenario": ["C", "C"],
                    "sweep": ["matrix", "matrix"],
                    "sweep_value": [None, None],
                    "policy": ["lru", "lfu"],
                    "seed": [1, 1],
                    "data_wait_ms": [105.0, 115.0],
                    "throughput": [10.2, 9.1],
                }
            ),
        ],
        ignore_index=True,
    )
    table, meta = bootstrap_tables(raw, resamples=500, seed=0)
    subset = table[
        (table["scenario"] == "C")
        & (table["baseline"] == "lru")
        & (table["sweep"] == "matrix")
    ]
    data_wait = subset[subset["metric"] == "data_wait_ms"].iloc[0]
    throughput = subset[subset["metric"] == "throughput"].iloc[0]
    assert data_wait["n_seeds"] == 2
    assert data_wait["mean_difference"] == pytest.approx(10.0)
    assert data_wait["seeds_won"] == 2
    assert throughput["mean_difference"] == pytest.approx(0.65)
    assert throughput["seeds_won"] == 2
    assert "p_values" in meta


def test_decomposition_table_uses_generated_columns() -> None:
    columns = ["policy", "policy_kind", "scenario", "n_seeds"]
    for name in DECOMPOSITION_COLUMNS:
        columns.append(f"{name}_mean")
    columns.extend(["elapsed_ms_mean", "data_wait_ms_mean", "throughput_mean"])
    rows = []
    for scenario in ("B", "C"):
        rows.append(
            {
                "policy": "lru",
                "policy_kind": "observable_baseline",
                "scenario": scenario,
                "n_seeds": 20,
                **{f"{name}_mean": 1.0 for name in DECOMPOSITION_COLUMNS},
                "elapsed_ms_mean": float(len(DECOMPOSITION_COLUMNS)),
                "data_wait_ms_mean": 0.0,
                "throughput_mean": 100.0,
            }
        )
    frame = decomposition_table(pd.DataFrame(rows))
    assert set(frame["scenario"]) == {"B", "C"}
    assert np.allclose(frame["residual_ms"].to_numpy(), 0.0)


def test_best_baseline_excludes_capacity_controls() -> None:
    summary = pd.DataFrame(
        {
            "policy": ["allslow", "allfast", "lru", "lfu", "arc", "full_context"],
            "scenario": ["C"] * 6,
            "data_wait_ms_mean": [1000.0, 0.0, 100.0, 90.0, 95.0, 80.0],
            "data_wait_ms_std": [0.0] * 6,
            "throughput_mean": [1.0, 99.0, 10.0, 11.0, 10.5, 12.0],
            "n_seeds": [20] * 6,
        }
    )
    table, meta = best_baseline_table(summary)
    assert meta["best_baseline"] == "lfu"
    assert table.iloc[0]["data_wait_saved_ms"] == pytest.approx(10.0)
    assert set(table.iloc[0]["candidate_baselines"]) == {"lru", "lfu", "arc"}
