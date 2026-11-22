"""Phase 8 tests: full experiment matrix, aggregation and sweeps."""

from __future__ import annotations

import json

import pandas as pd

from config import load_config
from experiments import (
    METRIC_NAMES,
    ROW_COLUMNS,
    LearnedModels,
    _copy_with_compute_ms,
    _copy_with_fast_capacity,
    policy_kind,
    run_all,
    run_matrix,
    summarize,
)
from simulator import run_simulation
from tiers import FAST


def test_metric_names_match_simulator_output() -> None:
    config = load_config()
    result = run_simulation(config, "lru", "A", 0, epochs=1, chunk_limit=5)
    assert set(METRIC_NAMES) == set(result.metrics)


def test_policy_kind_labels() -> None:
    assert policy_kind("belady") == "oracle_upper_bound"
    assert policy_kind("clairvoyant") == "oracle_upper_bound"
    assert policy_kind("context_rules") == "declared_type_upper_bound"
    assert policy_kind("context_rules_noctx") == "declared_type_upper_bound"
    assert policy_kind("classifier") == "learned_observable"
    assert policy_kind("full_context") == "learned_observable"
    assert policy_kind("lru") == "observable_baseline"


def test_summarize_computes_mean_and_sample_std() -> None:
    raw = pd.DataFrame(
        {
            "policy": ["lru", "lru", "lru"],
            "scenario": ["A", "A", "A"],
            "seed": [0, 1, 2],
            "data_wait_ms": [10.0, 20.0, 30.0],
        }
    )
    # Patch the metric list to the single column present in this fixture.
    import experiments

    original = experiments.METRIC_NAMES
    experiments.METRIC_NAMES = ("data_wait_ms",)
    try:
        frame = summarize(raw)
    finally:
        experiments.METRIC_NAMES = original
    assert frame.loc[0, "n_seeds"] == 3
    assert frame.loc[0, "data_wait_ms_mean"] == 20.0
    assert abs(frame.loc[0, "data_wait_ms_std"] - 10.0) < 1e-9


def test_summarize_empty_frame_has_full_columns() -> None:
    frame = summarize(pd.DataFrame(columns=list(ROW_COLUMNS)))
    assert "data_wait_ms_mean" in frame.columns
    assert "latency_p99_ms_std" in frame.columns
    assert frame.empty


def test_config_overrides_do_not_mutate_original() -> None:
    config = load_config()
    original_capacity = next(t["capacity_mb"] for t in config["tiers"] if t["id"] == FAST)
    original_compute = config["compute_ms"]
    capacity_config = _copy_with_fast_capacity(config, 7.0)
    compute_config = _copy_with_compute_ms(config, 2.0)
    assert next(t["capacity_mb"] for t in capacity_config["tiers"] if t["id"] == FAST) == 7.0
    assert compute_config["compute_ms"] == 2.0
    assert next(t["capacity_mb"] for t in config["tiers"] if t["id"] == FAST) == original_capacity
    assert config["compute_ms"] == original_compute


def test_learned_models_availability_without_loading() -> None:
    config = load_config()
    models = LearnedModels(config, load=False)
    assert not models.available("classifier")
    assert not models.available("embedding")
    assert not models.available("full_context")
    assert models.available("lru")
    assert models.kwargs("lru") == {}


def test_run_matrix_rows_have_all_metrics(tmp_path) -> None:
    config = load_config()
    models = LearnedModels(config, load=False)
    frame = run_matrix(
        config,
        ["lru", "context_rules"],
        ["A", "C"],
        [0, 1],
        epochs=1,
        chunk_limit=5,
        models=models,
    )
    assert len(frame) == 8
    assert set(ROW_COLUMNS) <= set(frame.columns)
    assert frame["policy_kind"].eq("observable_baseline").sum() == 4
    assert frame["policy_kind"].eq("declared_type_upper_bound").sum() == 4
    assert frame["data_wait_ms"].ge(0.0).all()


def test_run_all_writes_matrix_artifacts(tmp_path) -> None:
    config = load_config()
    output = tmp_path / "full"
    report = run_all(
        config,
        output_dir=output,
        policies=["lru", "context_rules"],
        scenarios=["A", "C"],
        seeds=[0, 1],
        epochs=1,
        chunk_limit=5,
        matrix=True,
        sweeps=False,
        load_models=False,
    )
    assert report["matrix_runs"] == 8
    assert report["sweep_runs"] == 0
    assert (output / "raw_runs.csv").is_file()
    assert (output / "summary.csv").is_file()
    assert (output / "experiment_manifest.json").is_file()
    manifest = json.loads((output / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["phase"] == 8
    assert "NOT retrained at each sweep point" in " ".join(manifest["honesty_notes"])


def test_run_all_writes_sweep_artifacts(tmp_path) -> None:
    config = load_config()
    output = tmp_path / "sweeps"
    report = run_all(
        config,
        output_dir=output,
        policies=["lru", "context_rules"],
        scenarios=["A", "C"],
        seeds=[0, 1],
        epochs=1,
        chunk_limit=5,
        matrix=False,
        sweeps=True,
        load_models=False,
        fast_capacities_mb=[10.0, 25.0],
        compute_values_ms=[1.0, 5.0],
    )
    assert report["matrix_runs"] == 0
    assert report["sweep_runs"] == 16  # 2 policies x 2 settings x 2 sweeps x 2 seeds
    sweeps = pd.read_csv(output / "sweep_runs.csv")
    assert set(sweeps["sweep"]) == {"fast_capacity_mb", "compute_ms"}
    assert set(sweeps[sweeps["sweep"] == "fast_capacity_mb"]["sweep_value"]) == {10.0, 25.0}
    assert set(sweeps[sweeps["sweep"] == "compute_ms"]["sweep_value"]) == {1.0, 5.0}
    summary = pd.read_csv(output / "sweep_summary.csv")
    assert {"sweep", "sweep_value", "policy", "scenario"} <= set(summary.columns)
    assert "data_wait_ms_mean" in summary.columns
