"""Phase 12 regression tests for selectable CPU training models."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from config import load_config
from train import (
    TinyCNN,
    TinyMLP,
    apply_model_selection,
    available_models,
    build_model,
    checkpoint_bytes,
    validate_profiles,
)
from simulator import run_simulation


def test_model_registry_builds_cnn_and_mlp() -> None:
    assert available_models() == ("cnn", "mlp")
    assert isinstance(build_model("cnn"), TinyCNN)
    assert isinstance(build_model("MLP"), TinyMLP)


def test_unknown_model_has_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown model 'unknown'.*cnn, mlp"):
        build_model("unknown")


def test_cnn_manual_selection_preserves_baseline_knobs() -> None:
    config = load_config()
    selected = apply_model_selection(config, name="cnn", compute_source="manual")
    assert selected["compute_ms"] == config["compute_ms"]
    assert selected["ckpt_size_mb"] == config["ckpt_size_mb"]
    assert selected is not config


def test_profile_selection_reads_profile_values(tmp_path) -> None:
    profile = tmp_path / "profiles.json"
    profile.write_text(
        json.dumps(
            {
                "profiles": {
                    name: {
                        "parameter_count": 1,
                        "median_step_ms": 12.5 if name == "mlp" else 10.0,
                        "p95_step_ms": 13.0,
                        "checkpoint_bytes": 3 * 1024**2,
                        "peak_ram_bytes": 1,
                    }
                    for name in available_models()
                },
                "schema_version": 1,
                "device": "cpu",
                "batch_size": 128,
                "warmup_steps": 20,
                "timed_steps": 200,
            }
        ),
        encoding="utf-8",
    )
    config = load_config()
    config["model"] = {"name": "mlp", "compute_source": "profile", "profile_path": str(profile)}
    selected = apply_model_selection(config)
    assert selected["compute_ms"] == 12.5
    assert selected["ckpt_size_mb"] == 3.0
    assert selected["classifier"]["model_path"].startswith("results/models/mlp/")
    assert selected["embeddings"]["report_dir"] == "results/models/mlp/embeddings"
    assert selected["rl"]["model_path"].startswith("results/models/mlp/")


def test_checkpoint_bytes_includes_model_and_optimizer() -> None:
    import torch

    model = build_model("cnn")
    optimizer = torch.optim.Adam(model.parameters())
    assert checkpoint_bytes(model, optimizer) > 0


def test_profile_schema_validation() -> None:
    profile = {
        "schema_version": 1,
        "device": "cpu",
        "batch_size": 128,
        "warmup_steps": 20,
        "timed_steps": 200,
        "profiles": {
            name: {
                "parameter_count": 1,
                "median_step_ms": 1.0,
                "p95_step_ms": 2.0,
                "checkpoint_bytes": 1,
                "peak_ram_bytes": 1,
            }
            for name in available_models()
        },
    }
    validate_profiles(profile)
    del profile["profiles"]["cnn"]["p95_step_ms"]
    with pytest.raises(ValueError, match="p95_step_ms"):
        validate_profiles(profile)


def test_cnn_manual_virtual_regression_matches_preserved_matrix() -> None:
    """The Phase-12 switch must not perturb the published CNN baseline."""
    config = apply_model_selection(load_config(), name="cnn", compute_source="manual")
    actual = run_simulation(config, "lru", "A", 0).metrics
    raw = pd.read_csv("results/full/raw_runs.csv")
    expected = raw[
        (raw["sweep"] == "matrix")
        & (raw["policy"] == "lru")
        & (raw["scenario"] == "A")
        & (raw["seed"] == 0)
    ].iloc[0]
    for metric in ("data_wait_ms", "throughput", "migration_bytes", "elapsed_ms"):
        assert actual[metric] == pytest.approx(float(expected[metric]), rel=0.0, abs=1e-9)
