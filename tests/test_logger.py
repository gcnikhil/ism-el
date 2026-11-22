"""Logger schema and output-format tests."""

from __future__ import annotations

import pandas as pd
import pytest

from logger import Logger


def make_logger() -> Logger:
    """Create a logger bound to a dummy run."""
    return Logger(run_id="r0", seed=0, scenario="A", policy="lru")


def log_one(logger: Logger, **overrides: object) -> None:
    """Append one fully populated access row."""
    row: dict[str, object] = {
        "ts": 12.5,
        "file_id": 7,
        "size_bytes": 150_000,
        "op": "read",
        "epoch": 1,
        "step": 42,
        "phase": "train",
        "tier_served": 0,
        "latency_ms": 0.06,
        "wait_ms": 1.5,
        "fast_hit": True,
        "true_type": "chunk",
    }
    row.update(overrides)
    logger.log_access(**row)


def test_trace_columns_are_exact() -> None:
    assert Logger.TRACE_COLUMNS == (
        "ts",
        "run_id",
        "seed",
        "scenario",
        "policy",
        "file_id",
        "size_bytes",
        "op",
        "epoch",
        "step",
        "phase",
        "tier_served",
        "latency_ms",
        "wait_ms",
        "fast_hit",
        "true_type",
    )


def test_logger_writes_csv_and_parquet(tmp_path) -> None:
    logger = make_logger()
    log_one(logger)
    logger.set_metric("data_wait_ms", 1.5)

    csv_path = logger.write_trace(tmp_path / "trace.csv", fmt="csv")
    parquet_path = logger.write_trace(tmp_path / "trace.parquet", fmt="parquet")
    metrics_path = logger.write_metrics(tmp_path / "metrics.csv")

    csv_frame = pd.read_csv(csv_path)
    parquet_frame = pd.read_parquet(parquet_path)
    metrics_frame = pd.read_csv(metrics_path)

    for frame in (csv_frame, parquet_frame):
        assert tuple(frame.columns) == Logger.TRACE_COLUMNS
        assert len(frame) == 1
        assert int(frame.iloc[0]["file_id"]) == 7
        assert str(frame.iloc[0]["true_type"]) == "chunk"
        assert bool(frame.iloc[0]["fast_hit"]) is True
    assert tuple(metrics_frame["name"]) == ("data_wait_ms",)
    assert float(metrics_frame.iloc[0]["value"]) == 1.5


def test_logger_rejects_missing_and_unknown_columns() -> None:
    logger = make_logger()
    with pytest.raises(ValueError, match="missing columns"):
        logger.log_access(file_id=1)
    with pytest.raises(ValueError, match="unknown columns"):
        log_one(logger, secret_type="checkpoint")
