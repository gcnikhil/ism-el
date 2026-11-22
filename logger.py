"""Run-scoped trace and metric logging for ContextTier.

    The Logger is the ONLY place where evaluation-only ``true_type`` and
    ``true_temp`` columns are joined into a trace. Policies never supply them.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class Logger:
    """Thread-safe in-memory collector that writes CSV/Parquet on demand."""

    TRACE_COLUMNS: tuple[str, ...] = (
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
    TEMPERATURE_COLUMN = "true_temp"

    MIGRATION_COLUMNS: tuple[str, ...] = (
        "ts",
        "run_id",
        "seed",
        "scenario",
        "policy",
        "file_id",
        "size_bytes",
        "src_tier",
        "dst_tier",
        "kind",
    )

    def __init__(
        self,
        run_id: str,
        seed: int,
        scenario: str,
        policy: str,
        include_temperature: bool = False,
    ) -> None:
        """Create a logger bound to one run.

        Args:
            run_id: Unique run identifier.
            seed: RNG seed for this run.
            scenario: Scenario label (A, B, C or D).
            policy: Policy name.
        """
        self.run_id = run_id
        self.seed = seed
        self.scenario = scenario
        self.policy = policy
        self.include_temperature = bool(include_temperature)
        self._lock = threading.RLock()
        self._access_rows: list[dict[str, Any]] = []
        self._migration_rows: list[dict[str, Any]] = []
        self._metrics: list[dict[str, Any]] = []

    def log_access(self, **kwargs: Any) -> None:
        """Append one access row using exactly :attr:`TRACE_COLUMNS`.

        ``run_id``, ``seed``, ``scenario`` and ``policy`` default to the values
        bound at construction; every other column must be supplied.
        """
        row: dict[str, Any] = {
            "ts": kwargs.get("ts", time.perf_counter()),
            "run_id": kwargs.get("run_id", self.run_id),
            "seed": kwargs.get("seed", self.seed),
            "scenario": kwargs.get("scenario", self.scenario),
            "policy": kwargs.get("policy", self.policy),
        }
        required = (
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
        missing = [column for column in required if column not in kwargs]
        if missing:
            raise ValueError(f"log_access missing columns: {missing}")
        for column in required:
            row[column] = kwargs[column]
        if self.include_temperature:
            row["true_temp"] = kwargs.get("true_temp", "unknown")
        allowed = set(self.TRACE_COLUMNS)
        if self.include_temperature:
            allowed.add(self.TEMPERATURE_COLUMN)
        extra = set(kwargs) - allowed
        if extra:
            raise ValueError(f"log_access received unknown columns: {sorted(extra)}")
        with self._lock:
            columns = self.trace_columns
            self._access_rows.append({column: row.get(column, "unknown") for column in columns})

    @property
    def trace_columns(self) -> tuple[str, ...]:
        """Return the legacy schema or the additive temperature schema."""
        if self.include_temperature:
            return self.TRACE_COLUMNS + (self.TEMPERATURE_COLUMN,)
        return self.TRACE_COLUMNS

    def log_migration(self, **kwargs: Any) -> None:
        """Append one auxiliary migration row (not part of the main trace)."""
        row: dict[str, Any] = {
            "ts": kwargs.get("ts", time.perf_counter()),
            "run_id": kwargs.get("run_id", self.run_id),
            "seed": kwargs.get("seed", self.seed),
            "scenario": kwargs.get("scenario", self.scenario),
            "policy": kwargs.get("policy", self.policy),
        }
        required = ("file_id", "size_bytes", "src_tier", "dst_tier", "kind")
        missing = [column for column in required if column not in kwargs]
        if missing:
            raise ValueError(f"log_migration missing columns: {missing}")
        for column in required:
            row[column] = kwargs[column]
        extra = set(kwargs) - set(self.MIGRATION_COLUMNS)
        if extra:
            raise ValueError(f"log_migration received unknown columns: {sorted(extra)}")
        with self._lock:
            self._migration_rows.append(
                {column: row[column] for column in self.MIGRATION_COLUMNS}
            )

    def set_metric(self, name: str, value: float, **tags: Any) -> None:
        """Record a scalar metric with arbitrary run tags."""
        record: dict[str, Any] = {
            "name": name,
            "value": value,
            "run_id": self.run_id,
            "seed": self.seed,
            "scenario": self.scenario,
            "policy": self.policy,
        }
        record.update(tags)
        with self._lock:
            self._metrics.append(record)

    def trace_frame(self) -> pd.DataFrame:
        """Return the access trace as a DataFrame with exact columns."""
        with self._lock:
            rows = tuple(dict(row) for row in self._access_rows)
        return pd.DataFrame(rows, columns=list(self.trace_columns))

    def migration_frame(self) -> pd.DataFrame:
        """Return the auxiliary migration log as a DataFrame."""
        with self._lock:
            rows = tuple(dict(row) for row in self._migration_rows)
        return pd.DataFrame(rows, columns=list(self.MIGRATION_COLUMNS))

    def metrics_frame(self) -> pd.DataFrame:
        """Return all recorded metrics as a DataFrame."""
        with self._lock:
            rows = tuple(dict(row) for row in self._metrics)
        return pd.DataFrame(rows)

    def write_trace(self, path: str | Path, fmt: str = "parquet") -> Path:
        """Write the access trace as ``csv`` or ``parquet``.

        Args:
            path: Destination path.
            fmt: ``"parquet"`` (default) or ``"csv"``.

        Returns:
            The resolved destination path.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame = self.trace_frame()
        with self._lock:
            if fmt == "parquet":
                frame.to_parquet(destination, index=False)
            elif fmt == "csv":
                frame.to_csv(destination, index=False)
            else:
                raise ValueError(f"Unknown trace format: {fmt}")
        return destination

    def write_metrics(self, path: str | Path) -> Path:
        """Write all recorded metrics to a CSV file."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_frame().to_csv(destination, index=False)
        return destination

    def write_migration_log(self, path: str | Path) -> Path:
        """Write the auxiliary migration log to a CSV file."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.migration_frame().to_csv(destination, index=False)
        return destination
