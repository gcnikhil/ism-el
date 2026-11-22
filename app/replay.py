"""Deterministic step-through replay of saved ContextTier traces.

The replay engine uses only the two CSVs written by :class:`logger.Logger`:
the access trace and the migration log. It reconstructs *approximate* tier
residency by starting from the known initial placement and applying completed
migrations in timestamp order. The authoritative served tier of each access is
the trace's own ``tier_served`` column; in-flight prefetches are not present in
the migration log and are therefore not shown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from app.data import TIER_NAMES
from logger import Logger
from tiers import FAST, SLOW

MIB: float = 1024.0 * 1024.0


@dataclass(frozen=True)
class ReplaySnapshot:
    """Everything the UI needs to draw one access of a saved run."""

    index: int
    access: Mapping[str, Any]
    tier_by_file: Mapping[int, int]
    occupancy_bytes: Mapping[int, int]
    migrations_applied: int
    migration_bytes: int
    cumulative_wait_ms: float
    cumulative_latency_ms: float
    fast_hits: int

    @property
    def served_tier(self) -> int:
        """Tier that actually served the current access (from the trace)."""
        return int(self.access["tier_served"])

    @property
    def served_tier_name(self) -> str:
        """Display name of the tier that served the current access."""
        return TIER_NAMES.get(self.served_tier, str(self.served_tier))

    @property
    def fast_hit(self) -> bool:
        """Whether the current access was a FAST hit."""
        return bool(self.access["fast_hit"])


class TraceReplay:
    """Replay one trace with deterministic, timestamp-ordered state changes."""

    def __init__(self, trace: pd.DataFrame, migrations: pd.DataFrame | None = None) -> None:
        missing = [column for column in Logger.TRACE_COLUMNS if column not in trace.columns]
        if missing:
            raise ValueError(f"Trace is missing required columns: {missing}")
        self.trace: pd.DataFrame = trace.sort_values("ts", kind="stable").reset_index(drop=True)

        if migrations is None or migrations.empty:
            migrations = pd.DataFrame(columns=list(Logger.MIGRATION_COLUMNS))
        self.migrations: pd.DataFrame = migrations.sort_values("ts", kind="stable").reset_index(
            drop=True
        )

        file_ids = set(self.trace["file_id"].astype(int))
        file_ids.update(self.migrations["file_id"].astype(int))
        self.file_sizes: dict[int, int] = {}
        for frame in (self.trace, self.migrations):
            for file_id, size in zip(frame["file_id"], frame["size_bytes"]):
                self.file_sizes[int(file_id)] = int(size)
        self.file_ids: tuple[int, ...] = tuple(sorted(file_ids))

        first = self.trace.iloc[0]
        self.initial_tier: int = (
            FAST if str(first["scenario"]) == "D" or str(first["policy"]) == "allfast" else SLOW
        )

        self._wait_prefix = np.concatenate(
            ([0.0], np.cumsum(self.trace["wait_ms"].to_numpy(dtype=float)))
        )
        self._latency_prefix = np.concatenate(
            ([0.0], np.cumsum(self.trace["latency_ms"].to_numpy(dtype=float)))
        )
        self._fast_prefix = np.concatenate(
            ([0], np.cumsum(self.trace["fast_hit"].astype(bool).to_numpy(dtype=np.int64)))
        )
        self._migration_ts = self.migrations["ts"].to_numpy(dtype=float) if len(self.migrations) else np.array([])

    def __len__(self) -> int:
        """Number of accesses in the trace."""
        return int(len(self.trace))

    @property
    def initial_placement(self) -> str:
        """Human-readable starting tier for all files."""
        return TIER_NAMES.get(self.initial_tier, str(self.initial_tier))

    def state_at(self, index: int) -> ReplaySnapshot:
        """Return the reconstructed state just after access ``index``."""
        if not 0 <= index < len(self):
            raise IndexError(f"Access index {index} outside 0..{len(self) - 1}")
        access = self.trace.iloc[index]
        cutoff = float(access["ts"])

        tier_by_file = {file_id: self.initial_tier for file_id in self.file_ids}
        migrations_applied = 0
        migration_bytes = 0
        for row in self.migrations.itertuples(index=False):
            if float(row.ts) > cutoff:
                break
            file_id = int(row.file_id)
            tier_by_file[file_id] = int(row.dst_tier)
            migration_bytes += int(row.size_bytes)
            migrations_applied += 1

        occupancy: dict[int, int] = {}
        for file_id, tier in tier_by_file.items():
            occupancy[tier] = occupancy.get(tier, 0) + self.file_sizes.get(file_id, 0)

        return ReplaySnapshot(
            index=index,
            access=access.to_dict(),
            tier_by_file=tier_by_file,
            occupancy_bytes=dict(sorted(occupancy.items())),
            migrations_applied=migrations_applied,
            migration_bytes=migration_bytes,
            cumulative_wait_ms=float(self._wait_prefix[index + 1]),
            cumulative_latency_ms=float(self._latency_prefix[index + 1]),
            fast_hits=int(self._fast_prefix[index + 1]),
        )

    def wait_series(self, upto_index: int) -> pd.DataFrame:
        """Return per-access wait values up to and including ``upto_index``."""
        end = max(0, min(upto_index + 1, len(self)))
        return pd.DataFrame(
            {
                "access": np.arange(end),
                "wait_ms": self.trace["wait_ms"].to_numpy(dtype=float)[:end],
            }
        )

    def recent_migrations(self, index: int, count: int = 8) -> pd.DataFrame:
        """Return the last ``count`` completed migrations before an access."""
        cutoff = float(self.trace.iloc[index]["ts"])
        valid = self.migrations[self.migrations["ts"] <= cutoff]
        return valid.tail(count).iloc[::-1].reset_index(drop=True)


def replay_from_frames(trace: pd.DataFrame, migrations: pd.DataFrame | None = None) -> TraceReplay:
    """Convenience constructor for tests and callers that already have frames."""
    return TraceReplay(trace, migrations)
