"""Observable features for file-type inference (Phase 5).

Every feature is computed from ``AccessEvent`` plus the policy's ``PolicyView``
and its own per-file history. No true type, opaque-name, manifest path or future
read order is available here. The harness uses the same extractor at training
time (offline, with evaluation-only labels) and inside ``ClassifierPolicy`` at
inference time.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from tiers import FAST, MID

FEATURE_NAMES: list[str] = [
    "size_kb",
    "log2_size",
    "is_write",
    "phase_warmup",
    "phase_train",
    "phase_validation",
    "phase_checkpoint",
    "epoch",
    "step",
    "read_count_log",
    "write_count_log",
    "step_gap_log",
    "tier_count",
    "min_tier",
    "max_tier",
    "in_fast",
    "in_mid",
    "fast_occupied_frac",
    "mid_occupied_frac",
    "fast_free_frac",
]

_PHASES = ("warmup", "train", "validation", "checkpoint")


@dataclass
class _FileStats:
    """Per-file counters kept in the policy's private history."""

    reads: int = 0
    writes: int = 0
    last_step_index: int = -1


class FeatureExtractor:
    """Build one feature vector per access from observable state only."""

    def __init__(self) -> None:
        """Create an extractor with empty per-file history."""
        self._stats: dict[int, _FileStats] = {}

    @staticmethod
    def _fraction(used: int, capacity: int | None) -> float:
        """Return a bounded occupancy fraction for one tier."""
        total = used if capacity is None else used + capacity
        if total <= 0:
            return 0.0
        return min(1.0, used / total)

    def extract(self, event, view) -> dict[str, float]:
        """Return the feature vector for ``event`` given ``view``."""
        stats = self._stats.get(event.file_id)
        if stats is None:
            stats = _FileStats()
            self._stats[event.file_id] = stats

        step_index = event.epoch * 1_000_000 + event.step
        gap = 0 if stats.last_step_index < 0 else max(0, step_index - stats.last_step_index)

        if event.op == "read":
            stats.reads += 1
        else:
            stats.writes += 1
        stats.last_step_index = step_index

        holders = tuple(view.holds(event.file_id))
        occupancy = view.occupancy()
        free = view.free_space()
        fast_used = int(occupancy.get(FAST, 0))
        mid_used = int(occupancy.get(1, 0))
        fast_free = free.get(FAST)
        mid_free = free.get(1)

        size = float(event.size_bytes)
        features: dict[str, float] = {
            "size_kb": size / 1024.0,
            "log2_size": math.log2(size + 1.0),
            "is_write": 1.0 if event.op == "write" else 0.0,
            "phase_warmup": 1.0 if event.phase == "warmup" else 0.0,
            "phase_train": 1.0 if event.phase == "train" else 0.0,
            "phase_validation": 1.0 if event.phase == "validation" else 0.0,
            "phase_checkpoint": 1.0 if event.phase == "checkpoint" else 0.0,
            "epoch": float(event.epoch),
            "step": float(event.step),
            "read_count_log": math.log1p(stats.reads),
            "write_count_log": math.log1p(stats.writes),
            "step_gap_log": math.log1p(gap),
            "tier_count": float(len(holders)),
            "min_tier": float(min(holders)) if holders else 3.0,
            "max_tier": float(max(holders)) if holders else 3.0,
            "in_fast": 1.0 if FAST in holders else 0.0,
            "in_mid": 1.0 if MID in holders else 0.0,
            "fast_occupied_frac": self._fraction(
                fast_used,
                None if fast_free is None else int(fast_free),
            ),
            "mid_occupied_frac": self._fraction(
                mid_used,
                None if mid_free is None else int(mid_free),
            ),
            "fast_free_frac": (
                0.0
                if fast_free is None
                else min(1.0, max(0.0, float(fast_free) / max(1.0, float(fast_used + fast_free))))
            ),
        }
        return features


def feature_vector(features: dict[str, float]) -> list[float]:
    """Order a feature dictionary into the model's expected column order."""
    return [float(features[name]) for name in FEATURE_NAMES]
