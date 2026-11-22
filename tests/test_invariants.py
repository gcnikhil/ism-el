"""Sanity invariants: AllFast <= Belady <= others <= AllSlow on every scenario."""

from __future__ import annotations

import pytest

from config import load_config
from simulator import run_simulation

# This is the Phase-2 acceptance invariant for the original baselines only.
# Learned / privileged policies added later can beat Belady on data_wait_ms:
# Belady minimises miss count, not data wait, and ignores migration overhead.
POLICIES = ["allslow", "allfast", "lru", "lfu", "arc", "clairvoyant", "belady"]
SCENARIOS = ["A", "B", "C", "D"]
TOLERANCE_MS = 1.0


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_data_wait_ordering(scenario: str) -> None:
    """Run every policy once and check the Phase-2 acceptance ordering."""
    config = load_config()
    waits = {
        policy: run_simulation(config, policy, scenario, seed=0, epochs=1).metrics[
            "data_wait_ms"
        ]
        for policy in POLICIES
    }

    assert waits["allfast"] <= waits["belady"] + TOLERANCE_MS
    others_min = min(
        waits[policy]
        for policy in POLICIES
        if policy not in {"allfast", "belady"}
    )
    assert waits["belady"] <= others_min + TOLERANCE_MS
    assert others_min <= waits["allslow"] + TOLERANCE_MS
