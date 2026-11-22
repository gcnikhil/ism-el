"""Final-paper checks: arrival-time fast_hit and exact elapsed decomposition."""

from __future__ import annotations

import pytest

from config import load_config
from experiments import METRIC_NAMES
from simulator import run_simulation
from store import TieredStore, VirtualClock
from tiers import FAST, MID, SLOW, Tier, TierSpec
from policies import AccessEvent


def _store() -> TieredStore:
    tiers = {
        FAST: Tier(TierSpec(FAST, "fast", 100 * 1024 * 1024, 0.01, 3000)),
        MID: Tier(TierSpec(MID, "mid", 100 * 1024 * 1024, 0.2, 500)),
        SLOW: Tier(TierSpec(SLOW, "slow", None, 8.0, 80)),
    }
    return TieredStore(tiers, VirtualClock(), max_queue=8)


def _event(file_id: int, size: int) -> AccessEvent:
    return AccessEvent(
        file_id=file_id,
        size_bytes=size,
        op="read",
        epoch=0,
        step=0,
        phase="train",
    )


def test_store_fast_hit_uses_arrival_snapshot() -> None:
    store = _store()
    try:
        store.register(7, 100_000)
        assert store.fast_resident(7) is False
        first = store.get(7, _event(7, 100_000))
        assert first.tier_served == SLOW
        assert first.fast_hit is False

        store.promote(7, FAST)
        assert store.fast_resident(7) is True
        second = store.get(7, _event(7, 100_000))
        assert second.tier_served == FAST
        assert second.fast_hit is True

        # Explicit snapshot wins: a demand promotion must not turn the same
        # read into a FAST hit.
        third = store.get(7, _event(7, 100_000), arrival_fast=False)
        assert third.tier_served == FAST
        assert third.fast_hit is False
    finally:
        store.close()


def test_simulator_fast_hit_is_arrival_residency_not_service_tier() -> None:
    config = load_config()
    result = run_simulation(config, "lru", "A", 0, epochs=1, chunk_limit=5)
    trace = result.logger.trace_frame()
    reads = trace[trace["op"] == "read"]
    assert not reads.empty
    first_read = reads.iloc[0]
    # LRU promotes synchronously on this miss, so the read is *served* by FAST
    # but was not resident in FAST when it arrived.
    assert int(first_read["tier_served"]) == FAST
    assert bool(first_read["fast_hit"]) is False
    # Later reads of the same cached chunk are genuine arrival-time FAST hits.
    assert reads["fast_hit"].astype(bool).any()
    assert 0.0 < float(reads["fast_hit"].astype(bool).mean()) < 1.0


@pytest.mark.parametrize("scenario", ("B", "C"))
@pytest.mark.parametrize("policy", ("lru", "context_rules"))
def test_elapsed_decomposition_sums_exactly(scenario: str, policy: str) -> None:
    config = load_config()
    # Enough chunks to cross checkpoint_every_steps so all three paper buckets
    # (data wait, validation, checkpoint) are actually exercised.
    result = run_simulation(config, policy, scenario, 0, epochs=1, chunk_limit=128)
    metrics = result.metrics
    components = (
        "promotion_ms",
        "train_read_ms",
        "validation_read_ms",
        "other_read_ms",
        "checkpoint_write_ms",
        "train_compute_ms",
        "flush_ms",
    )
    assert all(name in METRIC_NAMES for name in components)
    total = sum(float(metrics[name]) for name in components)
    assert total == pytest.approx(float(metrics["elapsed_ms"]), abs=1e-9, rel=0.0)
    assert float(metrics["validation_read_ms"]) > 0.0
    assert float(metrics["checkpoint_write_ms"]) > 0.0
