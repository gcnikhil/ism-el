"""Exact migration-byte accounting and bounded background queue tests."""

from __future__ import annotations

from policies import AccessEvent
from store import TieredStore, VirtualClock
from tiers import FAST, MID, SLOW, Tier, TierSpec


def make_store(max_queue: int = 8) -> TieredStore:
    """Build a store with large caches so eviction cannot skew byte counts."""
    tiers = {
        FAST: Tier(TierSpec(FAST, "fast", 100 * 1024 * 1024, 0.01, 3000)),
        MID: Tier(TierSpec(MID, "mid", 100 * 1024 * 1024, 0.2, 500)),
        SLOW: Tier(TierSpec(SLOW, "slow", None, 8.0, 80)),
    }
    return TieredStore(tiers, VirtualClock(), max_queue=max_queue)


def test_promote_flush_and_read_accounting() -> None:
    store = make_store()
    store.register(0, 100_000)
    assert store.migration_bytes == 0

    result = store.promote(0, FAST)
    assert result is not None
    assert result.size_bytes == 100_000
    assert store.migration_bytes == 100_000

    read = store.get(0)
    assert read.tier_served == FAST
    assert read.fast_hit is True
    assert store.migration_bytes == 100_000

    assert store.evict(0, FAST) is True
    assert store.migration_bytes == 100_000

    event = AccessEvent(
        file_id=0, size_bytes=100_000, op="write", epoch=0, step=1, phase="checkpoint"
    )
    store.put(event, MID, dirty=True)
    assert store.migration_bytes == 100_000
    assert store.tiers[MID].holds(0) and 0 in store.tiers[MID].dirty

    flushed = store.flush(0)
    assert flushed is not None
    assert flushed.dst_tier == SLOW
    assert store.migration_bytes == 200_000
    assert 0 not in store.tiers[MID].dirty


def test_prefetch_waits_only_for_remaining_transfer() -> None:
    store = make_store()
    store.register(0, 80_000)
    job = store.prefetch(0, MID)
    assert job is not None
    store.drain()
    assert store.migration_bytes == 80_000

    before = store.clock.now_ms()
    read = store.get(0)
    assert read.tier_served == MID
    assert read.wait_ms > 0
    assert store.clock.now_ms() >= before + read.wait_ms
    assert store.migration_bytes == 80_000


def test_wasted_prefetch_bytes_are_exact() -> None:
    store = make_store()
    store.register(0, 60_000)
    job = store.prefetch(0, MID)
    assert job is not None
    store.drain()
    store.clock.wait_until(job.completed_ms)
    store.tick(ctx=None)

    assert store.tiers[MID].holds(0)
    assert store.wasted_prefetch_bytes == 0
    assert store.evict(0, MID) is True
    assert store.wasted_prefetch_bytes == 60_000

    store.register(1, 40_000)
    good_job = store.prefetch(1, MID)
    assert good_job is not None
    store.drain()
    store.clock.wait_until(good_job.completed_ms)
    store.tick(ctx=None)
    store.get(1)
    assert store.evict(1, MID) is True
    assert store.wasted_prefetch_bytes == 60_000


def test_background_queue_is_bounded() -> None:
    store = make_store(max_queue=2)
    for file_id in range(5):
        store.register(file_id, 1000)

    accepted = [store.prefetch(file_id, FAST) is not None for file_id in range(5)]
    assert accepted == [True, True, False, False, False]
    assert store.migration_bytes == 2000
    store.drain()
    store.close()
