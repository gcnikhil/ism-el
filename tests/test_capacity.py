"""Capacity-limit and eviction-safety tests."""

from __future__ import annotations

import pytest

from policies import AccessEvent
from store import TieredStore, VirtualClock
from tiers import FAST, MID, SLOW, CapacityError, Tier, TierSpec


def make_store(fast_mb: float = 0.0034) -> TieredStore:
    """Build a store with a tiny fast tier and generous other tiers."""
    tiers = {
        FAST: Tier(TierSpec(FAST, "fast", int(fast_mb * 1024 * 1024), 0.01, 3000)),
        MID: Tier(TierSpec(MID, "mid", 60 * 1024 * 1024, 0.2, 500)),
        SLOW: Tier(TierSpec(SLOW, "slow", None, 8.0, 80)),
    }
    return TieredStore(tiers, VirtualClock())


def test_origin_is_unlimited_and_registration_is_free() -> None:
    store = make_store()
    for file_id in range(10):
        store.register(file_id, 1_000_000)
    assert store.tiers[SLOW].used_bytes() == 10_000_000
    assert store.tiers[SLOW].free_bytes() is None
    assert store.migration_bytes == 0


def test_fast_capacity_never_exceeded() -> None:
    store = make_store()
    for file_id in range(6):
        store.register(file_id, 1000)
        store.promote(file_id, FAST)
        assert store.tiers[FAST].used_bytes() <= store.tiers[FAST].spec.capacity_bytes
    assert store.tiers[FAST].holds(5)


def test_pinned_and_dirty_files_are_not_evicted() -> None:
    store = make_store()
    for file_id in range(3):
        store.register(file_id, 1000)
        store.promote(file_id, FAST)

    store.register(3, 1000)
    store.tiers[FAST].pinned.add(0)
    store.tiers[FAST].pinned.add(2)
    write_event = AccessEvent(
        file_id=1, size_bytes=1000, op="write", epoch=0, step=0, phase="checkpoint"
    )
    store.put(write_event, FAST, dirty=True)

    with pytest.raises(CapacityError):
        store.promote(3, FAST)
    assert store.tiers[FAST].holds(0)
    assert store.tiers[FAST].holds(1)
    assert store.tiers[FAST].holds(2)
    assert 1 in store.tiers[FAST].dirty
    assert not store.tiers[FAST].holds(3)


def test_origin_eviction_is_refused() -> None:
    store = make_store()
    store.register(0, 1000)
    assert store.evict(0, SLOW) is False
    assert store.tiers[SLOW].holds(0)
