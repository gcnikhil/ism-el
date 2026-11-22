"""LRU and ARC correctness tests (Phase 2)."""

from __future__ import annotations

from policies import ARCPolicy, AccessEvent, Evict, LRUPolicy, Promote
from store import TieredStore, VirtualClock
from tiers import FAST, MID, SLOW, Tier, TierSpec


class FakeView:
    """Minimal PolicyView for unit-testing policies without a full run."""

    def __init__(self, sizes: dict[int, int], fast_ids: set[int], capacity: int) -> None:
        """Build a fake view with one finite fast tier and an unlimited origin."""
        self.sizes = sizes
        self.fast_ids = fast_ids
        self.capacity = capacity
        self.history: dict[str, object] = {}

    def occupancy(self) -> dict[int, int]:
        """Return fast-tier occupancy."""
        return {FAST: sum(self.sizes[i] for i in self.fast_ids), 1: 0, SLOW: 0}

    def free_space(self) -> dict[int, int | None]:
        """Return free space per tier."""
        return {FAST: self.capacity - self.occupancy()[FAST], 1: 10**9, SLOW: None}

    def holds(self, file_id: int) -> tuple[int, ...]:
        """Return tier membership."""
        return (FAST,) if file_id in self.fast_ids else (SLOW,)

    def size_of(self, file_id: int) -> int:
        """Return the opaque size."""
        return self.sizes[file_id]


def event(file_id: int) -> AccessEvent:
    """Build one read event."""
    return AccessEvent(
        file_id=file_id, size_bytes=1000, op="read", epoch=0, step=0, phase="train"
    )


def test_lru_promotes_only_on_miss() -> None:
    policy = LRUPolicy()
    view = FakeView({1: 1000}, set(), 5000)
    assert policy.on_access(event(1), view) == [Promote(file_id=1, tier=FAST)]
    view.fast_ids.add(1)
    assert policy.on_access(event(1), view) == []


def test_store_evicts_least_recently_used_victim() -> None:
    tiers = {
        FAST: Tier(TierSpec(FAST, "fast", 2500, 0.01, 3000)),
        MID: Tier(TierSpec(MID, "mid", 10**7, 0.2, 500)),
        SLOW: Tier(TierSpec(SLOW, "slow", None, 8.0, 80)),
    }
    with TieredStore(tiers, VirtualClock()) as store:
        for file_id in (0, 1):
            store.register(file_id, 1000)
            store.promote(file_id, FAST)
        store.get(0)  # refresh file 0's recency
        store.register(2, 1000)
        store.promote(2, FAST)
        assert store.tiers[FAST].holds(0)
        assert not store.tiers[FAST].holds(1)
        assert store.tiers[FAST].holds(2)


def test_arc_moves_t1_entry_to_t2_on_hit() -> None:
    policy = ARCPolicy()
    sizes = {1: 1000, 2: 1000, 3: 1000}
    view = FakeView(sizes, {1, 2}, 2500)

    actions = policy.on_access(event(3), view)
    assert any(isinstance(action, Promote) for action in actions)
    assert 3 in policy.t1
    view.fast_ids.add(3)  # simulate the store applying the promotion
    actions = policy.on_access(event(3), view)
    assert actions == []
    assert 3 not in policy.t1 and 3 in policy.t2


def test_arc_ghost_hit_adjusts_p_and_promotes_to_t2() -> None:
    policy = ARCPolicy()
    policy.b1[5] = 1000
    view = FakeView({5: 1000}, set(), 5000)
    before = policy.p
    actions = policy.on_access(event(5), view)
    assert policy.p == before + 1000
    assert 5 in policy.t2 and 5 not in policy.b1
    assert any(isinstance(action, Promote) for action in actions)
    assert not any(isinstance(action, Evict) for action in actions)
