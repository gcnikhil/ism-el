"""Phase 4 tests: oracle-context upper bound and its ablation."""

from __future__ import annotations

from config import load_config
from policies import (
    AccessEvent,
    ContextRulesPolicy,
    Pin,
    POLICY_CLASSES,
    Promote,
)
from simulator import StoreView, run_simulation
from tiers import FAST, MID, SLOW


class FakeView:
    """Minimal PolicyView for action-level policy tests."""

    def __init__(self, fast_ids: set[int], mid_ids: set[int]) -> None:
        self.fast_ids = fast_ids
        self.mid_ids = mid_ids
        self.history: dict[str, object] = {}

    def occupancy(self) -> dict[int, int]:
        return {FAST: 0, MID: 0, SLOW: 0}

    def free_space(self) -> dict[int, int | None]:
        return {FAST: 1000, MID: 1000, SLOW: None}

    def holds(self, file_id: int) -> tuple[int, ...]:
        tiers = []
        if file_id in self.fast_ids:
            tiers.append(FAST)
        if file_id in self.mid_ids:
            tiers.append(MID)
        return tuple(tiers) or (SLOW,)

    def size_of(self, file_id: int) -> int:
        return 100


def event(file_id: int) -> AccessEvent:
    return AccessEvent(
        file_id=file_id, size_bytes=100, op="read", epoch=0, step=0, phase="train"
    )


def test_context_rules_use_declared_types() -> None:
    policy = ContextRulesPolicy({1: "metadata", 2: "validation", 3: "chunk"})
    view = FakeView(set(), set())

    assert policy.on_access(event(1), view) == [
        Promote(file_id=1, tier=FAST),
        Pin(file_id=1),
    ]
    assert policy.on_access(event(2), view) == [
        Promote(file_id=2, tier=MID),
        Pin(file_id=2),
    ]
    assert policy.on_access(event(3), view) == [Promote(file_id=3, tier=FAST)]


def test_context_rules_promote_only_once() -> None:
    policy = ContextRulesPolicy({1: "metadata", 2: "validation"})
    view = FakeView({1}, {2})
    assert policy.on_access(event(1), view) == []
    assert policy.on_access(event(2), view) == []


def test_no_context_ablation_ignores_declared_types() -> None:
    policy = ContextRulesPolicy({2: "validation"}, no_context=True)
    view = FakeView(set(), set())
    assert policy.on_access(event(2), view) == [Promote(file_id=2, tier=FAST)]


def test_context_channel_is_constructor_only() -> None:
    assert not hasattr(AccessEvent(1, 100, "read", 0, 0, "train"), "declared_type")
    assert not hasattr(StoreView, "declared_types")
    assert "declared_type" not in set(
        field for field in AccessEvent.__dataclass_fields__
    )


def test_context_ablation_improves_data_wait_and_migration() -> None:
    config = load_config()
    with_ctx = run_simulation(config, "context_rules", "B", 0).metrics
    no_ctx = run_simulation(config, "context_rules_noctx", "B", 0).metrics
    assert with_ctx["data_wait_ms"] <= no_ctx["data_wait_ms"] + 1.0
    assert with_ctx["migration_bytes"] <= no_ctx["migration_bytes"]
    assert with_ctx["throughput"] >= no_ctx["throughput"]


def test_context_policies_are_registered() -> None:
    assert "context_rules" in POLICY_CLASSES
    assert "context_rules_noctx" in POLICY_CLASSES
