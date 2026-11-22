"""Phase 3 tests: real clock, payload store, rank math and trace schema."""

from __future__ import annotations

import time

import pandas as pd

from config import load_config
from store import PayloadStore, RealClock
from tiers import FAST, MID, SLOW, Tier, TierSpec
from train import run_real_training, spearman_rho

TRACE_COLUMNS = [
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
]


def test_real_clock_advance_and_wait_until() -> None:
    clock = RealClock()
    start = clock.now_ms()
    clock.advance(5.0)
    assert 4.0 <= clock.now_ms() - start < 100.0
    target = clock.now_ms() + 5.0
    clock.wait_until(target)
    assert clock.now_ms() >= target - 0.5


def test_payload_store_caches_and_drops_payloads(tmp_path) -> None:
    path = tmp_path / "chunk.bin"
    payload = b"0123456789"
    path.write_bytes(payload)
    tiers = {
        FAST: Tier(TierSpec(FAST, "fast", 1024, 0.01, 3000)),
        MID: Tier(TierSpec(MID, "mid", 1024, 0.2, 500)),
        SLOW: Tier(TierSpec(SLOW, "slow", None, 8.0, 80)),
    }
    clock = RealClock()
    with PayloadStore(tiers, clock, {7: path}) as store:
        store.register(7, 10)
        store.promote(7, FAST)
        assert store.read_payload(7, FAST) == payload
        assert store.evict(7, FAST) is True
        assert 7 not in store.payloads
        assert store.read_payload(7, SLOW) == payload


def test_spearman_rho_handles_order_and_ties() -> None:
    assert spearman_rho([1, 2, 3], [10, 20, 30]) == 1.0
    assert spearman_rho([1, 2, 3], [30, 20, 10]) == -1.0
    assert spearman_rho([1, 2, 2], [10, 20, 30]) > 0.8


def test_real_training_writes_exact_trace_schema() -> None:
    config = load_config()
    result = run_real_training(
        config, "lru", "A", seed=0, subset_images=100, epochs=1
    )
    frame: pd.DataFrame = result.logger.trace_frame()
    assert list(frame.columns) == TRACE_COLUMNS
    assert not frame.empty
    assert set(frame["true_type"].unique()) <= {
        "chunk", "validation", "metadata", "checkpoint", "hot"
    }
    assert result.metrics["sample_count"] > 0
    assert result.metrics["elapsed_ms"] > 0
