"""One virtual-time simulator shared by every policy and scenario.

The simulator drives :class:`store.TieredStore` with a pre-generated workload,
applies policy actions, logs the exact trace schema and computes the metrics
defined in ``docs/DESIGN.md``. Real mode (Phase 3) will reuse the same loop with
a different clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np

from logger import Logger
from policies import (
    Action,
    BeladyPolicy,
    ClairvoyantPolicy,
    ClassifierPolicy,
    ContextRulesPolicy,
    EmbeddingPrefetchPolicy,
    Evict,
    FullContextTierPolicy,
    PhaseContext,
    Pin,
    POLICY_CLASSES,
    Prefetch,
    Promote,
    TickContext,
    Unpin,
)
from store import MigrationResult, ReadResult, TieredStore, VirtualClock
from tiers import FAST, SLOW, build_tiers
from workload import StreamOracle, Workload, generate_workload, load_manifest


@dataclass
class SimResult:
    """Everything a caller needs after one run."""

    metrics: dict[str, float]
    logger: Logger
    store: TieredStore
    clock: VirtualClock
    workload: Workload


class StoreView:
    """The restricted information surface handed to policies."""

    def __init__(self, store: TieredStore, history: dict[str, object]) -> None:
        """Bind a view to a store and one policy's private history."""
        self._store = store
        self._history = history

    def occupancy(self) -> dict[int, int]:
        """Return tier id to bytes used."""
        return {
            tier_id: tier.used_bytes() for tier_id, tier in self._store.tiers.items()
        }

    def free_space(self) -> dict[int, Optional[int]]:
        """Return tier id to free bytes."""
        return {
            tier_id: tier.free_bytes() for tier_id, tier in self._store.tiers.items()
        }

    def holds(self, file_id: int) -> tuple[int, ...]:
        """Return the sorted tiers holding a copy of ``file_id``."""
        return self._store.locate(file_id)

    def fast_resident_files(self) -> dict[int, int]:
        """Return ``{file_id: size_bytes}`` for files currently in FAST.

        A single snapshot is equivalent to calling ``holds`` and ``size_of``
        for each candidate (no actions are applied during a scan) but avoids
        O(n) tier lookups per candidate in eviction-heavy policies. It exposes
        only membership and size, already available through ``holds``/``size_of``.
        """
        with self._store._lock:
            return dict(self._store.tiers[FAST].files)

    def size_of(self, file_id: int) -> int:
        """Return the opaque size of ``file_id``."""
        for tier_id in self._store.locate(file_id):
            return self._store.tiers[tier_id].files[file_id]
        raise KeyError(f"Unknown file_id {file_id}")

    @property
    def history(self) -> dict[str, object]:
        """Return the bound policy's private history."""
        return self._history


def _tiers_for_run(config: dict, scenario: str, policy_name: str, manifest: dict) -> dict:
    """Build tiers, enlarging Tier 0 for AllFast and Scenario D only."""
    tier_config: list[dict] = [dict(entry) for entry in config["tiers"]]
    if policy_name == "allfast" or scenario == "D":
        total_bytes = sum(int(entry["size_bytes"]) for entry in manifest["files"])
        for entry in tier_config:
            if int(entry["id"]) == FAST:
                entry["capacity_mb"] = (total_bytes * 2) / (1024 * 1024)
    return build_tiers(tier_config)


def _load_full_context_models(
    config: dict,
    *,
    agent: Optional[object] = None,
    classifier: Optional[object] = None,
    predictor: Optional[object] = None,
) -> tuple[object, Optional[object], Optional[object]]:
    """Resolve the three learned components for ``FullContextTierPolicy``.

    The RL agent is required (it defines the learned decisions). The Phase 5/6
    artefacts are optional: when absent the policy simply falls back to RL plus
    its built-in rules, and the omission is printed rather than hidden.
    """
    if agent is None:
        from rl_agent import load_agent

        agent = load_agent(config)
    rl_cfg = config.get("rl", {})
    if classifier is None:
        if not rl_cfg.get("use_classifier", True):
            print("[full_context] classifier disabled by config")
        else:
            try:
                from classifier import load_model

                classifier = load_model(config)
            except FileNotFoundError:
                print("[full_context] classifier artefact missing; skipping type prediction")
    if predictor is None:
        if not rl_cfg.get("use_reuse_predictor", True):
            print("[full_context] reuse predictor disabled by config")
        else:
            try:
                from embeddings import load_predictor

                predictor = load_predictor(config)
            except FileNotFoundError:
                print("[full_context] embedding artefact missing; skipping reuse scoring")
    return agent, classifier, predictor


def _make_policy(
    config: dict,
    policy_name: str,
    scenario: str,
    oracle: Optional[StreamOracle],
    declared_types: Optional[Mapping[int, str]] = None,
    classifier_model: Optional[object] = None,
    rl_agent: Optional[object] = None,
    reuse_predictor: Optional[object] = None,
) -> Any:
    """Instantiate a policy, injecting privileged channels only where allowed."""
    policy_class = POLICY_CLASSES[policy_name]
    if policy_name == "clairvoyant":
        # Scenario C's order depends on training state, not on the shuffle seed.
        return ClairvoyantPolicy(oracle if scenario not in {"C", "E", "F"} else None)
    if policy_name == "belady":
        if oracle is None:
            raise ValueError("BeladyPolicy requires an oracle")
        return BeladyPolicy(oracle)
    if policy_name.startswith("context_rules"):
        return ContextRulesPolicy(
            declared_types or {}, no_context=policy_name.endswith("_noctx")
        )
    if policy_name == "classifier":
        return ClassifierPolicy(classifier_model)
    if policy_name == "embedding":
        return EmbeddingPrefetchPolicy(classifier_model)
    if policy_name == "full_context":
        agent, classifier, predictor = _load_full_context_models(
            config,
            agent=rl_agent,
            classifier=classifier_model,
            predictor=reuse_predictor,
        )
        return FullContextTierPolicy(agent, classifier, predictor)
    if policy_name == "temperature":
        settings = config.get("temperature", {})
        return policy_class(
            half_life=float(settings.get("ewma_half_life", 32)),
            hysteresis_ticks=int(settings.get("hysteresis_ticks", 3)),
        )
    if policy_name == "temperature_learned":
        if reuse_predictor is None:
            try:
                from embeddings import load_predictor

                reuse_predictor = load_predictor(config)
            except FileNotFoundError:
                reuse_predictor = None
        settings = config.get("temperature", {})
        return policy_class(
            reuse_predictor,
            half_life=float(settings.get("ewma_half_life", 32)),
            hysteresis_ticks=int(settings.get("hysteresis_ticks", 3)),
        )
    return policy_class()


def _apply_actions(
    actions: list[Action],
    store: TieredStore,
    logger: Logger,
    clock: VirtualClock,
) -> None:
    """Translate policy actions into store operations and log migrations."""
    for action in actions:
        result: Optional[MigrationResult] = None
        if isinstance(action, Promote):
            result = store.promote(action.file_id, action.tier)
        elif isinstance(action, Prefetch):
            result = store.prefetch(action.file_id, action.tier)
        elif isinstance(action, Evict):
            store.evict(action.file_id, action.tier)
        elif isinstance(action, Pin):
            holders = [
                tier_id
                for tier_id in (FAST, 1)
                if action.file_id in store.tiers[tier_id].files
            ]
            for tier_id in holders or [FAST]:
                store.tiers[tier_id].pinned.add(action.file_id)
        elif isinstance(action, Unpin):
            for tier_id in (FAST, 1):
                store.tiers[tier_id].pinned.discard(action.file_id)
        else:  # pragma: no cover - defensive
            raise TypeError(f"Unknown action type: {type(action)!r}")

        if result is not None:
            logger.log_migration(
                ts=clock.now_ms(),
                file_id=result.file_id,
                size_bytes=result.size_bytes,
                src_tier=result.src_tier,
                dst_tier=result.dst_tier,
                kind=result.kind,
            )


def run_simulation(
    config: dict,
    policy_name: str,
    scenario: str,
    seed: int,
    *,
    epochs: int | None = None,
    chunk_limit: int | None = None,
    on_access: Optional[
        Callable[[object, StoreView, Optional[ReadResult]], None]
    ] = None,
    classifier_model: Optional[object] = None,
    rl_agent: Optional[object] = None,
    reuse_predictor: Optional[object] = None,
) -> SimResult:
    """Run one complete virtual-time experiment.

    Args:
        config: Parsed ``config.yaml`` dictionary.
        policy_name: Lower-case key from :data:`policies.POLICY_CLASSES`.
        scenario: One of ``A``, ``B``, ``C``, ``D``.
        seed: RNG seed for the workload.
        epochs: Optional override of ``config["epochs"]`` for tests.
        chunk_limit: Optional cap on training chunks for real-mode parity.
        on_access: Optional harness callback after each access, used to collect
            training features without exposing privileged data to policies.
        classifier_model: Optional injected model for ``ClassifierPolicy``;
            when omitted it is loaded from the configured model path.
        rl_agent: Optional trained agent for ``FullContextTierPolicy``.
        reuse_predictor: Optional Phase 6 predictor for ``FullContextTierPolicy``.

    Returns:
        A :class:`SimResult` with metrics, logger, store, clock and workload.
    """
    policy_name = policy_name.lower()
    if policy_name not in POLICY_CLASSES:
        raise ValueError(f"Unknown policy: {policy_name}")

    manifest = load_manifest(config)
    truth = {int(entry["id"]): entry["true_type"] for entry in manifest["files"]}
    workload = generate_workload(
        config, scenario, seed, epochs=epochs, chunk_limit=chunk_limit
    )

    tiers = _tiers_for_run(config, scenario, policy_name, manifest)
    clock = VirtualClock()
    store = TieredStore(tiers, clock, max_queue=int(config["max_prefetch_queue"]))
    logger = Logger(
        run_id=f"{scenario}-{policy_name}-seed{seed}",
        seed=seed,
        scenario=scenario,
        policy=policy_name,
        include_temperature=scenario in {"E", "F"},
    )

    for entry in manifest["files"]:
        store.register(int(entry["id"]), int(entry["size_bytes"]))

    if policy_name == "allfast" or scenario == "D":
        # AllFast is an idealized upper bound; Scenario D makes the same state
        # physically possible by giving Tier 0 room for every file up front.
        for entry in manifest["files"]:
            store.tiers[FAST].add(int(entry["id"]), int(entry["size_bytes"]))

    oracle: Optional[StreamOracle] = None
    if policy_name in {"clairvoyant", "belady"}:
        oracle = StreamOracle(workload.read_stream, workload.chunk_ids)
    declared_types: Optional[dict[int, str]] = None
    if policy_name.startswith("context_rules"):
        declared_types = {
            int(entry["id"]): str(entry["true_type"]) for entry in manifest["files"]
        }
    if policy_name == "classifier" and classifier_model is None:
        from classifier import load_model

        classifier_model = load_model(config)
    if policy_name == "embedding" and classifier_model is None:
        from embeddings import load_predictor

        classifier_model = load_predictor(config)
    policy = _make_policy(
        config,
        policy_name,
        scenario,
        oracle,
        declared_types,
        classifier_model,
        rl_agent,
        reuse_predictor,
    )
    view = StoreView(store, getattr(policy, "history", {}))

    total_data_wait_ms = 0.0
    total_samples = 0
    reads = 0
    fast_hits = 0
    # Exact additive decomposition of elapsed_ms. Together these buckets sum
    # to the final clock value (no rounding, no residual).
    promotion_ms = 0.0
    train_read_ms = 0.0
    validation_read_ms = 0.0
    other_read_ms = 0.0
    checkpoint_write_ms = 0.0
    train_compute_ms = 0.0
    flush_ms = 0.0
    tier_served: dict[int, int] = {FAST: 0, 1: 0, SLOW: 0}
    latencies: list[float] = []
    current_phase = workload.steps[0][0].phase if workload.steps else "warmup"
    compute_ms = float(config["compute_ms"])

    try:
        for step_index, accesses in enumerate(workload.steps):
            epoch = accesses[0].epoch
            step = accesses[0].step
            read_time_ms = 0.0

            for event in accesses:
                if event.phase != current_phase:
                    phase_start_ms = clock.now_ms()
                    policy.on_phase_change(
                        PhaseContext(
                            now_ms=clock.now_ms(),
                            epoch=epoch,
                            previous_phase=current_phase,
                            new_phase=event.phase,
                        ),
                        view,
                    )
                    promotion_ms += clock.now_ms() - phase_start_ms
                    current_phase = event.phase

                if event.op == "read":
                    # Capture Tier-0 residency when the read ARRIVES, before
                    # the policy can promote this same file on demand.
                    arrival_fast = store.fast_resident(event.file_id)
                    action_start_ms = clock.now_ms()
                    actions = policy.on_access(event, view)
                    _apply_actions(actions, store, logger, clock)
                    # Synchronous promotions block the training loop before the
                    # read itself; count that time as data-path time.
                    action_ms = clock.now_ms() - action_start_ms
                    promotion_ms += action_ms
                    read_time_ms += action_ms
                    result = store.get(event.file_id, event, arrival_fast=arrival_fast)
                    read_ms = result.latency_ms + result.wait_ms
                    read_time_ms += read_ms
                    if event.phase in {"train", "warmup"}:
                        train_read_ms += read_ms
                    elif event.phase == "validation":
                        validation_read_ms += read_ms
                    else:
                        other_read_ms += read_ms
                    reads += 1
                    fast_hits += int(result.fast_hit)
                    tier_served[result.tier_served] = tier_served.get(result.tier_served, 0) + 1
                    latencies.append(result.latency_ms)
                    access_kwargs = dict(
                        ts=clock.now_ms(),
                        file_id=event.file_id,
                        size_bytes=event.size_bytes,
                        op="read",
                        epoch=event.epoch,
                        step=event.step,
                        phase=event.phase,
                        tier_served=result.tier_served,
                        latency_ms=result.latency_ms,
                        wait_ms=result.wait_ms,
                        fast_hit=result.fast_hit,
                        true_type=truth[event.file_id],
                    )
                    if logger.include_temperature:
                        access_kwargs["true_temp"] = workload.temperature_for(event)
                    logger.log_access(**access_kwargs)
                    if oracle is not None:
                        oracle.observe(event.file_id)
                    if on_access is not None:
                        on_access(event, view, result)
                else:
                    landing_tier = int(policy.on_write(event, view))
                    store.put(event, landing_tier, dirty=(landing_tier != SLOW))
                    write_ms = (
                        event.size_bytes / (tiers[landing_tier].spec.bandwidth_mbps * 1024.0)
                        + tiers[landing_tier].spec.latency_ms
                    )
                    clock.advance(write_ms)
                    checkpoint_write_ms += write_ms
                    access_kwargs = dict(
                        ts=clock.now_ms(),
                        file_id=event.file_id,
                        size_bytes=event.size_bytes,
                        op="write",
                        epoch=event.epoch,
                        step=event.step,
                        phase=event.phase,
                        tier_served=landing_tier,
                        latency_ms=write_ms,
                        wait_ms=0.0,
                        fast_hit=False,
                        true_type=truth[event.file_id],
                    )
                    if logger.include_temperature:
                        access_kwargs["true_temp"] = workload.temperature_for(event)
                    logger.log_access(**access_kwargs)
                    if on_access is not None:
                        on_access(event, view, None)

            store.tick(
                TickContext(
                    now_ms=clock.now_ms(),
                    epoch=epoch,
                    step=step,
                    phase=current_phase,
                )
            )
            tick_start_ms = clock.now_ms()
            tick_actions = policy.on_tick(
                TickContext(
                    now_ms=clock.now_ms(),
                    epoch=epoch,
                    step=step,
                    phase=current_phase,
                ),
                view,
            )
            _apply_actions(tick_actions, store, logger, clock)
            promotion_ms += clock.now_ms() - tick_start_ms

            if current_phase in {"train", "warmup"}:
                # Compute runs after the tick so in-flight prefetches overlap
                # with it (the GPU-idle proxy assumption from the contract).
                clock.advance(compute_ms)
                train_compute_ms += compute_ms
                total_samples += int(config["batch_size"])
                total_data_wait_ms += max(0.0, read_time_ms - compute_ms)
            del step_index

        # All dirty cached writes must be flushed before the run ends.
        flush_start_ms = clock.now_ms()
        for tier_id in (FAST, 1):
            dirty_ids = sorted(store.tiers[tier_id].dirty)
            for file_id in dirty_ids:
                store.flush(file_id)
        flush_ms = clock.now_ms() - flush_start_ms
    finally:
        store.close()

    elapsed_ms = clock.now_ms()
    latency_array = np.asarray(latencies, dtype=np.float64)
    metrics: dict[str, float] = {
        "data_wait_ms": total_data_wait_ms,
        "data_wait_pct": 100.0 * total_data_wait_ms / elapsed_ms if elapsed_ms else 0.0,
        "throughput": total_samples / (elapsed_ms / 1000.0) if elapsed_ms else 0.0,
        "fast_hit_rate": fast_hits / reads if reads else 0.0,
        "mid_hit_rate": tier_served.get(1, 0) / reads if reads else 0.0,
        "slow_hit_rate": tier_served.get(SLOW, 0) / reads if reads else 0.0,
        "migration_bytes": float(store.migration_bytes),
        "wasted_prefetch_bytes": float(store.wasted_prefetch_bytes),
        "latency_mean_ms": float(latency_array.mean()) if latency_array.size else 0.0,
        "latency_p95_ms": float(np.percentile(latency_array, 95)) if latency_array.size else 0.0,
        "latency_p99_ms": float(np.percentile(latency_array, 99)) if latency_array.size else 0.0,
        "elapsed_ms": elapsed_ms,
        "read_count": float(reads),
        "sample_count": float(total_samples),
        # Exact elapsed-time decomposition (sums to elapsed_ms).
        "promotion_ms": promotion_ms,
        "train_read_ms": train_read_ms,
        "validation_read_ms": validation_read_ms,
        "other_read_ms": other_read_ms,
        "checkpoint_write_ms": checkpoint_write_ms,
        "train_compute_ms": train_compute_ms,
        "flush_ms": flush_ms,
    }
    for name, value in metrics.items():
        logger.set_metric(name, value)
    return SimResult(
        metrics=metrics,
        logger=logger,
        store=store,
        clock=clock,
        workload=workload,
    )
