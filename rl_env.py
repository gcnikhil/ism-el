"""Phase 7: a small CPU-only RL environment over the tiered store.

``TieringEnv`` exposes one decision per storage access and reuses the exact
same :class:`~store.TieredStore`, :class:`~simulator.StoreView`, action
application and metric definitions as the virtual-time simulator. It is a
Gymnasium-style environment (``reset`` / ``step`` / spaces) but the RL agent is
plain NumPy, so Gymnasium is optional at training time.

Information barrier: the observation is the observable feature vector from
``features.py``. The environment never puts ``true_type``, names, the oracle or
declared types into the observation; those remain harness-only.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:  # Gymnasium is used for spaces/API shape but is not required by the agent.
    import gymnasium as gym

    _EnvBase = gym.Env
except ImportError:  # pragma: no cover - exercised only without the dependency
    gym = None  # type: ignore[assignment]
    _EnvBase = object  # type: ignore[assignment,misc]

from config import load_config, resolve_path
from features import FEATURE_NAMES, FeatureExtractor, feature_vector
from logger import Logger
from policies import RL_ACTION_NAMES, rl_action_to_actions, TickContext
from simulator import StoreView, _apply_actions, _tiers_for_run
from store import TieredStore, VirtualClock
from tiers import FAST, MID, SLOW
from workload import generate_workload, load_manifest


def _make_spaces() -> tuple[Any, Any]:
    """Return observation/action spaces (or ``None`` without Gymnasium)."""
    if gym is None:
        return None, None
    observation = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(len(FEATURE_NAMES),),
        dtype=np.float32,
    )
    action = gym.spaces.Discrete(len(RL_ACTION_NAMES))
    return observation, action


class TieringEnv(_EnvBase):
    """One access-level Gymnasium-style environment for RL training.

    The reward per access is exactly::

        service = 1.0 (fast) | 0.6 (mid) | 0.0 (origin)
        reward  = service - wait_ms / slow_latency_ms
                  - migration_weight * (migration_bytes_this_access / 1e6)
                  - (group_data_wait_ms / wait_scale)

    The tier-based service term is deliberately shaped: raw latency makes a
    mid hit (~0.5 ms) nearly indistinguishable from a fast hit (~0.06 ms) at
    this scale, so the agent learned to avoid the origin but not to prefer the
    fastest tier. The wait and data-wait terms keep the reward tied to the
    exact metrics. The data-wait term is applied once per workload step (on its
    final access). All scales live in the ``rl:`` block of ``config.yaml``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict,
        scenario: str,
        seed: int,
        *,
        epochs: int | None = None,
        chunk_limit: int | None = None,
        migration_weight: float | None = None,
        wait_scale: float | None = None,
    ) -> None:
        """Bind the environment to one workload configuration."""
        self.config = config
        self.scenario = str(scenario)
        self.seed = int(seed)
        self.epochs = epochs
        self.chunk_limit = chunk_limit
        rl_cfg = config.get("rl", {})
        self.migration_weight = float(
            migration_weight
            if migration_weight is not None
            else rl_cfg.get("migration_weight", 0.05)
        )
        self.wait_scale = float(
            wait_scale if wait_scale is not None else rl_cfg.get("wait_scale", 10.0)
        )
        self.observation_space, self.action_space = _make_spaces()

        self._store: Optional[TieredStore] = None
        self._clock: Optional[VirtualClock] = None
        self._tiers: dict[int, Any] = {}
        self._truth: dict[int, str] = {}
        self._logger: Optional[Logger] = None
        self._view: Optional[StoreView] = None
        self._extractor: Optional[FeatureExtractor] = None
        self._accesses: list[Any] = []
        self._step_sizes: list[int] = []
        self._step_phases: list[str] = []
        self._index = 0
        self._group_index = 0
        self._group_pos = 0
        self._group_read_time = 0.0
        self._terminated = False
        self._slow_latency_ms = 1.0

        # Episode accumulators, mirrored from the simulator's metric loop.
        self._total_data_wait_ms = 0.0
        self._reads = 0
        self._fast_hits = 0
        self._steps_taken = 0
        self._tier_served: dict[int, int] = {FAST: 0, MID: 0, SLOW: 0}
        self.final_info: dict[str, float] = {}

        if gym is not None:  # keep Gymnasium's RNG bookkeeping consistent
            super().reset(seed=self.seed)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Stop the store's background worker (idempotent)."""
        if self._store is not None:
            self._store.close()
            self._store = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a fresh episode and return the first observation."""
        del options
        if seed is not None:
            self.seed = int(seed)
            if gym is not None:
                super().reset(seed=self.seed)
        self.close()

        manifest = load_manifest(self.config)
        self._truth = {
            int(entry["id"]): str(entry["true_type"]) for entry in manifest["files"]
        }
        workload = generate_workload(
            self.config,
            self.scenario,
            self.seed,
            epochs=self.epochs,
            chunk_limit=self.chunk_limit,
        )
        self._tiers = _tiers_for_run(self.config, self.scenario, "lru", manifest)
        self._clock = VirtualClock()
        self._store = TieredStore(
            self._tiers,
            self._clock,
            max_queue=int(self.config["max_prefetch_queue"]),
        )
        self._logger = Logger(
            run_id=f"rl-{self.scenario}-seed{self.seed}",
            seed=self.seed,
            scenario=self.scenario,
            policy="rl_agent",
        )
        for entry in manifest["files"]:
            self._store.register(int(entry["id"]), int(entry["size_bytes"]))

        self._view = StoreView(self._store, {})
        self._extractor = FeatureExtractor()
        self._accesses = [event for step in workload.steps for event in step]
        self._step_sizes = [len(step) for step in workload.steps if step]
        self._step_phases = [
            step[0].phase for step in workload.steps if step
        ]
        self._index = 0
        self._group_index = 0
        self._group_pos = 0
        self._group_read_time = 0.0
        self._terminated = False
        self._total_data_wait_ms = 0.0
        self._reads = 0
        self._fast_hits = 0
        self._steps_taken = 0
        self._tier_served = {FAST: 0, MID: 0, SLOW: 0}
        self.final_info = {}
        self._slow_latency_ms = max(
            1e-6, float(self._tiers[SLOW].spec.latency_ms)
        )

        if not self._accesses:  # defensive: an empty workload cannot be stepped
            self._terminated = True
            return self._blank_observation(), {}

        return self._observation(), {}

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply ``action`` for the current access and advance one access."""
        if self._terminated or self._store is None:
            raise RuntimeError("Call reset() before step()")
        index = int(action)
        if index < 0 or index >= len(RL_ACTION_NAMES):
            raise ValueError(f"Action index out of range: {index}")

        event = self._accesses[self._index]
        migration_before = int(self._store.migration_bytes)
        action_start_ms = self._clock.now_ms()  # type: ignore[union-attr]
        latency_ms = 0.0
        wait_ms = 0.0
        tier_served = SLOW
        fast_hit = False
        read_time_ms = 0.0

        if event.op == "read":
            # Snapshot residency on arrival, before the RL action can promote.
            arrival_fast = self._store.fast_resident(event.file_id)  # type: ignore[union-attr]
            actions = rl_action_to_actions(index, event.file_id, self._view)
            _apply_actions(actions, self._store, self._logger, self._clock)
            action_ms = self._clock.now_ms() - action_start_ms  # type: ignore[union-attr]
            result = self._store.get(event.file_id, event, arrival_fast=arrival_fast)
            latency_ms = result.latency_ms
            wait_ms = result.wait_ms
            tier_served = result.tier_served
            fast_hit = result.fast_hit
            read_time_ms = action_ms + latency_ms + wait_ms
            self._reads += 1
            self._fast_hits += int(fast_hit)
            self._tier_served[tier_served] = self._tier_served.get(tier_served, 0) + 1
            self._logger.log_access(
                ts=self._clock.now_ms(),  # type: ignore[union-attr]
                file_id=event.file_id,
                size_bytes=event.size_bytes,
                op="read",
                epoch=event.epoch,
                step=event.step,
                phase=event.phase,
                tier_served=tier_served,
                latency_ms=latency_ms,
                wait_ms=wait_ms,
                fast_hit=fast_hit,
                true_type=self._truth[event.file_id],
            )
        else:
            # Writes always land in the origin, as in every learned policy.
            landing = SLOW
            self._store.put(event, landing, dirty=False)
            spec = self._tiers[landing].spec
            write_ms = event.size_bytes / (spec.bandwidth_mbps * 1024.0) + spec.latency_ms
            self._clock.advance(write_ms)  # type: ignore[union-attr]
            latency_ms = write_ms
            tier_served = landing
            read_time_ms = write_ms
            self._logger.log_access(
                ts=self._clock.now_ms(),  # type: ignore[union-attr]
                file_id=event.file_id,
                size_bytes=event.size_bytes,
                op="write",
                epoch=event.epoch,
                step=event.step,
                phase=event.phase,
                tier_served=tier_served,
                latency_ms=latency_ms,
                wait_ms=0.0,
                fast_hit=False,
                true_type=self._truth[event.file_id],
            )

        migration_bytes = int(self._store.migration_bytes) - migration_before
        service = {FAST: 1.0, MID: 0.6, SLOW: 0.0}.get(tier_served, 0.0)
        reward = (
            service
            - wait_ms / self._slow_latency_ms
            - self.migration_weight * (migration_bytes / 1_000_000.0)
        )

        # Mirror the simulator's per-workload-step overlap and compute advance.
        self._group_read_time += read_time_ms
        self._group_pos += 1
        if self._group_pos >= self._step_sizes[self._group_index]:
            phase = self._step_phases[self._group_index]
            ctx = TickContext(
                now_ms=self._clock.now_ms(),  # type: ignore[union-attr]
                epoch=event.epoch,
                step=event.step,
                phase=phase,
            )
            self._store.tick(ctx)
            if phase in {"train", "warmup"}:
                data_wait = max(
                    0.0,
                    self._group_read_time - float(self.config["compute_ms"]),
                )
                self._total_data_wait_ms += data_wait
                reward -= data_wait / self.wait_scale
            self._group_index += 1
            self._group_pos = 0
            self._group_read_time = 0.0

        self._index += 1
        self._steps_taken += 1
        terminated = self._index >= len(self._accesses)
        self._terminated = terminated

        info: dict[str, Any] = {
            "steps": self._steps_taken,
            "latency_ms": latency_ms,
            "wait_ms": wait_ms,
            "tier_served": tier_served,
            "fast_hit": fast_hit,
            "migration_bytes": migration_bytes,
            "fast_hit_rate": self._fast_hits / self._reads if self._reads else 0.0,
        }
        if terminated:
            # Flush dirty cached writes exactly like the simulator epilogue.
            for tier_id in (FAST, MID):
                for file_id in sorted(self._tiers[tier_id].dirty):
                    self._store.flush(file_id)
            elapsed_ms = self._clock.now_ms()  # type: ignore[union-attr]
            latency_scale = max(1, self._reads)
            info["data_wait_ms"] = self._total_data_wait_ms
            info["read_count"] = self._reads
            info["migration_bytes_total"] = float(self._store.migration_bytes)
            info["wasted_prefetch_bytes"] = float(self._store.wasted_prefetch_bytes)
            info["elapsed_ms"] = elapsed_ms
            self.final_info = dict(info)
            next_observation = self._blank_observation()
        else:
            next_observation = self._observation()

        return next_observation, float(reward), terminated, False, info

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _observation(self) -> np.ndarray:
        event = self._accesses[self._index]
        features = self._extractor.extract(event, self._view)
        return np.asarray(feature_vector(features), dtype=np.float32)

    def _blank_observation(self) -> np.ndarray:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"TieringEnv(scenario={self.scenario!r}, seed={self.seed}, "
            f"index={self._index}/{len(self._accesses)})"
        )


def make_env(config: dict, scenario: str, seed: int, **kwargs: Any) -> TieringEnv:
    """Convenience factory for one training episode."""
    return TieringEnv(config, scenario, seed, **kwargs)


def load_default_config() -> dict:
    """Load ``config.yaml`` from the project root (convenience helper)."""
    return load_config()


__all__ = [
    "TieringEnv",
    "make_env",
    "load_default_config",
    "RL_ACTION_NAMES",
    "resolve_path",
]
