"""Phase 7 tests: RL action space, Gym env, tabular agent and full-context policy."""

from __future__ import annotations

import json

import numpy as np
import pytest

from config import load_config
from features import FEATURE_NAMES
from policies import (
    AccessEvent,
    FullContextTierPolicy,
    POLICY_CLASSES,
    Promote,
    RL_ACTION_NAMES,
    rl_action_to_actions,
)
from rl_agent import (
    TabularQLearningAgent,
    discretize_state,
    load_agent,
    train_and_evaluate,
    train_agent,
)
from rl_env import TieringEnv
from simulator import run_simulation
from tiers import FAST, MID, SLOW


class FakeView:
    """Minimal PolicyView exposing only the observable occupancy API."""

    def __init__(self, holders: tuple[int, ...] = (SLOW,)) -> None:
        self._holders = holders
        self.history: dict[str, object] = {}

    def occupancy(self) -> dict[int, int]:
        return {FAST: 100, MID: 100, SLOW: 0}

    def free_space(self) -> dict[int, int | None]:
        return {FAST: 900, MID: 900, SLOW: None}

    def holds(self, file_id: int) -> tuple[int, ...]:
        del file_id
        return self._holders

    def size_of(self, file_id: int) -> int:
        del file_id
        return 154004


def event(file_id: int = 7) -> AccessEvent:
    return AccessEvent(
        file_id=file_id,
        size_bytes=154004,
        op="read",
        epoch=0,
        step=1,
        phase="train",
    )


# --------------------------------------------------------------------------- #
# Action space
# --------------------------------------------------------------------------- #
def test_action_names_are_placement_only() -> None:
    assert RL_ACTION_NAMES == ("NOOP", "PROMOTE_FAST", "PROMOTE_MID")
    # PIN/PREFETCH were removed: hidden long-term costs, identical immediate reward.
    assert "PIN_FAST" not in RL_ACTION_NAMES
    assert "PREFETCH_FAST" not in RL_ACTION_NAMES


def test_rl_action_mapping() -> None:
    view = FakeView()
    assert rl_action_to_actions(0, 7, view) == []
    assert rl_action_to_actions(1, 7, view) == [Promote(file_id=7, tier=FAST)]
    assert rl_action_to_actions(2, 7, view) == [Promote(file_id=7, tier=MID)]
    # Invalid / out-of-range indices degrade safely to NOOP.
    assert rl_action_to_actions(99, 7, view) == []
    assert rl_action_to_actions(-1, 7, view) == []


def test_rl_action_mapping_is_noop_when_already_in_tier() -> None:
    assert rl_action_to_actions(1, 7, FakeView((FAST,))) == []
    assert rl_action_to_actions(2, 7, FakeView((MID,))) == []


# --------------------------------------------------------------------------- #
# Tabular agent
# --------------------------------------------------------------------------- #
def test_discretize_state_uses_only_observable_fields() -> None:
    state = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    state[0] = 150.0  # size_kb -> chunk bucket
    state[4] = 1.0  # phase_train (index 4 in FEATURE_NAMES)
    state[15] = 0.0  # not in fast
    state[17] = 0.1  # fast occupancy
    assert discretize_state(state) == (1, 0, 1, 0)


def test_tabular_agent_learns_preferred_action() -> None:
    # Drive the update rule directly (deterministically) rather than depending
    # on epsilon-greedy exploration stumbling onto the rewarded action.
    agent = TabularQLearningAgent(3, learning_rate=0.5, seed=0)
    state = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    state[0] = 150.0
    state[4] = 1.0
    for _ in range(5):
        agent.update(state, 0, 0.0, state, True)
        agent.update(state, 1, 1.0, state, True)
        agent.update(state, 2, 0.0, state, True)
    assert agent.select_action(state, greedy=True) == 1
    q = agent.q_values(state)
    assert q[1] > q[0] and q[1] > q[2]
    assert agent.update_count == 15


def test_tabular_agent_save_load_roundtrip(tmp_path) -> None:
    agent = TabularQLearningAgent(3, seed=0)
    state = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    agent.update(state, 1, 1.0, state, True)
    path = agent.save(tmp_path / "agent.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["kind"] == "tabular"

    restored = load_agent({"rl": {"model_path": str(path)}})
    assert isinstance(restored, TabularQLearningAgent)
    assert restored.training is False
    assert restored.epsilon == agent.epsilon
    assert restored.q == agent.q


def test_load_agent_dispatches_legacy_linear(tmp_path) -> None:
    from rl_agent import LinearQLearningAgent

    legacy = LinearQLearningAgent(len(FEATURE_NAMES), 3, seed=0)
    payload = legacy.to_dict()
    payload.pop("kind", None)  # simulate an older artefact without kind
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    restored = load_agent({"rl": {"model_path": str(path)}})
    assert isinstance(restored, LinearQLearningAgent)
    assert restored.training is False


def test_load_agent_rejects_unknown_kind(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"kind": "deep", "version": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_agent({"rl": {"model_path": str(path)}})


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def test_env_reset_step_shape_and_metrics() -> None:
    config = load_config()
    config["epochs"] = 1
    env = TieringEnv(config, "A", 0, chunk_limit=5)
    observation, _ = env.reset()
    assert observation.shape == (len(FEATURE_NAMES),)
    assert env.action_space is not None
    assert env.action_space.n == len(RL_ACTION_NAMES)

    steps = 0
    terminated = False
    while not terminated:
        observation, reward, terminated, truncated, info = env.step(
            steps % len(RL_ACTION_NAMES)
        )
        assert observation.shape == (len(FEATURE_NAMES),)
        assert np.isfinite(reward)
        assert set(info) >= {"steps", "tier_served", "fast_hit_rate"}
        steps += 1
        assert steps < 1000, "episode did not terminate"
    assert "data_wait_ms" in env.final_info
    assert env.final_info["data_wait_ms"] >= 0.0
    env.close()


def test_env_rejects_invalid_action() -> None:
    config = load_config()
    config["epochs"] = 1
    env = TieringEnv(config, "A", 0, chunk_limit=5)
    env.reset()
    with pytest.raises(ValueError):
        env.step(99)
    env.close()


# --------------------------------------------------------------------------- #
# FullContextTierPolicy
# --------------------------------------------------------------------------- #
class FixedAgent:
    """Deterministic greedy agent for injecting a chosen RL action."""

    def __init__(self, action: int) -> None:
        self.action = action

    def select_action(self, state, *, greedy: bool = False) -> int:
        del state, greedy
        return self.action


def test_full_context_policy_injects_rl_action() -> None:
    policy = FullContextTierPolicy(agent=FixedAgent(1))
    actions = policy.on_access(event(), FakeView())
    assert actions == [Promote(file_id=7, tier=FAST)]

    policy_mid = FullContextTierPolicy(agent=FixedAgent(2))
    assert policy_mid.on_access(event(), FakeView()) == [Promote(file_id=7, tier=MID)]


def test_full_context_policy_registered() -> None:
    assert "full_context" in POLICY_CLASSES
    assert POLICY_CLASSES["full_context"] is FullContextTierPolicy


def test_full_context_policy_no_agent_degrades_to_promote_fast() -> None:
    policy = FullContextTierPolicy(agent=None)
    assert policy.on_access(event(), FakeView()) == [Promote(file_id=7, tier=FAST)]


def test_full_context_policy_writes_to_origin() -> None:
    policy = FullContextTierPolicy(agent=FixedAgent(1))
    assert policy.on_write(event(), FakeView()) == SLOW


# --------------------------------------------------------------------------- #
# Pipeline smoke
# --------------------------------------------------------------------------- #
def test_small_rl_pipeline(tmp_path) -> None:
    config = load_config()
    config["epochs"] = 1
    config["rl"] = dict(
        config["rl"],
        train_scenarios=["A"],
        train_seeds=[0],
        test_scenario="A",
        test_seeds=[1],
        episodes=2,
        model_path=str(tmp_path / "agent.json"),
        report_dir=str(tmp_path / "report"),
        use_classifier=False,
        use_reuse_predictor=False,
    )
    report = train_and_evaluate(config)
    assert set(report) >= {"split", "training", "evaluation", "artefacts"}
    assert report["training"]["episodes"] == 2
    assert set(report["evaluation"]) >= {"full_context", "rl_only", "lru"}
    for metrics in report["evaluation"].values():
        assert metrics["data_wait_ms"] >= 0.0
        assert 0.0 <= metrics["fast_hit_rate"] <= 1.0
    assert (tmp_path / "agent.json").is_file()
    assert (tmp_path / "report" / "reward_curve.png").is_file()
    assert (tmp_path / "report" / "reward_curves.csv").is_file()
    assert (tmp_path / "report" / "rl_report.json").is_file()


def test_train_agent_returns_frozen_tabular_agent() -> None:
    config = load_config()
    config["epochs"] = 1
    config["rl"] = dict(
        config["rl"],
        train_scenarios=["A"],
        train_seeds=[0],
        episodes=1,
    )
    agent, curve = train_agent(config, chunk_limit=5)
    assert isinstance(agent, TabularQLearningAgent)
    assert agent.training is False
    assert len(curve) == 1
    assert {"mean_reward", "data_wait_ms"} <= set(curve.columns)


def test_full_context_runs_in_simulator_with_injected_agent() -> None:
    config = load_config()
    config["epochs"] = 1
    config["rl"] = dict(config["rl"], use_classifier=False, use_reuse_predictor=False)
    agent = TabularQLearningAgent(len(RL_ACTION_NAMES))
    result = run_simulation(
        config, "full_context", "C", 3, epochs=1, chunk_limit=5, rl_agent=agent
    )
    assert result.metrics["data_wait_ms"] >= 0.0
    assert 0.0 <= result.metrics["fast_hit_rate"] <= 1.0
    assert set(result.logger.trace_frame()["true_type"].unique())
