"""Phase 7: CPU-only tabular Q-learning for dynamic data tiering.

The production agent is deliberately small: tabular Q-learning over a
discretisation of the observable feature vector from ``features.py`` (at most
``3 * 2 * 4 * 4`` states by 3 actions). No deep network, no GPU and no large
replay buffer are used. A linear function-approximation agent is retained as a
reference implementation; it was found to be unstable on this workload and is
not used by the reported pipeline.

Training happens through :class:`rl_env.TieringEnv` in virtual time. Evaluation
feeds the trained agent into :class:`~policies.FullContextTierPolicy` and runs
the ordinary simulator, so reported metrics use the exact same definitions as
every other policy. DQN is intentionally omitted: it adds memory/compute cost
without changing the research question for this scaled-down prototype.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from config import load_config, resolve_path
from features import FEATURE_NAMES
from policies import RL_ACTION_NAMES
from rl_env import TieringEnv
from simulator import run_simulation

AGENT_VERSION = 1

# Union of the two agent implementations. The pipeline uses tabular; linear is
# kept as a reference and for loading any earlier artefacts.
Agent = Union["TabularQLearningAgent", "LinearQLearningAgent"]


# --------------------------------------------------------------------------- #
# State discretisation for tabular Q-learning
# --------------------------------------------------------------------------- #
def discretize_state(state: Sequence[float]) -> tuple[int, int, int, int]:
    """Bucket the observable feature vector into a small tabular state.

    Only policy-observable fields are used: file size, whether the file is in
    the fast tier, the workload phase, and fast-tier occupancy. No names, true
    types, oracle or declared types are involved.
    """
    values = np.asarray(state, dtype=np.float64)
    size_kb = float(values[0]) if values.size > 0 else 0.0
    if size_kb < 1.0:
        size_bucket = 0  # metadata-like
    elif size_kb < 1000.0:
        size_bucket = 1  # chunk/validation-like
    else:
        size_bucket = 2  # checkpoint-like

    in_fast = 1 if values.size > 15 and float(values[15]) >= 0.5 else 0
    phase_block = values[3:7] if values.size > 6 else np.zeros(4)
    phase = int(np.argmax(phase_block)) if phase_block.size == 4 else 0
    occupancy = float(values[17]) if values.size > 17 else 0.0
    occupancy_bucket = int(np.clip(occupancy * 4.0, 0, 3))
    return size_bucket, in_fast, phase, occupancy_bucket


# --------------------------------------------------------------------------- #
# Tabular agent (used by the Phase 7 pipeline)
# --------------------------------------------------------------------------- #
class TabularQLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration.

    Unlike linear function approximation, a small discrete state table cannot
    suffer the divergent bootstrapping observed with a shared linear value over
    continuous features. The table has at most ``3 * 2 * 4 * 4`` states by 3
    actions, so it stays tiny and CPU/memory friendly.
    """

    kind = "tabular"

    def __init__(
        self,
        n_actions: int,
        *,
        learning_rate: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.2,
        min_epsilon: float = 0.02,
        epsilon_decay: float = 0.995,
        seed: int = 0,
    ) -> None:
        """Create an empty Q table."""
        self.n_actions = int(n_actions)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.initial_epsilon = float(epsilon)
        self.min_epsilon = float(min_epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.rng = np.random.default_rng(seed)
        self.q: dict[tuple[tuple[int, int, int, int], int], float] = {}
        self.training = True
        self.update_count = 0

    def observe(self, state: Sequence[float]) -> None:
        """Tabular learning needs no normalisation; keep the interface uniform."""

    def q_values(self, state: Sequence[float]) -> np.ndarray:
        """Return Q(s, a) for every action."""
        key = discretize_state(state)
        return np.asarray(
            [self.q.get((key, action), 0.0) for action in range(self.n_actions)],
            dtype=np.float64,
        )

    def select_action(self, state: Sequence[float], *, greedy: bool = False) -> int:
        """Choose an action index; ``greedy=True`` disables exploration."""
        if self.training and not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(state)))

    def update(
        self,
        state: Sequence[float],
        action: int,
        reward: float,
        next_state: Sequence[float],
        terminated: bool,
    ) -> float:
        """Perform one tabular Q-learning update and return the TD error."""
        if not self.training:
            return 0.0
        key = discretize_state(state)
        action = int(action)
        future = 0.0 if terminated else float(np.max(self.q_values(next_state)))
        current = self.q.get((key, action), 0.0)
        td_error = float(reward) + self.gamma * future - current
        self.q[(key, action)] = current + self.learning_rate * td_error
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.update_count += 1
        return float(td_error)

    def freeze(self) -> None:
        """Disable exploration and learning."""
        self.training = False

    def to_dict(self) -> dict:
        """Serialise the table and hyper-parameters."""
        return {
            "version": AGENT_VERSION,
            "kind": self.kind,
            "n_actions": self.n_actions,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "initial_epsilon": self.initial_epsilon,
            "min_epsilon": self.min_epsilon,
            "epsilon_decay": self.epsilon_decay,
            "update_count": self.update_count,
            "q": {
                f"{','.join(str(part) for part in key[0])}|{key[1]}": value
                for key, value in self.q.items()
            },
        }

    def save(self, path: str | Path) -> Path:
        """Write the agent to ``path`` as JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return destination

    @classmethod
    def from_dict(cls, payload: dict) -> "TabularQLearningAgent":
        """Restore an agent from a serialised dictionary."""
        if int(payload.get("version", -1)) != AGENT_VERSION:
            raise ValueError("Unsupported RL agent version")
        agent = cls(
            int(payload["n_actions"]),
            learning_rate=float(payload["learning_rate"]),
            gamma=float(payload["gamma"]),
            epsilon=float(payload["epsilon"]),
            min_epsilon=float(payload["min_epsilon"]),
            epsilon_decay=float(payload["epsilon_decay"]),
        )
        for encoded, value in payload["q"].items():
            state_text, action_text = encoded.split("|")
            # Accept both the new "1,0,2,0|1" and legacy "(1, 0, 2, 0)|1" forms.
            state_text = state_text.strip("()")
            state = tuple(int(part.strip()) for part in state_text.split(","))
            agent.q[(state, int(action_text))] = float(value)
        agent.update_count = int(payload["update_count"])
        agent.freeze()
        return agent


# --------------------------------------------------------------------------- #
# Linear Q-learning (reference implementation)
# --------------------------------------------------------------------------- #
class RunningStats:
    """Welford running mean/variance for stable online normalisation."""

    def __init__(self, size: int) -> None:
        """Create empty statistics for a vector of length ``size``."""
        self.count = 0
        self.mean = np.zeros(size, dtype=np.float64)
        self.m2 = np.zeros(size, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        """Absorb one observation into the running statistics."""
        x = np.asarray(values, dtype=np.float64)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Return a clipped, standardised copy of ``values``."""
        x = np.asarray(values, dtype=np.float64)
        if self.count == 0:
            return np.clip(x, -5.0, 5.0).astype(np.float32)
        std = np.sqrt(self.m2 / self.count) + 1e-8
        return np.clip((x - self.mean) / std, -5.0, 5.0).astype(np.float32)

    def to_dict(self) -> dict:
        """Serialise the statistics."""
        return {
            "count": self.count,
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RunningStats":
        """Restore statistics from a serialised dictionary."""
        stats = cls(len(payload["mean"]))
        stats.count = int(payload["count"])
        stats.mean = np.asarray(payload["mean"], dtype=np.float64)
        stats.m2 = np.asarray(payload["m2"], dtype=np.float64)
        return stats


class LinearQLearningAgent:
    """Linear Q-learning with epsilon-greedy exploration (reference agent)."""

    kind = "linear"

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        *,
        learning_rate: float = 0.01,
        gamma: float = 0.9,
        epsilon: float = 0.2,
        min_epsilon: float = 0.02,
        epsilon_decay: float = 0.995,
        max_td_error: float = 5.0,
        seed: int = 0,
    ) -> None:
        """Create an agent with small random initial weights."""
        self.n_features = int(n_features)
        self.n_actions = int(n_actions)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.initial_epsilon = float(epsilon)
        self.min_epsilon = float(min_epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.max_td_error = float(max_td_error)
        self.rng = np.random.default_rng(seed)
        self.stats = RunningStats(self.n_features)
        self.weights = self.rng.normal(
            loc=0.0, scale=0.01, size=(self.n_actions, self.n_features)
        ).astype(np.float64)
        self.training = True
        self.update_count = 0

    # ------------------------------------------------------------------ #
    # Inference and learning
    # ------------------------------------------------------------------ #
    def observe(self, state: Sequence[float]) -> None:
        """Update normalisation statistics during training only."""
        if self.training:
            self.stats.update(np.asarray(state, dtype=np.float64))

    def q_values(self, state: Sequence[float]) -> np.ndarray:
        """Return Q(s, a) for every action."""
        vector = self.stats.normalize(np.asarray(state, dtype=np.float64))
        return self.weights @ vector

    def select_action(self, state: Sequence[float], *, greedy: bool = False) -> int:
        """Choose an action index; ``greedy=True`` disables exploration."""
        if self.training and not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(state)))

    def update(
        self,
        state: Sequence[float],
        action: int,
        reward: float,
        next_state: Sequence[float],
        terminated: bool,
    ) -> float:
        """Perform one one-step Q-learning update and return the TD error."""
        if not self.training:
            return 0.0
        x = self.stats.normalize(np.asarray(state, dtype=np.float64))
        x_next = self.stats.normalize(np.asarray(next_state, dtype=np.float64))
        action = int(action)
        current = float(self.weights[action] @ x)
        future = 0.0 if terminated else float(np.max(self.weights @ x_next))
        td_error = float(reward) + self.gamma * future - current
        td_error = float(np.clip(td_error, -self.max_td_error, self.max_td_error))
        self.weights[action] += self.learning_rate * td_error * x
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.update_count += 1
        return td_error

    def freeze(self) -> None:
        """Disable exploration, normalisation updates and weight changes."""
        self.training = False

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Serialise weights, statistics and hyper-parameters."""
        return {
            "version": AGENT_VERSION,
            "kind": self.kind,
            "feature_names": list(FEATURE_NAMES),
            "action_names": list(RL_ACTION_NAMES),
            "n_features": self.n_features,
            "n_actions": self.n_actions,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "initial_epsilon": self.initial_epsilon,
            "min_epsilon": self.min_epsilon,
            "epsilon_decay": self.epsilon_decay,
            "max_td_error": self.max_td_error,
            "update_count": self.update_count,
            "weights": self.weights.tolist(),
            "stats": self.stats.to_dict(),
        }

    def save(self, path: str | Path) -> Path:
        """Write the agent to ``path`` as JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return destination

    @classmethod
    def from_dict(cls, payload: dict) -> "LinearQLearningAgent":
        """Restore an agent from a serialised dictionary."""
        if int(payload.get("version", -1)) != AGENT_VERSION:
            raise ValueError("Unsupported RL agent version")
        if list(payload["feature_names"]) != list(FEATURE_NAMES):
            raise ValueError("RL agent feature schema does not match this build")
        if list(payload["action_names"]) != list(RL_ACTION_NAMES):
            raise ValueError("RL agent action schema does not match this build")
        agent = cls(
            int(payload["n_features"]),
            int(payload["n_actions"]),
            learning_rate=float(payload["learning_rate"]),
            gamma=float(payload["gamma"]),
            epsilon=float(payload["epsilon"]),
            min_epsilon=float(payload["min_epsilon"]),
            epsilon_decay=float(payload["epsilon_decay"]),
            max_td_error=float(payload.get("max_td_error", 5.0)),
        )
        agent.stats = RunningStats.from_dict(payload["stats"])
        agent.weights = np.asarray(payload["weights"], dtype=np.float64)
        agent.update_count = int(payload["update_count"])
        agent.freeze()
        return agent


def load_agent(config: dict) -> Agent:
    """Load a trained agent, dispatching on the saved ``kind`` field.

    Older artefacts have no ``kind`` field and are treated as the legacy linear
    agent, so existing checkpoints remain loadable.
    """
    path = resolve_path(config["rl"]["model_path"])
    if not path.is_file():
        raise FileNotFoundError(
            f"RL agent not found: {path}. Run `python rl_agent.py` first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    kind = payload.get("kind", "linear")
    if kind == "tabular":
        return TabularQLearningAgent.from_dict(payload)
    if kind == "linear":
        return LinearQLearningAgent.from_dict(payload)
    raise ValueError(f"Unknown RL agent kind: {kind!r}")


def load_agent_from_path(path: str | Path) -> Agent:
    """Load an agent from an explicit path using a minimal config shim."""
    return load_agent({"rl": {"model_path": str(path)}})


# --------------------------------------------------------------------------- #
# Training and evaluation
# --------------------------------------------------------------------------- #
def _episode_pairs(config: dict) -> list[tuple[str, int]]:
    """Return the deterministic (scenario, seed) training cycle."""
    params = config["rl"]
    scenarios = [str(item) for item in params["train_scenarios"]]
    seeds = [int(seed) for seed in params["train_seeds"]]
    return [(scenario, seed) for scenario in scenarios for seed in seeds]


def train_agent(
    config: dict,
    *,
    episodes: int | None = None,
    epochs: int | None = None,
    chunk_limit: int | None = None,
) -> tuple[TabularQLearningAgent, pd.DataFrame]:
    """Train one agent and return it with the per-episode reward curve."""
    params = config["rl"]
    pairs = _episode_pairs(config)
    if not pairs:
        raise ValueError("rl.train_scenarios and rl.train_seeds must be non-empty")
    total_episodes = int(params["episodes"] if episodes is None else episodes)

    agent = TabularQLearningAgent(
        len(RL_ACTION_NAMES),
        learning_rate=float(params["learning_rate"]),
        gamma=float(params["gamma"]),
        epsilon=float(params["epsilon"]),
        min_epsilon=float(params["min_epsilon"]),
        epsilon_decay=float(params["epsilon_decay"]),
        seed=int(params.get("seed", 0)),
    )

    records: list[dict] = []
    for episode in range(total_episodes):
        scenario, seed = pairs[episode % len(pairs)]
        env = TieringEnv(
            config,
            scenario,
            seed,
            epochs=epochs,
            chunk_limit=chunk_limit,
        )
        observation, _ = env.reset()
        total_reward = 0.0
        tds: list[float] = []
        while True:
            agent.observe(observation)
            action = agent.select_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            tds.append(
                agent.update(
                    observation, action, reward, next_observation, terminated or truncated
                )
            )
            total_reward += float(reward)
            observation = next_observation
            if terminated or truncated:
                break
        info = env.final_info
        steps = max(1, int(info.get("steps", 1)))
        records.append(
            {
                "episode": episode + 1,
                "scenario": scenario,
                "seed": seed,
                "total_reward": total_reward,
                "mean_reward": total_reward / steps,
                "mean_abs_td_error": float(np.mean(np.abs(tds))) if tds else 0.0,
                "data_wait_ms": float(info.get("data_wait_ms", 0.0)),
                "fast_hit_rate": float(info.get("fast_hit_rate", 0.0)),
                "migration_bytes": float(info.get("migration_bytes_total", 0.0)),
                "epsilon": agent.epsilon,
            }
        )
        env.close()

    agent.freeze()
    return agent, pd.DataFrame(records)


def _mean_metrics(
    config: dict,
    policy_name: str,
    scenario: str,
    seeds: Iterable[int],
    *,
    agent: Optional[Agent] = None,
    epochs: int | None = None,
    chunk_limit: int | None = None,
) -> dict:
    """Average the exact simulator metrics over a list of seeds."""
    keys = (
        "data_wait_ms",
        "throughput",
        "fast_hit_rate",
        "migration_bytes",
    )
    seed_list = [int(seed) for seed in seeds]
    totals = {key: 0.0 for key in keys}
    for seed in seed_list:
        result = run_simulation(
            config,
            policy_name,
            scenario,
            seed,
            epochs=epochs,
            chunk_limit=chunk_limit,
            rl_agent=agent,
        )
        for key in keys:
            totals[key] += float(result.metrics[key])
    return {
        "scenario": scenario,
        "seeds": seed_list,
        **{key: value / len(seed_list) for key, value in totals.items()},
    }


def evaluate_agent(
    config: dict,
    agent: Agent,
    *,
    epochs: int | None = None,
    chunk_limit: int | None = None,
) -> dict:
    """Compare FullContextTierPolicy with LRU on the held-out scenario/seeds."""
    params = config["rl"]
    scenario = str(params["test_scenario"])
    seeds = [int(seed) for seed in params["test_seeds"]]

    # RL-only ablation: disable the Phase 5/6 learned components to isolate the
    # contribution of RL and to expose any distribution shift in the classifier.
    rl_only_config = dict(config)
    rl_only_config["rl"] = dict(
        config["rl"], use_classifier=False, use_reuse_predictor=False
    )
    return {
        "full_context": _mean_metrics(
            config, "full_context", scenario, seeds, agent=agent,
            epochs=epochs, chunk_limit=chunk_limit,
        ),
        "rl_only": _mean_metrics(
            rl_only_config, "full_context", scenario, seeds, agent=agent,
            epochs=epochs, chunk_limit=chunk_limit,
        ),
        "lru": _mean_metrics(
            config, "lru", scenario, seeds, epochs=epochs, chunk_limit=chunk_limit
        ),
    }


def write_reward_curve(curve: pd.DataFrame, report_dir: Path) -> list[Path]:
    """Write the reward curve as CSV and PNG (Matplotlib Agg, CPU-only)."""
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "reward_curves.csv"
    curve.to_csv(csv_path, index=False)
    png_path = report_dir / "reward_curve.png"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(curve["episode"], curve["mean_reward"], marker="o", markersize=3)
    axes[0].set_ylabel("mean reward / access")
    axes[0].set_title("ContextTier tabular Q-learning")
    axes[0].grid(alpha=0.3)
    axes[1].plot(
        curve["episode"], curve["data_wait_ms"], marker="s", markersize=3, color="tab:red"
    )
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("data wait (ms)")
    axes[1].grid(alpha=0.3)
    # bbox_inches handles tight cropping without conflicting with a globally
    # enabled constrained_layout (set by plots.py in the same test process).
    figure.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return [csv_path, png_path]


def train_and_evaluate(
    config: dict,
    *,
    episodes: int | None = None,
    epochs: int | None = None,
    chunk_limit: int | None = None,
) -> dict:
    """Run the full Phase 7 pipeline and write all artefacts."""
    params = config["rl"]
    agent, curve = train_agent(
        config, episodes=episodes, epochs=epochs, chunk_limit=chunk_limit
    )
    evaluation = evaluate_agent(
        config, agent, epochs=epochs, chunk_limit=chunk_limit
    )
    report_dir = resolve_path(params["report_dir"])
    model_path = agent.save(resolve_path(params["model_path"]))
    curve_paths = write_reward_curve(curve, report_dir)

    report = {
        "split": {
            "train_scenarios": list(params["train_scenarios"]),
            "train_seeds": list(params["train_seeds"]),
            "test_scenario": params["test_scenario"],
            "test_seeds": list(params["test_seeds"]),
        },
        "training": {
            "episodes": int(len(curve)),
            "first_mean_reward": float(curve["mean_reward"].iloc[0]) if len(curve) else 0.0,
            "last_mean_reward": float(curve["mean_reward"].iloc[-1]) if len(curve) else 0.0,
            "final_epsilon": agent.epsilon,
            "weight_updates": agent.update_count,
        },
        "evaluation": evaluation,
        "artefacts": {
            "agent": str(model_path),
            "reward_curve_csv": str(curve_paths[0]),
            "reward_curve_png": str(curve_paths[1]),
        },
        "notes": (
            "Reward = tier_service (fast 1.0 / mid 0.6 / origin 0.0) "
            "- wait/slow_latency - migration_weight*migration_MB "
            "- group_data_wait/wait_scale. Learned policy sees only observable "
            "features: no names, true types, oracle or declared types. DQN omitted "
            "deliberately: a tiny tabular Q-learner is sufficient and CPU/memory "
            "cheaper; linear function approximation was retained only as a reference "
            "because it diverged on this workload."
        ),
    }
    (report_dir / "rl_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    """Train, evaluate and print the Phase 7 RL report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--chunk-limit", type=int, default=None)
    args = parser.parse_args()
    report = train_and_evaluate(
        load_config(args.config),
        episodes=args.episodes,
        epochs=args.epochs,
        chunk_limit=args.chunk_limit,
    )
    print("split:", report["split"])
    print("training:", report["training"])
    print("evaluation:", report["evaluation"])
    print("artefacts:", report["artefacts"])


if __name__ == "__main__":
    main()
