"""Phase 8: full experiment matrix, mean+/-std aggregation and sweeps.

This is the single entry point for producing the paper's result tables. It runs
every registered policy on every scenario with every configured seed, then runs
two documented parameter sweeps. It writes *raw per-run rows* (all exact metrics)
and *aggregated mean/std tables* to CSV, plus a JSON manifest recording the
exact configuration and which artefacts were used.

Honesty notes (carried into the manifest so they cannot be lost):
* `clairvoyant` and `belady` are oracle upper bounds; in Scenario C the seeded
  oracle is deliberately withheld from `clairvoyant`.
* `context_rules` receives constructor-injected declared types (privileged
  context); `context_rules_noctx` is its ablation with an empty map.
* Learned models (classifier, embeddings, RL agent) are trained once on their
  declared splits and are NOT retrained at each sweep point. Sweep results are
  therefore transfer-under-shift measurements, not per-setting training runs.

Example:
    python experiments.py
    python experiments.py --matrix-only
    python experiments.py --sweeps-only --policies lru,full_context
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from config import load_config, resolve_path
from policies import POLICY_CLASSES
from simulator import run_simulation
from tiers import FAST

# Exact metric names produced by simulator.run_simulation (Phase 2 contract).
METRIC_NAMES: tuple[str, ...] = (
    "data_wait_ms",
    "data_wait_pct",
    "throughput",
    "fast_hit_rate",
    "mid_hit_rate",
    "slow_hit_rate",
    "migration_bytes",
    "wasted_prefetch_bytes",
    "latency_mean_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "elapsed_ms",
    "read_count",
    "sample_count",
    # Exact additive decomposition of elapsed_ms (paper check 2).
    "promotion_ms",
    "train_read_ms",
    "validation_read_ms",
    "other_read_ms",
    "checkpoint_write_ms",
    "train_compute_ms",
    "flush_ms",
)

SCENARIOS: tuple[str, ...] = ("A", "B", "C", "D")

# Sweep dimensions. Chosen as the two knobs that directly affect the research
# question: cache capacity and the compute time that data waits must overlap.
DEFAULT_FAST_CAPACITIES_MB: tuple[float, ...] = (10.0, 25.0, 50.0)
DEFAULT_COMPUTE_MS: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
SWEEP_SCENARIOS: tuple[str, ...] = ("C",)
SWEEP_POLICIES: tuple[str, ...] = ("lru", "context_rules", "full_context")

ROW_COLUMNS: tuple[str, ...] = (
    "sweep",
    "sweep_value",
    "policy",
    "policy_kind",
    "scenario",
    "seed",
) + METRIC_NAMES

ORACLE_POLICIES = frozenset({"clairvoyant", "belady"})
PRIVILEGED_POLICIES = frozenset({"context_rules", "context_rules_noctx"})
LEARNED_POLICIES = frozenset({"classifier", "embedding", "full_context"})


def policy_kind(policy: str) -> str:
    """Label a policy by the information it uses (for honest paper tables)."""
    if policy in ORACLE_POLICIES:
        return "oracle_upper_bound"
    if policy in PRIVILEGED_POLICIES:
        return "declared_type_upper_bound"
    if policy in LEARNED_POLICIES:
        return "learned_observable"
    return "observable_baseline"


# --------------------------------------------------------------------------- #
# Model reuse
# --------------------------------------------------------------------------- #
class LearnedModels:
    """Load each learned artefact once and inject the same objects into runs.

    The model objects are stateless at inference time (frozen RL agent, fixed
    boosters), so sharing them across runs is safe and avoids reloading XGBoost
    and Word2Vec for every seed.
    """

    def __init__(self, config: dict, *, load: bool = True) -> None:
        self.config = config
        self.classifier: object | None = None
        self.predictor: object | None = None
        self.agent: object | None = None
        self.missing: list[str] = []
        if load:
            self._load()

    def _load(self) -> None:
        from classifier import load_model as load_classifier
        from embeddings import load_predictor
        from rl_agent import load_agent

        loaders = (
            ("classifier", lambda: load_classifier(self.config)),
            ("embeddings", lambda: load_predictor(self.config)),
            ("rl_agent", lambda: load_agent(self.config)),
        )
        for name, loader in loaders:
            try:
                value = loader()
            except FileNotFoundError:
                self.missing.append(name)
                print(f"[experiments] {name} artefact missing; policy will be skipped")
                continue
            if name == "classifier":
                self.classifier = value
            elif name == "embeddings":
                self.predictor = value
            else:
                self.agent = value

    def available(self, policy: str) -> bool:
        """Return whether ``policy`` has all the artefacts it needs."""
        if policy == "classifier":
            return self.classifier is not None
        if policy == "embedding":
            return self.predictor is not None
        if policy == "full_context":
            # The RL agent is required; classifier/predictor are optional.
            return self.agent is not None
        return True

    def kwargs(self, policy: str) -> dict:
        """Return the injection kwargs for one ``run_simulation`` call.

        ``run_simulation`` uses its ``classifier_model`` parameter polymorphically
        (a classifier for ``ClassifierPolicy``, a reuse predictor for
        ``EmbeddingPrefetchPolicy``), so the correct object must be selected.
        """
        if policy == "classifier":
            return {"classifier_model": self.classifier}
        if policy == "embedding":
            return {"classifier_model": self.predictor}
        if policy == "full_context":
            return {
                "classifier_model": self.classifier,
                "rl_agent": self.agent,
                "reuse_predictor": self.predictor,
            }
        return {}


# --------------------------------------------------------------------------- #
# Core run helpers
# --------------------------------------------------------------------------- #
def run_one(
    config: dict,
    policy: str,
    scenario: str,
    seed: int,
    *,
    epochs: int | None = None,
    models: Optional[LearnedModels] = None,
    chunk_limit: int | None = None,
    sweep: str = "matrix",
    sweep_value: float | None = None,
) -> dict:
    """Run one simulation and return a flat CSV row with all exact metrics."""
    # Flushed progress line makes a stalled run diagnosable when the matrix is
    # launched as a long-lived process (and documents exactly what was run).
    print(
        f"[experiments] start sweep={sweep} value={sweep_value} "
        f"policy={policy} scenario={scenario} seed={seed}",
        flush=True,
    )
    kwargs = models.kwargs(policy) if models is not None else {}
    result = run_simulation(
        config,
        policy_name=policy,
        scenario=scenario,
        seed=int(seed),
        epochs=epochs,
        chunk_limit=chunk_limit,
        **kwargs,
    )
    row: dict = {
        "sweep": sweep,
        "sweep_value": sweep_value,
        "policy": policy,
        "policy_kind": policy_kind(policy),
        "scenario": scenario,
        "seed": int(seed),
    }
    for name in METRIC_NAMES:
        row[name] = float(result.metrics[name])
    return row


def run_matrix(
    config: dict,
    policies: Sequence[str],
    scenarios: Sequence[str],
    seeds: Iterable[int],
    *,
    epochs: int | None = None,
    models: Optional[LearnedModels] = None,
    chunk_limit: int | None = None,
) -> pd.DataFrame:
    """Run every policy x scenario x seed combination."""
    rows: list[dict] = []
    seed_list = [int(seed) for seed in seeds]
    for policy in policies:
        if models is not None and not models.available(policy):
            print(f"[experiments] skipping {policy}: required artefact missing")
            continue
        for scenario in scenarios:
            for seed in seed_list:
                rows.append(
                    run_one(
                        config,
                        policy,
                        scenario,
                        seed,
                        epochs=epochs,
                        models=models,
                        chunk_limit=chunk_limit,
                    )
                )
    return pd.DataFrame(rows)


def summarize(
    raw: pd.DataFrame,
    group_cols: Sequence[str] = ("policy", "policy_kind", "scenario"),
) -> pd.DataFrame:
    """Aggregate raw rows into mean and sample std (ddof=1) for every metric.

    ``policy_kind`` is retained when present so the information class of each
    row (baseline / learned / declared-type / oracle) survives aggregation.
    """
    group_cols = [column for column in group_cols if column in raw.columns]
    if not group_cols:
        group_cols = ["policy"]
    if raw.empty:
        return pd.DataFrame(
            columns=group_cols + ["n_seeds"] + [f"{m}_{s}" for m in METRIC_NAMES for s in ("mean", "std")]
        )
    columns = list(group_cols) + ["n_seeds"]
    for metric in METRIC_NAMES:
        columns.extend((f"{metric}_mean", f"{metric}_std"))
    grouped = raw.groupby(list(group_cols), sort=True)
    records: list[dict] = []
    for key, frame in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        record = dict(zip(group_cols, key_tuple))
        record["n_seeds"] = int(len(frame))
        for metric in METRIC_NAMES:
            record[f"{metric}_mean"] = float(frame[metric].mean())
            record[f"{metric}_std"] = float(frame[metric].std(ddof=1)) if len(frame) > 1 else 0.0
        records.append(record)
    return pd.DataFrame(records, columns=columns)


def _copy_with_fast_capacity(config: dict, capacity_mb: float) -> dict:
    """Return a config copy with the fast-tier capacity overridden."""
    result = copy.deepcopy(config)
    for tier in result["tiers"]:
        if int(tier["id"]) == FAST:
            tier["capacity_mb"] = float(capacity_mb)
    return result


def _copy_with_compute_ms(config: dict, compute_ms: float) -> dict:
    """Return a config copy with ``compute_ms`` overridden."""
    result = copy.deepcopy(config)
    result["compute_ms"] = float(compute_ms)
    return result


def run_sweeps(
    config: dict,
    policies: Sequence[str],
    scenarios: Sequence[str],
    seeds: Iterable[int],
    *,
    epochs: int | None = None,
    models: Optional[LearnedModels] = None,
    chunk_limit: int | None = None,
    fast_capacities_mb: Sequence[float] = DEFAULT_FAST_CAPACITIES_MB,
    compute_values_ms: Sequence[float] = DEFAULT_COMPUTE_MS,
) -> pd.DataFrame:
    """Run the fast-capacity and compute-time sweeps; return raw rows."""
    rows: list[dict] = []
    seed_list = [int(seed) for seed in seeds]

    # Scenario D is excluded: run_simulation forces its fast tier large enough
    # to hold every file, so a capacity override would have no effect.
    sweep_scenarios = [scenario for scenario in scenarios if scenario in SWEEP_SCENARIOS]
    if not sweep_scenarios:
        return pd.DataFrame(columns=list(ROW_COLUMNS))

    for policy in policies:
        if policy not in SWEEP_POLICIES:
            continue
        if models is not None and not models.available(policy):
            print(f"[experiments] skipping sweep policy {policy}: artefact missing")
            continue

        for capacity in fast_capacities_mb:
            sweep_config = _copy_with_fast_capacity(config, capacity)
            for scenario in sweep_scenarios:
                for seed in seed_list:
                    rows.append(
                        run_one(
                            sweep_config,
                            policy,
                            scenario,
                            seed,
                            epochs=epochs,
                            models=models,
                            chunk_limit=chunk_limit,
                            sweep="fast_capacity_mb",
                            sweep_value=float(capacity),
                        )
                    )

        for compute_ms in compute_values_ms:
            sweep_config = _copy_with_compute_ms(config, compute_ms)
            for scenario in sweep_scenarios:
                for seed in seed_list:
                    rows.append(
                        run_one(
                            sweep_config,
                            policy,
                            scenario,
                            seed,
                            epochs=epochs,
                            models=models,
                            chunk_limit=chunk_limit,
                            sweep="compute_ms",
                            sweep_value=float(compute_ms),
                        )
                    )
    return pd.DataFrame(rows)


def write_manifest(
    output_dir: Path,
    config: dict,
    *,
    policies: Sequence[str],
    scenarios: Sequence[str],
    seeds: Sequence[int],
    epochs: int | None,
    fast_capacities_mb: Sequence[float],
    compute_values_ms: Sequence[float],
    models: LearnedModels,
    raw_matrix: pd.DataFrame,
    raw_sweeps: pd.DataFrame,
) -> Path:
    """Write configuration provenance and honesty notes as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "phase": 8,
        "engine": "virtual",
        "config": {
            "epochs": int(epochs if epochs is not None else config.get("epochs", 3)),
            "compute_ms": config["compute_ms"],
            "compute_ms_source": "results/real/step_time_benchmark.json (median of 200 CPU steps)",
            "batch_size": config["batch_size"],
            "tiers": config["tiers"],
            "seeds": list(seeds),
        },
        "policies": list(policies),
        "scenarios": list(scenarios),
        "sweeps": {
            "fast_capacity_mb": list(fast_capacities_mb),
            "compute_ms": list(compute_values_ms),
            "sweep_scenarios": list(SWEEP_SCENARIOS),
            "sweep_policies": list(SWEEP_POLICIES),
        },
        "model_paths": {
            "classifier": config["classifier"]["model_path"],
            "embeddings": config["embeddings"]["model_path"],
            "rl_agent": config["rl"]["model_path"],
        },
        "missing_artefacts": list(models.missing),
        "run_counts": {
            "matrix_runs": int(len(raw_matrix)),
            "sweep_runs": int(len(raw_sweeps)),
        },
        "metric_names": list(METRIC_NAMES),
        "fast_hit_semantics": (
            "fast_hit is Tier-0 residency when the read arrived, snapshotted before "
            "on_access/on_tick demand promotions; tier_served is the tier that "
            "actually served the read and may differ after a promotion."
        ),
        "elapsed_decomposition": (
            "promotion_ms + train_read_ms + validation_read_ms + other_read_ms + "
            "checkpoint_write_ms + train_compute_ms + flush_ms == elapsed_ms exactly."
        ),
        "honesty_notes": [
            "fast_hit is arrival-time residency, not post-promotion service tier.",
            "compute_ms is the measured median of 200 real TinyCNN CPU steps on this machine, not a placeholder.",
            "clairvoyant and belady are oracle upper bounds; Scenario C withholds the seeded oracle from clairvoyant.",
            "context_rules receives constructor-injected declared types; context_rules_noctx is its empty-map ablation.",
            "Learned models are trained once on their declared splits and are NOT retrained at each sweep point; sweeps measure transfer under shift.",
            "Scenario D is excluded from the capacity sweep because run_simulation forces its fast tier large enough to hold every file.",
            "All results come from the user's own runs; no numbers are illustrative or invented.",
        ],
    }
    path = output_dir / "experiment_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def run_all(
    config: dict,
    *,
    output_dir: str | Path = "results/full",
    policies: Optional[Sequence[str]] = None,
    scenarios: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    epochs: int | None = None,
    matrix: bool = True,
    sweeps: bool = True,
    chunk_limit: int | None = None,
    fast_capacities_mb: Sequence[float] = DEFAULT_FAST_CAPACITIES_MB,
    compute_values_ms: Sequence[float] = DEFAULT_COMPUTE_MS,
    load_models: bool = True,
) -> dict:
    """Run the requested parts and write all CSV/JSON artefacts."""
    output = Path(output_dir)
    policies = list(policies) if policies is not None else list(POLICY_CLASSES)
    scenarios = list(scenarios) if scenarios is not None else list(SCENARIOS)
    seeds = [int(seed) for seed in seeds] if seeds is not None else [int(s) for s in config["seeds"]]
    models = LearnedModels(config, load=load_models)

    raw_matrix = run_matrix(
        config, policies, scenarios, seeds, epochs=epochs, models=models, chunk_limit=chunk_limit
    ) if matrix else pd.DataFrame()
    raw_sweeps = run_sweeps(
        config,
        policies,
        scenarios,
        seeds,
        epochs=epochs,
        models=models,
        chunk_limit=chunk_limit,
        fast_capacities_mb=fast_capacities_mb,
        compute_values_ms=compute_values_ms,
    ) if sweeps else pd.DataFrame()

    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    if matrix:
        artifacts["raw_runs"] = output / "raw_runs.csv"
        raw_matrix.to_csv(artifacts["raw_runs"], index=False)
        artifacts["summary"] = output / "summary.csv"
        summarize(raw_matrix).to_csv(artifacts["summary"], index=False)
    if sweeps:
        artifacts["sweep_runs"] = output / "sweep_runs.csv"
        raw_sweeps.to_csv(artifacts["sweep_runs"], index=False)
        artifacts["sweep_summary"] = output / "sweep_summary.csv"
        summarize(
            raw_sweeps,
            group_cols=("sweep", "sweep_value", "policy", "policy_kind", "scenario"),
        ).to_csv(artifacts["sweep_summary"], index=False)
    artifacts["manifest"] = write_manifest(
        output,
        config,
        policies=policies,
        scenarios=scenarios,
        seeds=seeds,
        epochs=epochs,
        fast_capacities_mb=fast_capacities_mb,
        compute_values_ms=compute_values_ms,
        models=models,
        raw_matrix=raw_matrix,
        raw_sweeps=raw_sweeps,
    )
    return {
        "artifacts": artifacts,
        "matrix_runs": int(len(raw_matrix)),
        "sweep_runs": int(len(raw_sweeps)),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="results/full")
    parser.add_argument("--model", choices=("cnn", "mlp"), default=None)
    parser.add_argument("--compute-source", choices=("manual", "profile"), default=None)
    parser.add_argument("--policies", default=None, help="Comma-separated policy names.")
    parser.add_argument("--scenarios", default=None, help="Comma-separated scenario letters.")
    parser.add_argument("--seeds", default=None, help="Comma-separated integer seeds.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--chunk-limit", type=int, default=None)
    parser.add_argument("--matrix-only", action="store_true")
    parser.add_argument("--sweeps-only", action="store_true")
    parser.add_argument(
        "--fast-capacities", default=None, help="Comma-separated fast-tier capacities in MB."
    )
    parser.add_argument(
        "--compute-values", default=None, help="Comma-separated compute_ms values."
    )
    return parser.parse_args()


def main() -> None:
    """Run the full Phase 8 experiment suite and print the artefact paths."""
    args = parse_args()
    config = load_config(args.config)
    from train import apply_model_selection

    config = apply_model_selection(
        config, name=args.model, compute_source=args.compute_source
    )
    policies = args.policies.split(",") if args.policies else None
    scenarios = args.scenarios.split(",") if args.scenarios else None
    seeds = [int(item) for item in args.seeds.split(",")] if args.seeds else None
    capacities = (
        [float(item) for item in args.fast_capacities.split(",")]
        if args.fast_capacities
        else DEFAULT_FAST_CAPACITIES_MB
    )
    computes = (
        [float(item) for item in args.compute_values.split(",")]
        if args.compute_values
        else DEFAULT_COMPUTE_MS
    )
    matrix = not args.sweeps_only
    sweeps = not args.matrix_only
    output_dir = args.output_dir
    if args.model is not None and args.model != "cnn" and output_dir == "results/full":
        output_dir = str(resolve_path("results/models") / args.model / "full")
    report = run_all(
        config,
        output_dir=output_dir,
        policies=policies,
        scenarios=scenarios,
        seeds=seeds,
        epochs=args.epochs,
        matrix=matrix,
        sweeps=sweeps,
        chunk_limit=args.chunk_limit,
        fast_capacities_mb=capacities,
        compute_values_ms=computes,
    )
    print(f"matrix runs: {report['matrix_runs']}")
    print(f"sweep runs: {report['sweep_runs']}")
    for name, path in report["artifacts"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
