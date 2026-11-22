"""Command-line entry point for single virtual-time experiments.

Example:
    python run_experiments.py --policy lru --scenario A --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import load_config, resolve_path
from policies import POLICY_CLASSES
from simulator import run_simulation


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=sorted(POLICY_CLASSES))
    parser.add_argument("--scenario", required=True, choices=["A", "B", "C", "D", "E", "F"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--trace-fmt", default="csv", choices=["csv", "parquet"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--model", choices=("cnn", "mlp"), default=None)
    parser.add_argument("--compute-source", choices=("manual", "profile"), default=None)
    parser.add_argument(
        "--context-ablation",
        action="store_true",
        help="Run ContextRules with and without its declared-type channel.",
    )
    return parser.parse_args()


def run_context_ablation(
    config: dict,
    scenario: str,
    seed: int,
    *,
    epochs: int | None = None,
) -> pd.DataFrame:
    """Return one context-vs-no-context ablation row for one run."""
    metrics = {
        name: run_simulation(config, name, scenario, seed, epochs=epochs).metrics
        for name in ("context_rules", "context_rules_noctx")
    }
    with_ctx = metrics["context_rules"]
    without_ctx = metrics["context_rules_noctx"]
    keys = ("data_wait_ms", "throughput", "fast_hit_rate", "migration_bytes")
    row = {"scenario": scenario, "seed": seed}
    for key in keys:
        row[f"{key}_context"] = with_ctx[key]
        row[f"{key}_noctx"] = without_ctx[key]
    row["data_wait_saved_ms"] = without_ctx["data_wait_ms"] - with_ctx["data_wait_ms"]
    return pd.DataFrame([row])


def main() -> None:
    """Run one experiment and write its metrics and trace files."""
    args = parse_args()
    config = load_config(args.config)
    from train import apply_model_selection, model_name

    config = apply_model_selection(
        config, name=args.model, compute_source=args.compute_source
    )

    if args.context_ablation:
        frame = run_context_ablation(
            config, args.scenario, args.seed, epochs=args.epochs
        )
        output = (
            Path(args.output_dir)
            / args.scenario
            / f"context_ablation_seed{args.seed}.csv"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        print(frame.to_string(index=False))
        print(f"ablation: {output}")
        return

    if not args.policy:
        raise SystemExit("--policy is required unless --context-ablation is set")

    result = run_simulation(
        config,
        policy_name=args.policy,
        scenario=args.scenario,
        seed=args.seed,
        epochs=args.epochs,
    )

    output_base = Path(args.output_dir)
    if args.model is not None and args.model != "cnn":
        output_base = resolve_path("results/models") / args.model / "virtual"
    output_dir = output_base / args.scenario
    stem = f"{args.policy}_seed{args.seed}"
    trace_path = result.logger.write_trace(
        output_dir / f"{stem}_trace.{args.trace_fmt}", fmt=args.trace_fmt
    )
    metrics_path = result.logger.write_metrics(output_dir / f"{stem}_metrics.csv")
    migration_path = result.logger.write_migration_log(
        output_dir / f"{stem}_migrations.csv"
    )

    print(
        f"policy={args.policy} scenario={args.scenario} seed={args.seed} "
        f"model={model_name(config)} compute_source={config['model']['compute_source']}"
    )
    for name in (
        "data_wait_ms",
        "throughput",
        "fast_hit_rate",
        "migration_bytes",
        "wasted_prefetch_bytes",
        "latency_mean_ms",
        "latency_p95_ms",
        "elapsed_ms",
    ):
        print(f"  {name}: {result.metrics[name]:.4f}")
    print(f"trace: {trace_path}")
    print(f"metrics: {metrics_path}")
    print(f"migrations: {migration_path}")


if __name__ == "__main__":
    main()
