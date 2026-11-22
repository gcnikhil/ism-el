"""Post-processing for the paper's final statistical tables.

Every value written by this module is derived from ``results/full/*.csv``.
It produces:

* ``bootstrap_comparison.csv`` / ``.json`` — paired FullContextTier advantage
  over LRU and LFU: mean, 95% percentile-bootstrap CI, and seeds won. No
  p-values are reported: with a paired design the effect size and its
  interval are the readable summary, and no test statistic is pre-registered.
* ``elapsed_decomposition.csv`` — exact additive components of elapsed time
  for every policy in Scenarios B and C, plus the FullContextTier-minus-LRU
  deltas that explain any throughput difference.
* ``scenario_c_best_baseline.csv`` — comparison against the best *deployable*
  observable baseline in Scenario C (LRU, LFU or ARC), chosen by mean data
  wait from the same 20-seed matrix.

Run after ``python experiments.py``:

    python paper_stats.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCENARIOS: tuple[str, ...] = ("A", "B", "C", "D")
BASELINES: tuple[str, ...] = ("lru", "lfu")
# Capacity-bound controls are excluded from "best deployable baseline":
# AllSlow has no cache and AllFast assumes enough Tier 0 for every file.
DEPLOYABLE_BASELINES: tuple[str, ...] = ("lru", "lfu", "arc")
FOCUS_POLICY = "full_context"

DECOMPOSITION_COLUMNS: tuple[str, ...] = (
    "promotion_ms",
    "train_read_ms",
    "validation_read_ms",
    "other_read_ms",
    "checkpoint_write_ms",
    "train_compute_ms",
    "flush_ms",
)

DEFAULT_RESULTS = Path("results/full")
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 0


def _load_tables(results_dir: Path) -> dict[str, pd.DataFrame]:
    required = ("raw_runs.csv", "summary.csv")
    missing = [name for name in required if not (results_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing result tables in {results_dir}: {', '.join(missing)}. "
            "Run `python experiments.py` first."
        )
    tables = {
        "raw": pd.read_csv(results_dir / "raw_runs.csv"),
        "summary": pd.read_csv(results_dir / "summary.csv"),
    }
    sweep_path = results_dir / "sweep_runs.csv"
    if sweep_path.is_file():
        tables["sweep"] = pd.read_csv(sweep_path)
    return tables


def paired_bootstrap(
    differences: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return (mean, 95% CI low, 95% CI high) of a paired difference.

    The CI is the percentile interval of bootstrapped means, resampling pairs
    with replacement. ``differences`` must be one value per seed; the sign
    convention is set by the caller.
    """
    values = np.asarray(differences, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, values.size), replace=True)
    boot_means = draws.mean(axis=1)
    low, high = np.percentile(boot_means, (2.5, 97.5))
    return float(values.mean()), float(low), float(high)


def _paired_frame(
    raw: pd.DataFrame,
    scenario: str,
    baseline: str,
    metric: str,
    sweep: str = "matrix",
    sweep_value: float | None = None,
) -> pd.Series:
    """Return baseline-minus-focus differences indexed by seed.

    For ``data_wait_ms`` a positive difference means FullContextTier stalled
    less; for ``throughput`` we invert the sign so positive always means
    "FullContextTier is better". ``sweep``/``sweep_value`` select the main
    matrix or one point of the compute-time sweep.
    """
    subset = raw[raw["scenario"] == scenario]
    if "sweep" in raw.columns:
        subset = subset[subset["sweep"] == sweep]
    if sweep_value is not None and "sweep_value" in raw.columns:
        subset = subset[subset["sweep_value"] == sweep_value]
    base = subset[subset["policy"] == baseline].set_index("seed")[metric]
    focus = subset[subset["policy"] == FOCUS_POLICY].set_index("seed")[metric]
    difference = base.reindex(focus.index) - focus
    if metric == "throughput":
        difference = -difference
    return difference.dropna()


def bootstrap_tables(
    raw: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Build the paired-difference table for every scenario and baseline."""
    records: list[dict] = []
    sources: list[tuple[str, float | None, tuple[str, ...]]] = [("matrix", None, SCENARIOS)]
    if "sweep" in raw.columns:
        compute = raw[raw["sweep"] == "compute_ms"]
        for value in sorted(float(item) for item in compute["sweep_value"].unique()):
            sources.append(("compute_ms", value, ("C",)))
    for sweep_name, sweep_value, scenarios in sources:
        for scenario in scenarios:
            for baseline in BASELINES:
                for metric in ("data_wait_ms", "throughput"):
                    paired = _paired_frame(
                        raw, scenario, baseline, metric, sweep_name, sweep_value
                    )
                    mean, low, high = paired_bootstrap(
                        paired.to_numpy(), resamples=resamples, seed=seed
                    )
                    seeds_won = int((paired > 0).sum()) if len(paired) else 0
                    records.append(
                        {
                            "sweep": sweep_name,
                            "sweep_value": sweep_value,
                            "scenario": scenario,
                            "baseline": baseline,
                            "focus": FOCUS_POLICY,
                            "metric": metric,
                            "difference_direction": (
                                "baseline_minus_focus"
                                if metric == "data_wait_ms"
                                else "focus_minus_baseline"
                            ),
                            "n_seeds": int(len(paired)),
                            "mean_difference": mean,
                            "ci95_low": low,
                            "ci95_high": high,
                            "seeds_won": seeds_won,
                            "win_fraction": (
                                seeds_won / len(paired) if len(paired) else float("nan")
                            ),
                    }
                )
    table = pd.DataFrame(records)
    metadata = {
        "method": "paired percentile bootstrap of the mean difference",
        "resamples": int(resamples),
        "bootstrap_seed": int(seed),
        "win_definition": (
            "positive paired data_wait difference (baseline - focus), or positive "
            "throughput difference (focus - baseline)"
        ),
        "p_values": "omitted: effect sizes and percentile intervals are reported instead",
        "coverage": "20-seed main matrix plus every compute_ms sweep point on Scenario C",
        "source": "results/full/raw_runs.csv",
    }
    return table, metadata


def decomposition_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Exact elapsed-time decomposition for Scenarios B and C, all policies."""
    columns = list(DECOMPOSITION_COLUMNS)
    needed = [f"{name}_mean" for name in columns] + ["elapsed_ms_mean", "n_seeds"]
    missing = [name for name in needed if name not in summary.columns]
    if missing:
        raise ValueError(f"summary.csv is missing decomposition columns: {missing}")
    metric_columns = [f"{name}_mean" for name in columns] + [
        "elapsed_ms_mean",
        "data_wait_ms_mean",
        "throughput_mean",
    ]
    base_columns = ["policy", "policy_kind", "scenario", "n_seeds"]
    frames: list[pd.DataFrame] = []
    for scenario in ("B", "C"):
        subset = summary[summary["scenario"] == scenario].copy()
        subset = subset[base_columns + metric_columns]
        subset["component_sum_ms"] = subset[[f"{name}_mean" for name in columns]].sum(axis=1)
        subset["residual_ms"] = subset["elapsed_ms_mean"] - subset["component_sum_ms"]
        frames.append(subset)
    return pd.concat(frames, ignore_index=True)


def _policies_in(summary: pd.DataFrame) -> list[str]:
    return sorted(str(policy) for policy in summary["policy"].unique())


def best_baseline_table(summary: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compare FullContextTier with the best deployable baseline in Scenario C."""
    subset = summary[
        (summary["scenario"] == "C")
        & (summary["policy"].isin(list(DEPLOYABLE_BASELINES) + [FOCUS_POLICY]))
    ].copy()
    if subset.empty:
        return pd.DataFrame(), {"best_baseline": None, "n_seeds": 0}
    baseline_rows = subset[subset["policy"] != FOCUS_POLICY].sort_values(
        "data_wait_ms_mean"
    )
    if baseline_rows.empty:
        return pd.DataFrame(), {"best_baseline": None, "n_seeds": 0}
    best = str(baseline_rows.iloc[0]["policy"])
    focus_row = subset[subset["policy"] == FOCUS_POLICY]
    if focus_row.empty:
        return pd.DataFrame(), {"best_baseline": best, "n_seeds": 0}
    focus = focus_row.iloc[0]
    chosen = baseline_rows.iloc[0]
    saved = float(chosen["data_wait_ms_mean"]) - float(focus["data_wait_ms_mean"])
    record = {
        "scenario": "C",
        "best_baseline": best,
        "selection_metric": "data_wait_ms_mean",
        "n_seeds": int(focus["n_seeds"]),
        "baseline_data_wait_ms_mean": float(chosen["data_wait_ms_mean"]),
        "baseline_data_wait_ms_std": float(chosen.get("data_wait_ms_std", 0.0)),
        "focus_data_wait_ms_mean": float(focus["data_wait_ms_mean"]),
        "focus_data_wait_ms_std": float(focus.get("data_wait_ms_std", 0.0)),
        "data_wait_saved_ms": saved,
        "data_wait_saved_pct": (
            100.0 * saved / float(chosen["data_wait_ms_mean"])
            if chosen["data_wait_ms_mean"]
            else 0.0
        ),
        "baseline_throughput": float(chosen["throughput_mean"]),
        "focus_throughput": float(focus["throughput_mean"]),
        "throughput_gain": float(focus["throughput_mean"]) - float(chosen["throughput_mean"]),
        "candidate_baselines": list(DEPLOYABLE_BASELINES),
        "source": "results/full/summary.csv",
    }
    return pd.DataFrame([record]), {
        "best_baseline": best,
        "n_seeds": int(focus["n_seeds"]),
        "excluded": ["allslow", "allfast"],
        "exclusion_reason": "capacity-bound controls, not deployable cache policies",
    }


def generate_all(
    results_dir: str | Path = DEFAULT_RESULTS,
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Path]:
    """Write every post-processing artefact and return their paths."""
    directory = Path(results_dir)
    tables = _load_tables(directory)
    # Bootstrap comparisons cover the 20-seed matrix AND every compute-sweep
    # point; the latter live in sweep_runs.csv, not raw_runs.csv.
    combined = tables["raw"]
    if "sweep" in tables:
        combined = pd.concat([tables["raw"], tables["sweep"]], ignore_index=True)
    bootstrap, bootstrap_meta = bootstrap_tables(
        combined, resamples=resamples, seed=bootstrap_seed
    )
    decomposition = decomposition_table(tables["summary"])
    best_baseline, best_meta = best_baseline_table(tables["summary"])

    outputs = {
        "bootstrap_csv": directory / "bootstrap_comparison.csv",
        "bootstrap_json": directory / "bootstrap_comparison.json",
        "decomposition_csv": directory / "elapsed_decomposition.csv",
        "best_baseline_csv": directory / "scenario_c_best_baseline.csv",
    }
    bootstrap.to_csv(outputs["bootstrap_csv"], index=False)
    outputs["bootstrap_json"].write_text(
        json.dumps(bootstrap_meta, indent=2), encoding="utf-8"
    )
    decomposition.to_csv(outputs["decomposition_csv"], index=False)
    best_baseline.to_csv(outputs["best_baseline_csv"], index=False)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name, path in generate_all(
        args.results_dir,
        resamples=args.resamples,
        bootstrap_seed=args.bootstrap_seed,
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
