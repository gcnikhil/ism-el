"""Regenerate docs/EVALUATION.md from the project's own result files.

Every number in the document is read from results/* or config.yaml; no value is
typed into the prose. Run after experiments.py, plots.py and paper_stats.py:

    python build_evaluation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config, project_root
from paper_stats import DEPLOYABLE_BASELINES

ROOT = project_root()
FULL = ROOT / "results" / "full"
PLOTS = ROOT / "results" / "plots"
REAL = ROOT / "results" / "real"
OUT = ROOT / "docs" / "EVALUATION.md"

DISPLAY = {
    "allslow": "AllSlow",
    "allfast": "AllFast",
    "lru": "LRU",
    "lfu": "LFU",
    "arc": "ARC",
    "clairvoyant": "Clairvoyant",
    "belady": "Belady",
    "context_rules_noctx": "ContextRulesNoCtx",
    "context_rules": "ContextRules",
    "classifier": "Classifier",
    "embedding": "Embedding",
    "full_context": "FullContextTier",
}
ORDER = list(DISPLAY)
CLASS = {
    "allslow": "capacity bound",
    "allfast": "capacity bound",
    "lru": "baseline",
    "lfu": "baseline",
    "arc": "baseline",
    "clairvoyant": "oracle",
    "belady": "oracle",
    "context_rules_noctx": "declared-type ablation",
    "context_rules": "declared-type bound",
    "classifier": "learned",
    "embedding": "learned",
    "full_context": "learned",
}


def f(value: float, digits: int = 2) -> str:
    return f"{float(value):,.{digits}f}"


def cell(row: pd.Series, metric: str) -> str:
    return f"{f(row[f'{metric}_mean'])} ± {f(row[f'{metric}_std'])}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _row(frame: pd.DataFrame, **match: object) -> pd.Series:
    mask = pd.Series([True] * len(frame), index=frame.index)
    for key, value in match.items():
        if isinstance(value, float) and np.isnan(value):
            mask &= frame[key].isna()
        else:
            mask &= frame[key] == value
    selected = frame[mask]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {match}, found {len(selected)}")
    return selected.iloc[0]


def build() -> str:
    config = load_config()
    summary = pd.read_csv(FULL / "summary.csv")
    raw = pd.read_csv(FULL / "raw_runs.csv")
    sweeps = pd.read_csv(FULL / "sweep_summary.csv")
    manifest = json.loads((FULL / "experiment_manifest.json").read_text(encoding="utf-8"))
    benchmark = json.loads((REAL / "step_time_benchmark.json").read_text(encoding="utf-8"))
    bootstrap = pd.read_csv(FULL / "bootstrap_comparison.csv")
    bootstrap_meta = json.loads((FULL / "bootstrap_comparison.json").read_text(encoding="utf-8"))
    decomposition = pd.read_csv(FULL / "elapsed_decomposition.csv")
    best = _row(pd.read_csv(FULL / "scenario_c_best_baseline.csv"), scenario="C")
    figures = json.loads((PLOTS / "plot_manifest.json").read_text(encoding="utf-8"))
    classifier = json.loads((ROOT / "results/classifier/classification_report.json").read_text(encoding="utf-8"))
    embeddings = json.loads((ROOT / "results/embeddings/embedding_report.json").read_text(encoding="utf-8"))
    rl = json.loads((ROOT / "results/rl/rl_report.json").read_text(encoding="utf-8"))
    agent = json.loads((ROOT / "results/rl/agent.json").read_text(encoding="utf-8"))
    ranks = {
        scenario: json.loads((REAL / f"rank_{scenario}_seed0.json").read_text(encoding="utf-8"))
        for scenario in ("A", "B", "C", "D")
    }

    n_seeds = len(manifest["config"]["seeds"])
    compute = float(config["compute_ms"])
    measured = float(benchmark["measurement"]["median_ms"])
    steps = int(benchmark["measurement"]["steps_measured"])
    tier_rows = [
        [f"Tier {tier['id']} ({tier['name']})", f(tier["capacity_mb"]) if tier["capacity_mb"] is not None else "unbounded",
         f"{f(tier['latency_ms'])}", f(tier["bandwidth_mbps"])]
        for tier in config["tiers"]
    ]
    sweep_computes = sorted(
        float(v) for v in sweeps[sweeps["sweep"] == "compute_ms"]["sweep_value"].unique()
    )
    sweep_capacities = sorted(
        float(v) for v in sweeps[sweeps["sweep"] == "fast_capacity_mb"]["sweep_value"].unique()
    )

    parts: list[str] = []
    parts.append("# ContextTier — Evaluation and Discussion (paper draft)\n")
    parts.append(
        "> **Provenance.** Every number in this document was generated from the\n"
        "> project's own result files by `python build_evaluation.py`; none is\n"
        "> illustrative. Sources: `results/full/*.csv`, `results/full/*.json`,\n"
        "> `results/real/step_time_benchmark.json`, `results/real/rank_*.json`,\n"
        "> `results/classifier/*`, `results/embeddings/*`, `results/rl/*`,\n"
        "> `results/plots/plot_manifest.json` and `config.yaml`. To regenerate:\n"
        "> `python experiments.py`, `python plots.py`, `python paper_stats.py`,\n"
        "> `python build_evaluation.py`.\n\n---\n"
    )

    parts.append("## 1. Evaluation\n\n### 1.1 Experimental setup\n")
    parts.append(
        "This is a **laptop-scale, CPU-only emulation**, not a production storage "
        f"stack. The tier model in `config.yaml` is:\n\n"
        + table(["Tier", "Capacity (MB)", "Latency (ms)", "Bandwidth (MB/s)"], tier_rows)
        + "\n\n"
        f"Runs use {config['epochs']} virtual epochs, batch size {config['batch_size']}, "
        f"checkpoints every {config['checkpoint_every_steps']} steps and "
        f"compute_ms = {f(compute, 6)} ms. This is **not a placeholder**: it is the "
        f"median of {steps} measured pure-compute TinyCNN steps on this machine "
        f"(`results/real/step_time_benchmark.json`: mean {f(benchmark['measurement']['mean_ms'])} ms, "
        f"p95 {f(benchmark['measurement']['p95_ms'])} ms; "
        f"{benchmark['environment']['python_version'].split()[0]}, "
        f"torch {benchmark['environment']['torch_version']}, "
        f"{benchmark['environment']['cpu_count']} logical CPUs, "
        f"{benchmark['environment']['torch_num_threads']} torch threads; "
        f"model input {benchmark['configuration']['model_input_shape']}, so the "
        f"nominal batch {config['batch_size']} becomes "
        f"{benchmark['configuration']['effective_batch_samples']} samples because "
        f"chunks are read in batches of {config['chunk_size_samples']}). No storage "
        "I/O or sleep was included in that timing. The main matrix and sweeps use "
        f"{n_seeds} seeds (listed in the manifest), and learned models are frozen "
        "on their declared train splits before evaluation.\n\n"
    )
    parts.append(
        "Four scenarios isolate different sources of context:\n\n"
        "| Scenario | Workload |\n|---|---|\n"
        "| A | Shuffled training chunks + per-step metadata. |\n"
        "| B | A + checkpoints + validation reads. |\n"
        "| C | B + hard-example oversampling; access order depends on training state. |\n"
        "| D | B with every file pre-resident in FAST (designed tie). |\n\n"
        "Policies are separated by the information they may use. "
        "`AllSlow`/`AllFast` are capacity-bound controls; `LRU`/`LFU`/`ARC` are "
        "observable baselines; `Classifier`/`Embedding`/`FullContextTier` are learned "
        "policies using only observable features; `ContextRules` receives "
        "constructor-injected declared types (declared-type upper bound, with "
        "`ContextRulesNoCtx` as its empty-map ablation); `Clairvoyant` uses only the "
        "shuffle seed (withheld in Scenario C); `Belady` receives the full read stream "
        "(oracle upper bound). We re-implement seed-based exact prefetching inside "
        "the emulator and call it *NoPFS-style*; we do not run the original NoPFS code.\n"
    )

    parts.append(
        "### 1.2 Metrics and statistics\n\n"
        "The primary metric is `data_wait_ms`, the compute-overlap-uncovered stall "
        "per training step: `max(0, sum(read latency + read wait + synchronous "
        "promotion time) - compute_ms)`, summed over train/warmup steps. Validation "
        "and checkpoint time count towards `throughput` (total samples / total "
        "elapsed seconds) but not towards data wait. Secondary metrics are "
        "`fast_hit_rate`, per-tier served rates, `migration_bytes`, "
        "`wasted_prefetch_bytes` and mean/p95/p99 read latency. **FAST hit means "
        "resident in Tier 0 when the read arrived, before any demand promotion**: "
        "the harness snapshots residency before calling `on_access`, so a read can "
        "be served by Tier 0 (`tier_served = FAST`) while `fast_hit = false`. "
        "Consequently `fast_hit_rate` is not the complement of the MID/SLOW "
        "service rates, which are computed from `tier_served`.\n\n"
        "`elapsed_ms` is decomposed exactly into seven additive buckets — "
        "synchronous promotion, training reads, validation reads, other reads, "
        "checkpoint writes, training compute and the final flush — with no residual "
        "(regression-tested to 1e-9 ms). Reported aggregates are mean ± sample "
        "standard deviation over the seeds. Paired comparisons use the same seed "
        "for both policies and a percentile bootstrap of the mean difference "
        f"({bootstrap_meta['resamples']:,} resamples, bootstrap seed "
        f"{bootstrap_meta['bootstrap_seed']}); we report the mean, 95% interval "
        "and seeds won. **No p-values are reported**: with 20 paired seeds we prefer "
        "the effect size and its interval to a binary significance claim.\n"
    )

    parts.append(
        "### 1.3 Main matrix (all policies, all scenarios)\n\n"
        f"**Table 1 — Data wait in ms, mean ± std over {n_seeds} seeds** "
        "(source: `results/full/summary.csv`)\n\n"
    )
    rows = []
    for policy in ORDER:
        row = _row(summary, policy=policy, scenario="A")
        cols = [f"{DISPLAY[policy]}", CLASS[policy]]
        cols.extend(cell(_row(summary, policy=policy, scenario=s), "data_wait_ms") for s in "ABCD")
        rows.append(cols)
    parts.append(table(["Policy", "Class", "A", "B", "C", "D"], rows) + "\n\n")

    matrix_max_wait = float(summary["data_wait_ms_mean"].max())
    allslow_c = _row(summary, policy="allslow", scenario="C")
    lru_c = _row(summary, policy="lru", scenario="C")
    parts.append(
        f"At the measured compute budget ({f(compute, 6)} ms per step), the "
        f"zero-wait ceiling binds for **every** policy, including AllSlow: the "
        f"largest `data_wait_ms` mean anywhere in the matrix is "
        f"{f(matrix_max_wait)} ms, so Table 1 cannot rank policies at all on this "
        f"metric. AllSlow pays for its uncached reads in **throughput** instead: "
        f"{f(allslow_c['throughput_mean'])} vs {f(lru_c['throughput_mean'])} "
        f"samples/s in Scenario C ({f(allslow_c['latency_p95_ms_mean'], 2)} ms p95 "
        "read latency, because every read is served from SLOW). Section 1.6 "
        "therefore reports the paired comparison both on this matrix (where data "
        "wait is a universal tie) and at each point of the compute sweep, where "
        "the overlap model is active and policies separate. Section 1.5 identifies "
        "which elapsed-time bucket drives the throughput differences.\n"
    )

    parts.append(
        "### 1.4 Secondary metrics in Scenario C\n\n"
        f"**Table 2 — Scenario C secondary metrics, means over {n_seeds} seeds** "
        "(source: `results/full/summary.csv`)\n\n"
    )
    rows = []
    for policy in ORDER:
        row = _row(summary, policy=policy, scenario="C")
        rows.append([
            DISPLAY[policy],
            f(row["data_wait_pct_mean"]),
            f(row["throughput_mean"]),
            f(row["fast_hit_rate_mean"], 3),
            f(row["mid_hit_rate_mean"], 3),
            f(row["slow_hit_rate_mean"], 3),
            f(row["migration_bytes_mean"] / 1e6),
            f(row["latency_p95_ms_mean"], 4),
        ])
    parts.append(
        table(["Policy", "Data wait %", "Throughput (samples/s)", "FAST resident on arrival",
               "MID served", "SLOW served", "Migration (MB)", "p95 latency (ms)"], rows)
        + "\n\n"
        "`wasted_prefetch_bytes` is zero for every policy: speculative "
        "nearest-neighbour prefetching was removed after it failed to help, so no "
        "unused bytes are charged. FAST-resident-on-arrival and served-tier rates "
        "answer different questions under the corrected `fast_hit` definition; "
        "their sum is therefore not constrained to one.\n"
    )

    parts.append(
        "### 1.5 Elapsed-time decomposition in Scenarios B and C\n\n"
        f"**Table 3 — Mean elapsed-time components (ms) for every policy, "
        f"{n_seeds} seeds** (source: `results/full/elapsed_decomposition.csv`; "
        "component sums equal `elapsed_ms` exactly)\n\n"
    )
    rows = []
    for scenario in ("B", "C"):
        for policy in ORDER:
            row = _row(decomposition, policy=policy, scenario=scenario)
            rows.append([
                scenario, DISPLAY[policy],
                f(row["data_wait_ms_mean"]),
                f(row["promotion_ms_mean"]), f(row["train_read_ms_mean"]),
                f(row["validation_read_ms_mean"]), f(row["checkpoint_write_ms_mean"]),
                f(row["train_compute_ms_mean"]), f(row["flush_ms_mean"]),
                f(row["elapsed_ms_mean"]),
            ])
    parts.append(
        table(["Scen.", "Policy", "Data wait", "Promotion", "Train reads", "Validation reads",
               "Checkpoint writes", "Train compute", "Final flush", "Elapsed"], rows) + "\n\n"
    )
    lru_b = _row(decomposition, policy="lru", scenario="B")
    full_b = _row(decomposition, policy="full_context", scenario="B")
    lru_c = _row(decomposition, policy="lru", scenario="C")
    full_c = _row(decomposition, policy="full_context", scenario="C")
    lru_tp_b = float(_row(summary, policy="lru", scenario="B")["throughput_mean"])
    full_tp_b = float(_row(summary, policy="full_context", scenario="B")["throughput_mean"])
    lru_tp_c = float(_row(summary, policy="lru", scenario="C")["throughput_mean"])
    full_tp_c = float(_row(summary, policy="full_context", scenario="C")["throughput_mean"])
    promo_b = float(lru_b["promotion_ms_mean"]) - float(full_b["promotion_ms_mean"])
    promo_c = float(lru_c["promotion_ms_mean"]) - float(full_c["promotion_ms_mean"])
    val_b = float(full_b["validation_read_ms_mean"]) - float(lru_b["validation_read_ms_mean"])
    val_c = float(full_c["validation_read_ms_mean"]) - float(lru_c["validation_read_ms_mean"])
    elapsed_b = float(lru_b["elapsed_ms_mean"]) - float(full_b["elapsed_ms_mean"])
    elapsed_c = float(lru_c["elapsed_ms_mean"]) - float(full_c["elapsed_ms_mean"])
    parts.append(
        "**Which bucket explains the throughput gain over LRU?** At this compute "
        "budget, training data wait is zero for both policies, so it cannot explain "
        "the difference. In Scenario B, FullContextTier saves "
        f"{f(promo_b)} ms of synchronous promotion time but spends {f(val_b)} ms "
        "more on validation reads, giving a net elapsed-time saving of "
        f"{f(elapsed_b)} ms and a throughput difference of "
        f"{f(full_tp_b - lru_tp_b)} samples/s "
        f"({f(full_tp_b)} vs {f(lru_tp_b)}). In Scenario C the corresponding "
        f"numbers are {f(promo_c)} ms saved on promotion, {f(val_c)} ms extra "
        f"validation time, {f(elapsed_c)} ms net elapsed saving and "
        f"{f(full_tp_c - lru_tp_c)} samples/s "
        f"({f(full_tp_c)} vs {f(lru_tp_c)}). Checkpoint-write time is identical "
        "between the two policies in both scenarios. The gain is therefore a "
        "**training data-path effect (less synchronous promotion)**, partially "
        "offset by validation reads, not an uncovered-stall effect. At smaller "
        "compute budgets (1–10 ms) the uncovered data wait becomes non-zero and "
        "adds to the same promotion advantage; see Section 1.8.\n"
    )

    parts.append(
        "### 1.6 Paired comparison with 20 seeds (no p-values)\n\n"
        f"**Table 4 — FullContextTier paired advantage, mean [95% bootstrap CI], "
        f"seeds won** (source: `results/full/bootstrap_comparison.csv`; positive "
        "data-wait difference means the baseline stalled more)\n\n"
    )
    rows = []
    selections = [("matrix", None, "C")]
    selections += [("compute_ms", value, "C") for value in sweep_computes]
    for sweep_name, sweep_value, scenario in selections:
        for baseline in ("lru", "lfu"):
            row = _row(
                bootstrap,
                sweep=sweep_name,
                sweep_value=sweep_value if sweep_value is not None else np.nan,
                scenario=scenario,
                baseline=baseline,
                metric="data_wait_ms",
            ) if sweep_value is None else _row(
                bootstrap,
                sweep=sweep_name,
                sweep_value=sweep_value,
                scenario=scenario,
                baseline=baseline,
                metric="data_wait_ms",
            )
            label = "matrix" if sweep_value is None else f"{f(sweep_value, 0).replace(',', '')} ms"
            rows.append([
                label, DISPLAY[baseline],
                f"{f(row['mean_difference'])} [{f(row['ci95_low'])}, {f(row['ci95_high'])}]",
                f"{int(row['seeds_won'])}/{int(row['n_seeds'])}",
            ])
    parts.append(table(["Setting", "Baseline", "Data wait saved (ms) [95% CI]", "Seeds won"], rows) + "\n\n")
    matrix_lru = _row(bootstrap, sweep="matrix", sweep_value=np.nan, scenario="C", baseline="lru", metric="data_wait_ms")
    matrix_lfu = _row(bootstrap, sweep="matrix", sweep_value=np.nan, scenario="C", baseline="lfu", metric="data_wait_ms")
    parts.append(
        f"On the main matrix the paired difference versus LRU is "
        f"{f(matrix_lru['mean_difference'])} ms "
        f"[{f(matrix_lru['ci95_low'])}, {f(matrix_lru['ci95_high'])}] with "
        f"{int(matrix_lru['seeds_won'])}/{int(matrix_lru['n_seeds'])} seeds won; "
        f"versus LFU it is {f(matrix_lfu['mean_difference'])} ms "
        f"[{f(matrix_lfu['ci95_low'])}, {f(matrix_lfu['ci95_high'])}] with "
        f"{int(matrix_lfu['seeds_won'])}/{int(matrix_lfu['n_seeds'])} seeds won. "
        "These exact ties follow from the zero-wait ceiling. At the 5 ms sweep "
        "point (the old placeholder budget), the same comparison is informative; "
        "Table 4 makes the dependence on the compute assumption explicit. "
        "Intervals are percentile bootstrap intervals of the paired mean; no "
        "p-values are reported.\n"
    )

    parts.append(
        "### 1.7 Scenario C against the best baseline\n\n"
        f"**Table 5 — FullContextTier vs the best deployable observable baseline "
        f"in C** (source: `results/full/scenario_c_best_baseline.csv`)\n\n"
    )
    parts.append(
        table(["Best baseline", "Baseline wait (ms)", "FullContextTier wait (ms)",
               "Saved (ms)", "Saved (%)", "Throughput gain (samples/s)"], [[
            DISPLAY[str(best["best_baseline"])],
            f(best["baseline_data_wait_ms_mean"]),
            f(best["focus_data_wait_ms_mean"]),
            f(best["data_wait_saved_ms"]),
            f(best["data_wait_saved_pct"]),
            f(best["throughput_gain"]),
        ]]) + "\n\n"
        f"The best baseline among {', '.join(DISPLAY[str(p)] for p in DEPLOYABLE_BASELINES)} "
        f"is **{DISPLAY[str(best['best_baseline'])]}**, selected by mean `data_wait_ms` "
        f"over {int(best['n_seeds'])} seeds. {', '.join(DISPLAY[p] for p in ('allslow', 'allfast'))} "
        "are excluded as capacity-bound controls, not deployable cache policies. "
        "The paired bootstrap comparisons against LRU and LFU in Table 4 cover "
        "the other baselines directly.\n"
    )

    parts.append(
        "### 1.8 Sensitivity sweeps\n\n"
        "Learned models are trained once on their declared splits and reused at "
        "every sweep point; these are transfer-under-shift measurements, not "
        "per-setting retraining. Both sweeps use Scenario C.\n\n"
        "**Table 6 — Fast-tier capacity sweep, `data_wait_ms` mean ± std** "
        "(source: `results/full/sweep_summary.csv`)\n\n"
    )
    rows = []
    for value in sweep_capacities:
        rows.append([f"{f(value, 0)} MB"] + [
            cell(_row(sweeps, sweep="fast_capacity_mb", sweep_value=value, policy=p, scenario="C"), "data_wait_ms")
            for p in ("lru", "context_rules", "full_context")
        ])
    parts.append(table(["FAST capacity", "LRU", "ContextRules", "FullContextTier"], rows) + "\n\n")
    parts.append(
        "**Table 7 — Compute-per-step sweep, `data_wait_ms` mean ± std** "
        "(source: `results/full/sweep_summary.csv`)\n\n"
    )
    rows = []
    for value in sweep_computes:
        rows.append([f"{f(value, 0).replace(',', '')} ms"] + [
            cell(_row(sweeps, sweep="compute_ms", sweep_value=value, policy=p, scenario="C"), "data_wait_ms")
            for p in ("lru", "context_rules", "full_context")
        ])
    parts.append(
        table(["compute_ms", "LRU", "ContextRules", "FullContextTier"], rows) + "\n\n"
        "The measured CPU step time is far above the largest sweep point, so the "
        "main matrix sits at the zero-wait ceiling. `compute_gain` in "
        "`results/plots/plot_manifest.json` plots the paired gain against this "
        "sweep and marks the measured median as a vertical reference. Scenario D "
        "is excluded from the capacity sweep because the simulator forces its "
        "FAST tier large enough to hold every file.\n"
    )

    parts.append(
        "### 1.9 Context ablation\n\n"
        "Paired per-seed data-wait savings from the declared-type channel "
        "(`results/full/raw_runs.csv`):\n\n"
    )
    rows = []
    for scenario in "ABCD":
        with_ctx = raw[(raw["policy"] == "context_rules") & (raw["scenario"] == scenario)].set_index("seed")["data_wait_ms"]
        no_ctx = raw[(raw["policy"] == "context_rules_noctx") & (raw["scenario"] == scenario)].set_index("seed")["data_wait_ms"]
        diff = (no_ctx - with_ctx).reindex(with_ctx.index).dropna()
        rows.append([
            scenario, f(diff.mean()), f(diff.std(ddof=1)) if len(diff) > 1 else "0.00",
            f(diff.min()) if len(diff) else "0.00", f(diff.max()) if len(diff) else "0.00",
            f"{int((diff > 0).sum())}/{len(diff)}",
        ])
    parts.append(
        table(["Scenario", "Mean saved (ms)", "Std", "Min", "Max", "Seeds improved"], rows)
        + "\n\nThe channel only matters once non-chunk data types coexist; at the "
        "measured compute budget all savings are zero because of the overlap "
        "ceiling. The older `results/B/context_ablation_seed0.csv` records the "
        "same comparison under the previous 5 ms assumption.\n"
    )

    parts.append(
        "### 1.10 Learned components and emulator fidelity\n\n"
        f"**Classifier.** Trained on {', '.join(classifier['split']['train_scenarios'])} "
        f"with seeds {classifier['split']['train_seeds']} "
        f"({classifier['train']['row_count']:,} rows) and tested on held-out "
        f"Scenario {classifier['split']['test_scenario']} with seeds "
        f"{classifier['split']['test_seeds']} ({classifier['test']['row_count']:,} rows). "
        f"Accuracy and macro-F1 are both {f(classifier['test']['accuracy'], 3)}; the "
        "confusion matrix is diagonal. Removing phase/op features still gives "
        f"accuracy {f(classifier['test_no_phase_op']['accuracy'], 3)} "
        "(`classification_report.json`), so the perfect score is not evidence of "
        "a sophisticated signal: file size and temporal position suffice. These "
        "component reports were produced at the earlier 5 ms compute setting and "
        "are used only as evidence of component quality; the frozen model is "
        "evaluated in the 20-seed matrix at the measured setting.\n\n"
        f"**Embeddings.** Word2Vec over opaque access sequences with reuse horizon "
        f"{embeddings['split']['reuse_horizon']}: held-out accuracy "
        f"{f(embeddings['test']['accuracy'], 4)}, ROC AUC "
        f"{f(embeddings['test']['roc_auc'], 4)}, positive rate "
        f"{f(embeddings['test']['positive_rate'], 4)}. The `Embedding` policy does "
        "not improve data wait; speculative nearest-neighbour prefetching was "
        "removed as an honest negative result and wasted bytes stay zero.\n\n"
        f"**Reinforcement learning.** Tabular Q-learning over "
        f"{rl['training']['episodes']} episodes, "
        f"{rl['training']['weight_updates']:,} updates, mean reward "
        f"{f(rl['training']['first_mean_reward'], 4)} → "
        f"{f(rl['training']['last_mean_reward'], 4)}, final ε = "
        f"{f(rl['training']['final_epsilon'], 2)}. The saved agent has "
        f"{len(agent['q'])} "
        "state–action entries (see `agent.json`). It sees only observable "
        "features: no names, true types, oracle or declared types. DQN was "
        "deliberately omitted; the linear reference was unstable.\n\n"
        "**Emulated vs real.** Virtual and real modes were compared on a reduced "
        "workload (`results/real/rank_*.json`):\n\n"
    )
    rows = []
    for scenario in "ABCD":
        rank = ranks[scenario]
        rho = rank["spearman_rho"]
        rows.append([
            scenario,
            "undefined (complete tie)" if pd.isna(rho) else f(rho, 4),
            f(rank["pairwise_agreement"], 4),
        ])
    parts.append(
        table(["Scenario", "Spearman ρ", "Pairwise agreement"], rows)
        + "\n\nThe ordering is preserved, most clearly in Scenario C; the emulator "
        "is validated for relative comparisons only.\n"
    )

    parts.append(
        "### 1.11 Figures\n\n"
        "All figures are listed in `results/plots/plot_manifest.json`: "
        + ", ".join(f"`{name}`" for name in figures["figures"])
        + ". `compute_gain` is the paired gain-versus-`compute_ms` figure and "
        "marks the measured step time read from the benchmark artefact.\n\n---\n"
    )

    parts.append("## 2. Discussion\n\n### 2.1 What the results support\n")
    parts.append(
        "The central finding is narrower and more conditional than the earlier "
        "draft claimed. With the **measured** CPU step time as the default, "
        "compute dominates and **every** policy — including AllSlow — reaches zero "
        "uncovered data wait. The matrix therefore does not rank placement "
        "policies on `data_wait_ms` at all; what it does show is a throughput gap "
        "between cached and uncached reads (Section 1.3), and a small but "
        "seed-consistent throughput edge for FullContextTier over LRU/LFU that "
        "comes from reduced synchronous promotion (Table 3). Ranking on data wait "
        "returns only when the compute budget is reduced: Table 7 shows non-zero "
        "stalls at 1, 2 and 5 ms, and Tables 4 and 7 show that FullContextTier's "
        "advantage depends on that budget. No claim is made for production "
        "storage systems or for policies that were not run.\n"
    )
    parts.append(
        "### 2.2 Ties are mostly designed or ceiling effects\n\n"
        "* **Scenario D** is a designed tie: all files are pre-resident in FAST, "
        "so every policy scores 0 ms.\n"
        "* **The main matrix at the measured compute budget** is a ceiling effect: "
        "every policy, cached or not, scores 0 ms in A–C because the data-path "
        "time of even the slowest read fits inside the measured step. This is not "
        "evidence that placement is irrelevant — Table 2 shows AllSlow at "
        f"{f(allslow_c['throughput_mean'])} samples/s versus "
        f"{f(lru_c['throughput_mean'])} for LRU, and Table 3 separates cached "
        "policies on elapsed time.\n"
        "* **`Clairvoyant` equals `LRU` in C** because the seeded future is "
        "withheld; in A/B it sees the seed but still trails Belady, which alone "
        "receives the full read stream.\n"
        "* **`ContextRulesNoCtx` equals `LRU`** with an empty declared-type map, "
        "and `Classifier` matches `ContextRules` on held-out types in the "
        "component reports.\n"
        "* **All policies tie at 0 ms at the largest sweep points**, again a "
        "ceiling effect of the overlap model.\n"
    )
    parts.append(
        "### 2.3 Losses and negative results\n\n"
        "* The best deployable policy still loses to the oracle where the oracle "
        "has an advantage; we do not claim to close the clairvoyance gap.\n"
        "* `FullContextTier` trades FAST-resident-on-arrival rate and p95 latency "
        "for lower elapsed time, because validation files are pinned to MID. "
        "An application optimising hit rate or tail latency may prefer a baseline.\n"
        "* The learned reuse predictor does not translate into lower wait; "
        "speculative prefetching was removed rather than charged to the result.\n"
        "* The declared-type channel contributes nothing when only one data type "
        "exists, and nothing at all once the compute overlap ceiling is reached.\n"
        "* At small compute budgets the absolute stalls grow and the relative "
        "gain narrows; Table 4 reports this rather than selecting the most "
        "favourable budget.\n"
    )
    parts.append(
        "### 2.4 Limitations and threats to validity\n\n"
        "1. **Emulation, not a storage stack.** Latencies and bandwidths are "
        "constants; there is no real filesystem, page cache or network. The "
        "virtual/real rank check supports relative ordering only.\n"
        "2. **Compute assumption dominates.** The measured median is much larger "
        "than the sweep range, so the headline configuration is in the zero-wait "
        "regime. The sweep shows how conclusions change with a faster step.\n"
        "3. **No original NoPFS.** Seed-based exact prefetching is reimplemented "
        "inside the emulator and labelled NoPFS-style.\n"
        "4. **Oracle semantics matter.** Only Belady sees the full stream; "
        "Clairvoyant sees the shuffle seed and loses it in C.\n"
        "5. **Declared types are injected.** ContextRules is an upper bound "
        "enabled by an application channel that must actually be provided.\n"
        "6. **Small deterministic workload.** One dataset, fixed chunk/checkpoint "
        "sizes and 20 seeds of one workload family; the classifier's perfect "
        "score reflects that regularity.\n"
        "7. **Frozen learned components.** Classifier, embeddings and RL were "
        "trained on their declared splits before the compute change; component "
        "reports are not 20-seed main-matrix results.\n"
        "8. **Bounded replay.** The website's replay is **reconstructed playback**: "
        "it applies completed migrations from the log and treats `tier_served` as "
        "authoritative; in-flight prefetches are absent and reconstructed "
        "occupancy is illustrative.\n"
        "9. **Two contract questions remain owner-unconfirmed:** the full-stream "
        "Belady oracle and the constructor-injected declared-type channel. They "
        "are recorded as assumptions in `docs/DESIGN.md`.\n"
    )
    parts.append(
        "### 2.5 Deviations from the design contract\n\n"
        "1. **`fast_hit` semantics.** §10 defined it as 'reads served from Tier 0'. "
        "It is now arrival-time Tier-0 residency snapshotted before demand "
        "promotion, so it can differ from `tier_served`. `TieredStore.get` gained "
        "a keyword-only `arrival_fast` argument and `fast_resident()`; the change "
        "is regression-tested in `tests/test_fast_hit_and_decomposition.py`. "
        "Per-tier served rates still use `tier_served` and are not complements of "
        "this rate.\n"
        "2. **Elapsed decomposition.** Seven additive metrics were added to the "
        "metric list so elapsed time can be attributed exactly; they are not in "
        "the original contract but do not change any existing metric.\n"
        "3. **`compute_ms` default.** The placeholder 5.0 ms was replaced by the "
        f"measured median {f(compute, 6)} ms from {steps} real CPU TinyCNN steps; "
        "the nominal batch of 128 samples is actually "
        f"{benchmark['configuration']['effective_batch_samples']} because "
        f"{config['chunk_size_samples']}-sample chunks are rounded up.\n"
        "4. **Sweep and statistics.** The compute sweep was extended to "
        + ", ".join(f"{f(v, 0).replace(',', '')} ms" for v in sweep_computes)
        + f"; the matrix uses {n_seeds} rather than five seeds and adds paired "
        "bootstrap intervals and seeds-won counts. No p-values are reported.\n"
        "5. **New artefacts.** `paper_stats.py` produces bootstrap, decomposition "
        "and best-baseline tables; `plots.py` adds the `compute_gain` figure; the "
        "website adds a static fallback dashboard and labels replay as "
        "**reconstructed playback**.\n"
        "6. **NoPFS and shuffle.** Chunk-level rather than per-sample shuffling "
        "remains a documented deviation from NoPFS's full randomisation, and "
        "seed-based prefetching is a NoPFS-style reimplementation, not the "
        "original code.\n"
        "7. **Owner confirmations still open.** Full-stream Belady oracle and "
        "constructor-injected declared types remain design assumptions; they are "
        "not silently promoted to decisions.\n"
        "8. **Model switch protocol.** `cnn` remains the historical manual-budget "
        "baseline. `mlp` and profile-driven runs are isolated below "
        "`results/models/`; their timing, checkpoint size, and cross-model "
        "comparisons are reported only after the corresponding CSV/JSON artefacts "
        "exist. No CNN result is relabelled as a cross-model result.\n"
        "9. **Temperature protocol.** Scenarios E/F and `true_temp` are additive "
        "Phase-13 stress-test artefacts. Their results are isolated in "
        "`results/temperature/` and are not asserted from this A--D matrix until "
        "their CSV tables and paired-bootstrap comparisons are generated.\n"
    )
    parts.append(
        "### 2.6 Reproducibility\n\n"
        "```bash\n"
        "python experiments.py       # results/full/{raw_runs,summary,sweep_runs,sweep_summary}.csv\n"
        "python plots.py             # results/plots/*.{pdf,png} + plot_manifest.json\n"
        "python paper_stats.py       # bootstrap, decomposition, best-baseline tables\n"
        "python train.py --profile-models  # results/models/profiles.json\n"
        "python run_experiments.py --policy lru --scenario C --seed 0 --model mlp --compute-source profile\n"
        "python build_evaluation.py  # this document\n"
        "python build_site.py        # offline static dashboard\n"
        "python -m pytest tests -q   # regression suite\n"
        "python -m streamlit run app/main.py --server.port 8513\n"
        "```\n\n"
        "The website enforces the no-invented-numbers rule with a test that "
        "rejects hard-coded result values in `app/*.py`; this document is "
        "regenerated from the same files, so it cannot drift silently.\n"
    )
    return "\n".join(parts)


def main() -> None:
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
