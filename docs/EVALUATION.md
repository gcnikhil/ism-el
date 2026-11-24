# ContextTier — Evaluation and Discussion (paper draft)

> **Provenance.** Every number in this document was generated from the
> project's own result files by `python build_evaluation.py`; none is
> illustrative. Sources: `results/full/*.csv`, `results/full/*.json`,
> `results/real/step_time_benchmark.json`, `results/real/rank_*.json`,
> `results/classifier/*`, `results/embeddings/*`, `results/rl/*`,
> `results/plots/plot_manifest.json` and `config.yaml`. To regenerate:
> `python experiments.py`, `python plots.py`, `python paper_stats.py`,
> `python build_evaluation.py`.

---

## 1. Evaluation

### 1.1 Experimental setup

This is a **laptop-scale, CPU-only emulation**, not a production storage stack. The tier model in `config.yaml` is:

| Tier | Capacity (MB) | Latency (ms) | Bandwidth (MB/s) |
|---|---|---|---|
| Tier 0 (fast) | 25.00 | 0.01 | 3,000.00 |
| Tier 1 (mid) | 60.00 | 0.20 | 500.00 |
| Tier 2 (slow) | unbounded | 8.00 | 80.00 |

Runs use 3 virtual epochs, batch size 128, checkpoints every 50 steps and compute_ms = 166.985800 ms. This is **not a placeholder**: it is the median of 200 measured pure-compute TinyCNN steps on this machine (`results/real/step_time_benchmark.json`: mean 167.57 ms, p95 190.85 ms; 3.11.9, torch 2.15.0+cpu, 8 logical CPUs, 2 torch threads; model input [150, 3, 32, 32], so the nominal batch 128 becomes 150 samples because chunks are read in batches of 50). No storage I/O or sleep was included in that timing. The main matrix and sweeps use 20 seeds (listed in the manifest), and learned models are frozen on their declared train splits before evaluation.


Four scenarios isolate different sources of context:

| Scenario | Workload |
|---|---|
| A | Shuffled training chunks + per-step metadata. |
| B | A + checkpoints + validation reads. |
| C | B + hard-example oversampling; access order depends on training state. |
| D | B with every file pre-resident in FAST (designed tie). |

Policies are separated by the information they may use. `AllSlow`/`AllFast` are capacity-bound controls; `LRU`/`LFU`/`ARC` are observable baselines; `Classifier`/`Embedding`/`FullContextTier` are learned policies using only observable features; `ContextRules` receives constructor-injected declared types (declared-type upper bound, with `ContextRulesNoCtx` as its empty-map ablation); `Clairvoyant` uses only the shuffle seed (withheld in Scenario C); `Belady` receives the full read stream (oracle upper bound). We re-implement seed-based exact prefetching inside the emulator and call it *NoPFS-style*; we do not run the original NoPFS code.

### 1.2 Metrics and statistics

The primary metric is `data_wait_ms`, the compute-overlap-uncovered stall per training step: `max(0, sum(read latency + read wait + synchronous promotion time) - compute_ms)`, summed over train/warmup steps. Validation and checkpoint time count towards `throughput` (total samples / total elapsed seconds) but not towards data wait. Secondary metrics are `fast_hit_rate`, per-tier served rates, `migration_bytes`, `wasted_prefetch_bytes` and mean/p95/p99 read latency. **FAST hit means resident in Tier 0 when the read arrived, before any demand promotion**: the harness snapshots residency before calling `on_access`, so a read can be served by Tier 0 (`tier_served = FAST`) while `fast_hit = false`. Consequently `fast_hit_rate` is not the complement of the MID/SLOW service rates, which are computed from `tier_served`.

`elapsed_ms` is decomposed exactly into seven additive buckets — synchronous promotion, training reads, validation reads, other reads, checkpoint writes, training compute and the final flush — with no residual (regression-tested to 1e-9 ms). Reported aggregates are mean ± sample standard deviation over the seeds. Paired comparisons use the same seed for both policies and a percentile bootstrap of the mean difference (10,000 resamples, bootstrap seed 0); we report the mean, 95% interval and seeds won. **No p-values are reported**: with 20 paired seeds we prefer the effect size and its interval to a binary significance claim.

### 1.3 Main matrix (all policies, all scenarios)

**Table 1 — Data wait in ms, mean ± std over 20 seeds** (source: `results/full/summary.csv`)


| Policy | Class | A | B | C | D |
|---|---|---|---|---|---|
| AllSlow | capacity bound | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| AllFast | capacity bound | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| LRU | baseline | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| LFU | baseline | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| ARC | baseline | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Clairvoyant | oracle | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Belady | oracle | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| ContextRulesNoCtx | declared-type ablation | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| ContextRules | declared-type bound | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Classifier | learned | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Embedding | learned | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| FullContextTier | learned | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |


At the measured compute budget (166.985800 ms per step), the zero-wait ceiling binds for **every** policy, including AllSlow: the largest `data_wait_ms` mean anywhere in the matrix is 0.00 ms, so Table 1 cannot rank policies at all on this metric. AllSlow pays for its uncached reads in **throughput** instead: 605.07 vs 732.72 samples/s in Scenario C (9.88 ms p95 read latency, because every read is served from SLOW). Section 1.6 therefore reports the paired comparison both on this matrix (where data wait is a universal tie) and at each point of the compute sweep, where the overlap model is active and policies separate. Section 1.5 identifies which elapsed-time bucket drives the throughput differences.

### 1.4 Secondary metrics in Scenario C

**Table 2 — Scenario C secondary metrics, means over 20 seeds** (source: `results/full/summary.csv`)


| Policy | Data wait % | Throughput (samples/s) | FAST resident on arrival | MID served | SLOW served | Migration (MB) | p95 latency (ms) |
|---|---|---|---|---|---|---|---|
| AllSlow | 0.00 | 605.07 | 0.000 | 0.000 | 1.000 | 0.00 | 9.8799 |
| AllFast | 0.00 | 761.34 | 1.000 | 0.000 | 0.000 | 83.89 | 0.0601 |
| LRU | 0.00 | 732.72 | 0.258 | 0.000 | 0.000 | 604.07 | 0.0601 |
| LFU | 0.00 | 732.52 | 0.253 | 0.000 | 0.000 | 608.49 | 0.0601 |
| ARC | 0.00 | 732.55 | 0.253 | 0.000 | 0.000 | 607.98 | 0.0601 |
| Clairvoyant | 0.00 | 732.72 | 0.258 | 0.000 | 0.000 | 604.07 | 0.0601 |
| Belady | 0.00 | 735.55 | 0.337 | 0.000 | 0.000 | 562.32 | 0.0601 |
| ContextRulesNoCtx | 0.00 | 732.72 | 0.258 | 0.000 | 0.000 | 604.07 | 0.0601 |
| ContextRules | 0.00 | 734.72 | 0.265 | 0.113 | 0.000 | 537.64 | 0.5008 |
| Classifier | 0.00 | 734.72 | 0.265 | 0.113 | 0.000 | 537.64 | 0.5008 |
| Embedding | 0.00 | 732.72 | 0.259 | 0.000 | 0.000 | 604.06 | 0.0601 |
| FullContextTier | 0.00 | 735.21 | 0.265 | 0.119 | 0.000 | 537.64 | 0.5008 |

`wasted_prefetch_bytes` is zero for every policy: speculative nearest-neighbour prefetching was removed after it failed to help, so no unused bytes are charged. FAST-resident-on-arrival and served-tier rates answer different questions under the corrected `fast_hit` definition; their sum is therefore not constrained to one.

### 1.5 Elapsed-time decomposition in Scenarios B and C

**Table 3 — Mean elapsed-time components (ms) for every policy, 20 seeds** (source: `results/full/elapsed_decomposition.csv`; component sums equal `elapsed_ms` exactly)


| Scen. | Policy | Data wait | Promotion | Train reads | Validation reads | Checkpoint writes | Train compute | Final flush | Elapsed |
|---|---|---|---|---|---|---|---|---|---|
| B | AllSlow | 0.00 | 0.00 | 44,162.43 | 5,927.96 | 1,243.20 | 192,367.64 | 0.00 | 243,701.23 |
| B | AllFast | 0.00 | 0.00 | 223.62 | 36.08 | 28.88 | 192,367.64 | 1,024.00 | 193,680.23 |
| B | LRU | 0.00 | 7,547.58 | 223.62 | 36.08 | 1,243.20 | 192,367.64 | 0.00 | 201,418.12 |
| B | LFU | 0.00 | 7,288.60 | 223.62 | 36.08 | 1,243.20 | 192,367.64 | 0.00 | 201,159.14 |
| B | ARC | 0.00 | 7,359.48 | 223.62 | 36.08 | 1,243.20 | 192,367.64 | 0.00 | 201,230.03 |
| B | Clairvoyant | 0.00 | 7,251.36 | 223.62 | 36.08 | 1,243.20 | 192,367.64 | 0.00 | 201,121.91 |
| B | Belady | 0.00 | 6,294.49 | 223.62 | 36.08 | 1,243.20 | 192,367.64 | 0.00 | 200,165.04 |
| B | ContextRulesNoCtx | 0.00 | 7,547.58 | 223.62 | 36.08 | 1,243.20 | 192,367.64 | 0.00 | 201,418.12 |
| B | ContextRules | 0.00 | 6,732.80 | 223.62 | 300.47 | 1,243.20 | 192,367.64 | 0.00 | 200,867.74 |
| B | Classifier | 0.00 | 6,732.80 | 223.62 | 300.47 | 1,243.20 | 192,367.64 | 0.00 | 200,867.74 |
| B | Embedding | 0.00 | 7,547.18 | 223.62 | 36.08 | 1,243.20 | 192,367.64 | 0.00 | 201,417.73 |
| B | FullContextTier | 0.00 | 6,580.13 | 236.84 | 300.47 | 1,243.20 | 192,367.64 | 0.00 | 200,728.29 |
| C | AllSlow | 0.00 | 0.00 | 44,162.43 | 5,927.96 | 1,243.20 | 192,367.64 | 0.00 | 243,701.23 |
| C | AllFast | 0.00 | 0.00 | 223.62 | 36.08 | 28.88 | 192,367.64 | 1,024.00 | 193,680.23 |
| C | LRU | 0.00 | 7,373.87 | 223.62 | 36.08 | 1,243.20 | 192,367.64 | 0.00 | 201,244.42 |
| C | LFU | 0.00 | 7,427.81 | 223.62 | 36.08 | 1,243.20 | 192,367.64 | 0.00 | 201,298.35 |
| C | ARC | 0.00 | 7,421.63 | 223.62 | 36.08 | 1,243.20 | 192,367.64 | 0.00 | 201,292.18 |
| C | Clairvoyant | 0.00 | 7,373.87 | 223.62 | 36.08 | 1,243.20 | 192,367.64 | 0.00 | 201,244.42 |
| C | Belady | 0.00 | 6,601.20 | 223.62 | 36.08 | 1,243.20 | 192,367.64 | 0.00 | 200,471.75 |
| C | ContextRulesNoCtx | 0.00 | 7,373.87 | 223.62 | 36.08 | 1,243.20 | 192,367.64 | 0.00 | 201,244.42 |
| C | ContextRules | 0.00 | 6,563.04 | 223.62 | 300.47 | 1,243.20 | 192,367.64 | 0.00 | 200,697.98 |
| C | Classifier | 0.00 | 6,563.04 | 223.62 | 300.47 | 1,243.20 | 192,367.64 | 0.00 | 200,697.98 |
| C | Embedding | 0.00 | 7,373.76 | 223.62 | 36.08 | 1,243.20 | 192,367.64 | 0.00 | 201,244.30 |
| C | FullContextTier | 0.00 | 6,415.37 | 236.84 | 300.47 | 1,243.20 | 192,367.64 | 0.00 | 200,563.53 |


**Which bucket explains the throughput gain over LRU?** At this compute budget, training data wait is zero for both policies, so it cannot explain the difference. In Scenario B, FullContextTier saves 967.45 ms of synchronous promotion time but spends 264.39 ms more on validation reads, giving a net elapsed-time saving of 689.84 ms and a throughput difference of 2.52 samples/s (734.60 vs 732.09). In Scenario C the corresponding numbers are 958.50 ms saved on promotion, 264.39 ms extra validation time, 680.88 ms net elapsed saving and 2.49 samples/s (735.21 vs 732.72). Checkpoint-write time is identical between the two policies in both scenarios. The gain is therefore a **training data-path effect (less synchronous promotion)**, partially offset by validation reads, not an uncovered-stall effect. At smaller compute budgets (1–10 ms) the uncovered data wait becomes non-zero and adds to the same promotion advantage; see Section 1.8.

### 1.6 Paired comparison with 20 seeds (no p-values)

**Table 4 — FullContextTier paired advantage, mean [95% bootstrap CI], seeds won** (source: `results/full/bootstrap_comparison.csv`; positive data-wait difference means the baseline stalled more)


| Setting | Baseline | Data wait saved (ms) [95% CI] | Seeds won |
|---|---|---|---|
| matrix | LRU | 0.00 [0.00, 0.00] | 0/20 |
| matrix | LFU | 0.00 [0.00, 0.00] | 0/20 |
| 1 ms | LRU | 190.09 [184.68, 195.66] | 20/20 |
| 1 ms | LFU | nan [nan, nan] | 0/0 |
| 2 ms | LRU | 180.73 [175.30, 186.29] | 20/20 |
| 2 ms | LFU | nan [nan, nan] | 0/0 |
| 5 ms | LRU | 57.68 [55.48, 59.88] | 20/20 |
| 5 ms | LFU | nan [nan, nan] | 0/0 |
| 10 ms | LRU | 0.00 [0.00, 0.00] | 0/20 |
| 10 ms | LFU | nan [nan, nan] | 0/0 |
| 20 ms | LRU | 0.00 [0.00, 0.00] | 0/20 |
| 20 ms | LFU | nan [nan, nan] | 0/0 |
| 50 ms | LRU | 0.00 [0.00, 0.00] | 0/20 |
| 50 ms | LFU | nan [nan, nan] | 0/0 |


On the main matrix the paired difference versus LRU is 0.00 ms [0.00, 0.00] with 0/20 seeds won; versus LFU it is 0.00 ms [0.00, 0.00] with 0/20 seeds won. These exact ties follow from the zero-wait ceiling. At the 5 ms sweep point (the old placeholder budget), the same comparison is informative; Table 4 makes the dependence on the compute assumption explicit. Intervals are percentile bootstrap intervals of the paired mean; no p-values are reported.

### 1.7 Scenario C against the best baseline

**Table 5 — FullContextTier vs the best deployable observable baseline in C** (source: `results/full/scenario_c_best_baseline.csv`)


| Best baseline | Baseline wait (ms) | FullContextTier wait (ms) | Saved (ms) | Saved (%) | Throughput gain (samples/s) |
|---|---|---|---|---|---|
| ARC | 0.00 | 0.00 | 0.00 | 0.00 | 2.66 |

The best baseline among LRU, LFU, ARC is **ARC**, selected by mean `data_wait_ms` over 20 seeds. AllSlow, AllFast are excluded as capacity-bound controls, not deployable cache policies. The paired bootstrap comparisons against LRU and LFU in Table 4 cover the other baselines directly.

### 1.8 Sensitivity sweeps

Learned models are trained once on their declared splits and reused at every sweep point; these are transfer-under-shift measurements, not per-setting retraining. Both sweeps use Scenario C.

**Table 6 — Fast-tier capacity sweep, `data_wait_ms` mean ± std** (source: `results/full/sweep_summary.csv`)


| FAST capacity | LRU | ContextRules | FullContextTier |
|---|---|---|---|
| 10 MB | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 25 MB | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 50 MB | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |


**Table 7 — Compute-per-step sweep, `data_wait_ms` mean ± std** (source: `results/full/sweep_summary.csv`)


| compute_ms | LRU | ContextRules | FullContextTier |
|---|---|---|---|
| 1 ms | 5,202.47 ± 24.50 | 5,144.07 ± 26.17 | 5,012.38 ± 25.65 |
| 2 ms | 4,050.87 ± 24.48 | 3,992.57 ± 26.13 | 3,870.15 ± 25.85 |
| 5 ms | 810.33 ± 9.54 | 788.34 ± 9.84 | 752.65 ± 9.72 |
| 10 ms | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 20 ms | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 50 ms | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

The measured CPU step time is far above the largest sweep point, so the main matrix sits at the zero-wait ceiling. `compute_gain` in `results/plots/plot_manifest.json` plots the paired gain against this sweep and marks the measured median as a vertical reference. Scenario D is excluded from the capacity sweep because the simulator forces its FAST tier large enough to hold every file.

### 1.9 Context ablation

Paired per-seed data-wait savings from the declared-type channel (`results/full/raw_runs.csv`):


| Scenario | Mean saved (ms) | Std | Min | Max | Seeds improved |
|---|---|---|---|---|---|
| A | 0.00 | 0.00 | 0.00 | 0.00 | 0/20 |
| B | 0.00 | 0.00 | 0.00 | 0.00 | 0/20 |
| C | 0.00 | 0.00 | 0.00 | 0.00 | 0/20 |
| D | 0.00 | 0.00 | 0.00 | 0.00 | 0/20 |

The channel only matters once non-chunk data types coexist; at the measured compute budget all savings are zero because of the overlap ceiling. The older `results/B/context_ablation_seed0.csv` records the same comparison under the previous 5 ms assumption.

### 1.10 Learned components and emulator fidelity

**Classifier.** Trained on A, B with seeds [0, 1, 2] (30,015 rows) and tested on held-out Scenario C with seeds [3, 4] (10,626 rows). Accuracy and macro-F1 are both 1.000; the confusion matrix is diagonal. Removing phase/op features still gives accuracy 1.000 (`classification_report.json`), so the perfect score is not evidence of a sophisticated signal: file size and temporal position suffice. These component reports were produced at the earlier 5 ms compute setting and are used only as evidence of component quality; the frozen model is evaluated in the 20-seed matrix at the measured setting.

**Embeddings.** Word2Vec over opaque access sequences with reuse horizon 5: held-out accuracy 0.9992, ROC AUC 0.9992, positive rate 0.2207. The `Embedding` policy does not improve data wait; speculative nearest-neighbour prefetching was removed as an honest negative result and wasted bytes stay zero.

**Reinforcement learning.** Tabular Q-learning over 12 episodes, 63,756 updates, mean reward 0.8028 → 0.9701, final ε = 0.02. The saved agent has 42 state–action entries (see `agent.json`). It sees only observable features: no names, true types, oracle or declared types. DQN was deliberately omitted; the linear reference was unstable.

**Emulated vs real.** Virtual and real modes were compared on a reduced workload (`results/real/rank_*.json`):


| Scenario | Spearman ρ | Pairwise agreement |
|---|---|---|
| A | 0.8000 | 0.8333 |
| B | 0.8000 | 0.8333 |
| C | 0.9487 | 1.0000 |
| D | undefined (complete tie) | 1.0000 |

The ordering is preserved, most clearly in Scenario C; the emulator is validated for relative comparisons only.

### 1.11 Figures

All figures are listed in `results/plots/plot_manifest.json`: `data_wait_all`, `data_wait_deployable`, `scenario_c_focus`, `capacity_sweep`, `compute_sweep`, `compute_gain`, `context_ablation`, `latency_percentiles`. `compute_gain` is the paired gain-versus-`compute_ms` figure and marks the measured step time read from the benchmark artefact.

---

## 2. Discussion

### 2.1 What the results support

The central finding is narrower and more conditional than the earlier draft claimed. With the **measured** CPU step time as the default, compute dominates and **every** policy — including AllSlow — reaches zero uncovered data wait. The matrix therefore does not rank placement policies on `data_wait_ms` at all; what it does show is a throughput gap between cached and uncached reads (Section 1.3), and a small but seed-consistent throughput edge for FullContextTier over LRU/LFU that comes from reduced synchronous promotion (Table 3). Ranking on data wait returns only when the compute budget is reduced: Table 7 shows non-zero stalls at 1, 2 and 5 ms, and Tables 4 and 7 show that FullContextTier's advantage depends on that budget. No claim is made for production storage systems or for policies that were not run.

### 2.2 Ties are mostly designed or ceiling effects

* **Scenario D** is a designed tie: all files are pre-resident in FAST, so every policy scores 0 ms.
* **The main matrix at the measured compute budget** is a ceiling effect: every policy, cached or not, scores 0 ms in A–C because the data-path time of even the slowest read fits inside the measured step. This is not evidence that placement is irrelevant — Table 2 shows AllSlow at 605.07 samples/s versus 732.72 for LRU, and Table 3 separates cached policies on elapsed time.
* **`Clairvoyant` equals `LRU` in C** because the seeded future is withheld; in A/B it sees the seed but still trails Belady, which alone receives the full read stream.
* **`ContextRulesNoCtx` equals `LRU`** with an empty declared-type map, and `Classifier` matches `ContextRules` on held-out types in the component reports.
* **All policies tie at 0 ms at the largest sweep points**, again a ceiling effect of the overlap model.

### 2.3 Losses and negative results

* The best deployable policy still loses to the oracle where the oracle has an advantage; we do not claim to close the clairvoyance gap.
* `FullContextTier` trades FAST-resident-on-arrival rate and p95 latency for lower elapsed time, because validation files are pinned to MID. An application optimising hit rate or tail latency may prefer a baseline.
* The learned reuse predictor does not translate into lower wait; speculative prefetching was removed rather than charged to the result.
* The declared-type channel contributes nothing when only one data type exists, and nothing at all once the compute overlap ceiling is reached.
* At small compute budgets the absolute stalls grow and the relative gain narrows; Table 4 reports this rather than selecting the most favourable budget.

### 2.4 Limitations and threats to validity

1. **Emulation, not a storage stack.** Latencies and bandwidths are constants; there is no real filesystem, page cache or network. The virtual/real rank check supports relative ordering only.
2. **Compute assumption dominates.** The measured median is much larger than the sweep range, so the headline configuration is in the zero-wait regime. The sweep shows how conclusions change with a faster step.
3. **No original NoPFS.** Seed-based exact prefetching is reimplemented inside the emulator and labelled NoPFS-style.
4. **Oracle semantics matter.** Only Belady sees the full stream; Clairvoyant sees the shuffle seed and loses it in C.
5. **Declared types are injected.** ContextRules is an upper bound enabled by an application channel that must actually be provided.
6. **Small deterministic workload.** One dataset, fixed chunk/checkpoint sizes and 20 seeds of one workload family; the classifier's perfect score reflects that regularity.
7. **Frozen learned components.** Classifier, embeddings and RL were trained on their declared splits before the compute change; component reports are not 20-seed main-matrix results.
8. **Bounded replay.** The website's replay is **reconstructed playback**: it applies completed migrations from the log and treats `tier_served` as authoritative; in-flight prefetches are absent and reconstructed occupancy is illustrative.
9. **Two contract questions remain owner-unconfirmed:** the full-stream Belady oracle and the constructor-injected declared-type channel. They are recorded as assumptions in `docs/DESIGN.md`.

### 2.5 Deviations from the design contract

1. **`fast_hit` semantics.** §10 defined it as 'reads served from Tier 0'. It is now arrival-time Tier-0 residency snapshotted before demand promotion, so it can differ from `tier_served`. `TieredStore.get` gained a keyword-only `arrival_fast` argument and `fast_resident()`; the change is regression-tested in `tests/test_fast_hit_and_decomposition.py`. Per-tier served rates still use `tier_served` and are not complements of this rate.
2. **Elapsed decomposition.** Seven additive metrics were added to the metric list so elapsed time can be attributed exactly; they are not in the original contract but do not change any existing metric.
3. **`compute_ms` default.** The placeholder 5.0 ms was replaced by the measured median 166.985800 ms from 200 real CPU TinyCNN steps; the nominal batch of 128 samples is actually 150 because 50-sample chunks are rounded up.
4. **Sweep and statistics.** The compute sweep was extended to 1 ms, 2 ms, 5 ms, 10 ms, 20 ms, 50 ms; the matrix uses 20 rather than five seeds and adds paired bootstrap intervals and seeds-won counts. No p-values are reported.
5. **New artefacts.** `paper_stats.py` produces bootstrap, decomposition and best-baseline tables; `plots.py` adds the `compute_gain` figure; the website adds a static fallback dashboard and labels replay as **reconstructed playback**.
6. **NoPFS and shuffle.** Chunk-level rather than per-sample shuffling remains a documented deviation from NoPFS's full randomisation, and seed-based prefetching is a NoPFS-style reimplementation, not the original code.
7. **Owner confirmations still open.** Full-stream Belady oracle and constructor-injected declared types remain design assumptions; they are not silently promoted to decisions.
8. **Model switch protocol.** `cnn` remains the historical manual-budget baseline. `mlp` and profile-driven runs are isolated below `results/models/`; their timing, checkpoint size, and cross-model comparisons are reported only after the corresponding CSV/JSON artefacts exist. No CNN result is relabelled as a cross-model result.
9. **Temperature protocol.** Scenarios E/F and `true_temp` are additive Phase-13 stress-test artefacts. Their results are isolated in `results/temperature/` and are not asserted from this A--D matrix until their CSV tables and paired-bootstrap comparisons are generated.

### 2.6 Reproducibility

```bash
python experiments.py       # results/full/{raw_runs,summary,sweep_runs,sweep_summary}.csv
python plots.py             # results/plots/*.{pdf,png} + plot_manifest.json
python paper_stats.py       # bootstrap, decomposition, best-baseline tables
python train.py --profile-models  # results/models/profiles.json
python run_experiments.py --policy lru --scenario C --seed 0 --model mlp --compute-source profile
python build_evaluation.py  # this document
python build_site.py        # offline static dashboard
python -m pytest tests -q   # regression suite
python -m streamlit run app/main.py --server.port 8513
```

The website enforces the no-invented-numbers rule with a test that rejects hard-coded result values in `app/*.py`; this document is regenerated from the same files, so it cannot drift silently.
