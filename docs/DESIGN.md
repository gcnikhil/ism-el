# ContextTier — Design Contract (Phase 0)

Status: **design freeze candidate**. Everything below is the fixed contract from the
project brief, restated in implementation terms. Changing any item requires an
explicit decision from the project owner. No experimental numbers appear here.

---

## 1. Purpose and positioning

ContextTier is a **laptop-scale, CPU-only research prototype** of context-aware
dynamic data tiering for AI training. It serves two consumers:

1. a live demo website (Streamlit), and
2. a research paper (Evaluation + Discussion written from the owner's own runs).

### Base paper relationship

* Base paper: NoPFS, Dryden, Boehringer, Ben-Nun, Hoefler, *Clairvoyant Prefetching
  for Distributed Machine Learning I/O*, SC 2021, arXiv:2101.08734.
* NoPFS exploits **clairvoyance**: the shuffle seed makes the future sample-access
  order exactly predictable; a performance-model-driven policy caches and
  prefetches across a storage hierarchy.
* The NoPFS authors list **dynamic cache management** (samples migrating between
  caches) as future work.

### Our three extensions

1. **Context-awareness across ALL data types**: training chunks, checkpoints,
   metadata, validation, hot-sample cache — not only training-sample reads.
2. **Dynamic migration between tiers**.
3. **Learned classification, embeddings, and RL** for workloads whose access
   order is *not* seed-determined (hard-example oversampling), where clairvoyance
   does not apply.

### Honesty constraints

* We **do not run the original NoPFS code**. We re-implement its core idea
  (seed-based exact prefetching) inside our emulator and name it **"NoPFS-style"**.
* The implementation is an **emulation with scaled-down tiers**, not a production
  storage stack. The paper must say this explicitly.
* All results come from the owner's runs. Illustrative values are labelled
  *placeholder* and never presented as measurements.

---

## 2. Machine constraints (hard)

* RAM: 8 GB total; keep Python process memory **under ~3 GB**.
* No GPU. Everything runs on CPU in reasonable time.
* No Linux-only paths (no `/dev/shm`, no POSIX-only syscalls).
* Cross-platform paths via `pathlib.Path`.
* Real-mode timing depends on the actual host; the host is recorded at runtime
  and written into every results file (OS, CPU model, logical/physical cores,
  Python version, key library versions).

Machine fields left as placeholders in the brief (OS, CPU model/cores, Arrow
path) must be resolved before Phase 1/3. See `Open questions` at the end.

---

## 3. Data contract

* Input: HuggingFace Arrow files.
  * `[path]/data-00000-of-00001.arrow` → 50,000 train rows (~114 MB).
  * `[path]/data-00000-of-00001_1.arrow` → 10,000 test rows (~23 MB).
* Schema: `img: struct<bytes: binary, path: string>` (PNG, 32×32 RGB),
  `label: int64`.
* Read with `pyarrow.ipc.open_stream`, falling back to `pyarrow.ipc.open_file`.
* **First action of `prepare_data.py`**: print the schema and one decoded sample
  (image shape/dtype + label) to confirm the reader is correct.
* Memory-safe lazy loading: decode one row at a time into `uint8` arrays; never
  materialize all 60,000 decoded images at once.

---

## 4. Tier model (fixed)

* **Tier 2 (slow) is the ORIGIN.** It permanently holds every file.
* **Tier 0 (fast) and Tier 1 (mid) are CACHES** holding copies.
* Operations:
  * `promote` / `prefetch` = copy a file **up** a tier; consumes bandwidth;
    counted in `migration_bytes`.
  * `evict` = drop a cached copy; **free**.
* Reads are served from the **fastest tier holding a copy**.
  A plain `get` is **non-mutating**: it does not insert into a cache. Insertion
  only happens through an explicit `Promote`/`Prefetch` action returned by a
  policy. This is what makes migration-byte accounting exact.
* Writes (checkpoints, hot cache): the **policy chooses the landing tier** via
  `on_write`. A write landing in a cache tier is **dirty** and must be flushed to
  Tier 2 before it can be evicted. Flush bytes count as `migration_bytes`.
* Origin files are never dirty and never evicted.

### Bandwidth / timing rule

Each tier owns `busy_until_ms` on the simulated clock. A migration from source
`tier a` to destination `tier b` takes
`t_ms = size_bytes / (min(bandwidth_a, bandwidth_b) * 1e6) * 1000`, advances
`busy_until_ms` on **both** endpoints by `t_ms`, and records `migration_bytes +=
size_bytes`. A foreground read on tier `t` waits until
`max(now_ms, busy_until_ms[t])`, then adds `latency_ms[t] + size_bytes /
bandwidth[t]`. In-flight prefetches therefore delay a read only by their
*remaining* transfer time. Real mode realizes the same delays with scaled
`time.sleep`.

---

## 5. File units and opaque ids

* Training data = chunks of `chunk_size_samples` (default 50) decoded `uint8`
  samples, ~150 KB each, ~1,000 train chunks.
* Validation = ~200 chunks.
* Other files: checkpoints (`ckpt_size_mb`, configurable, default a few MB),
  one metadata file, hot-cache files.
* Internally every file has an **opaque integer id**.
* Human-readable names exist **only** in the dashboard and evaluation side
  tables.
* Policies that infer behavior (classifier, embeddings, RL) must **never** see
  names or true types.

### Truth channel (evaluation-only)

`true_type` is stored in a side table `file_id -> {chunk, checkpoint, metadata,
validation, hot_cache}` owned by the harness/Logger. It is appended to the trace
as an **EVAL-ONLY** column after the policy has already acted. It is never a
field of `AccessEvent`, `PolicyView`, or `TickContext`.

---

## 6. Workload generator

Per epoch:

1. Shuffle chunk order with a seeded RNG (chunk-level shuffling).
   * Config flag `sample_level_shuffle` enables full per-sample randomization;
     the paper must note this as a **deviation from NoPFS's full per-sample
     randomization**.
2. Per step:
   * read needed chunks,
   * read metadata,
   * run compute (`compute_ms`),
   * every `checkpoint_every_steps` steps write a checkpoint,
   * every `validation_every_steps` steps / at epoch end run a validation burst.

Phases: `warmup`, `train`, `validation`, `checkpoint`.

---

## 7. Non-seeded scenario (hard-example oversampling, Scenario C)

After each epoch, a subset of high-loss samples is oversampled in the next
epoch. Access order therefore depends on **training state**, not just the seed.

* **Real mode:** use real per-sample loss from the trained model.
* **Virtual mode:** no model exists, so use a documented **synthetic difficulty
  model**: per-sample difficulty ~ seeded lognormal, decaying each time the
  sample is seen.
* We check that real-mode traces show a **qualitatively similar reuse pattern**
  and report this in the paper. No quantitative equivalence is claimed.

---

## 8. Two execution modes, ONE core simulator

* **Virtual-time mode** (default for sweeps, RL, ablations):
  * no PyTorch, no real disk, no `sleep`;
  * files are `(id, size)` records;
  * simulated clock adds `latency + size/bandwidth` per access and `compute_ms`
    per batch;
  * background migrations consume bandwidth via per-tier `busy_until`;
  * deterministic, seconds per run.
* **Real mode** (headline runs + live demo):
  * tiny CNN (2 conv layers + FC, a few hundred K params), PyTorch on CPU,
    trained through `TieredStore`;
  * `time.sleep` realizes tier delays; wall-clock measurement;
  * optional 20,000-image subset. Accuracy is irrelevant; storage behavior is.
* **Validation:** on a few configurations, check virtual and real modes rank
  policies in the same order and report the agreement (e.g. Kendall tau / rank
  overlap). Do not claim more than was measured.

---

## 9. Policy interface and information barriers

Methods:

* `on_access(event, view) -> list[Action]`
* `on_write(event, view) -> landing_tier: int`
* `on_tick(ctx, view) -> list[Action]`
* `on_phase_change(ctx, view) -> None`

Actions: `Promote(id, tier)`, `Prefetch(id, tier)`, `Evict(id, tier)`, `Pin(id)`,
`Unpin(id)`.

`view` exposes **only**:

* tier occupancy and free space,
* which tiers hold a file,
* file size,
* the policy's own history.

`AccessEvent` has **no true-type field**. True types live in the evaluation-only
column.

**Oracle rule:** only `ClairvoyantPolicy` and `BeladyPolicy` receive the seeded
future order (`oracle`). No other policy may receive it.

**Context channel:** `ContextRulesPolicy` is an explicitly privileged
"oracle-context" upper bound. It receives an application-declared
`file_id -> declared_type` map **by constructor injection**, analogous to the
oracle, never through `AccessEvent`/`view`. Classifier/embeddings/RL policies
never receive either channel.

---

## 10. Metric definitions (exact)

| Metric | Definition |
|---|---|
| `data_wait_ms` | Per step: time the training loop is blocked waiting for data **after** overlap with compute and in-flight prefetches (GPU-idle proxy). Formally `max(0, data_ready_step - compute_available_step)`. |
| `throughput` | `total_samples_processed / total_elapsed_seconds` (simulated or wall). Includes checkpoint/validation time because it is *total* time. |
| `fast_hit_rate` | reads served from Tier 0 / all reads. Per-tier rates reported too. |
| `migration_bytes` | bytes copied between tiers: prefetch + promote + dirty flushes. Reads are not migrations. |
| `wasted_prefetch_bytes` | prefetched bytes evicted unused. |
| `latency` | per-read latency mean, p95, p99. |

---

## 11. Trace schema

CSV/Parquet columns, in this order:

```
ts, run_id, seed, scenario, policy, file_id, size_bytes, op, epoch, step,
phase, tier_served, latency_ms, wait_ms, fast_hit, true_type, true_temp
```

`file_id` is the opaque int. `true_type` and `true_temp` are EVAL-ONLY.

---

## 12. Config defaults (`config.yaml`, all changeable)

Scaled-down emulation — say so explicitly in the paper.

```yaml
tiers:
  - {id: 0, name: fast, capacity_mb: 25,   latency_ms: 0.01, bandwidth_mbps: 3000}
  - {id: 1, name: mid,  capacity_mb: 60,   latency_ms: 0.2,  bandwidth_mbps: 500}
  - {id: 2, name: slow, capacity_mb: null, latency_ms: 8.0,  bandwidth_mbps: 80}
chunk_size_samples: 50
batch_size: 128                # ~391 steps/epoch
checkpoint_every_steps: 50
validation_every_steps: null   # epoch-end by default
compute_ms: 166.98580000047514 # measured TinyCNN CPU-step median; manual baseline setting
ckpt_size_mb: 4.0
model:
  name: cnn                    # cnn | mlp
  compute_source: manual       # manual preserves baseline; profile reads profiles.json
  profile_path: results/models/profiles.json
seeds: [0, ..., 19]
sample_level_shuffle: false
mode: virtual                  # virtual | real
real_subset_images: null       # e.g. 20000
```

Tier 0 holds ~16% of the dataset. Tier 2 is unlimited.

---

## 13. Sanity invariants (Phase 2 acceptance)

```
data_wait:  AllFast <= Belady <= {LRU, LFU, ARC, Clairvoyant, ...} <= AllSlow
```

* `AllFast` ignores capacity and starts with all files in Tier 0 at zero
  migration cost — an idealized, unachievable upper bound.
* `AllSlow` serves everything from the origin — the lower bound on placement.
* `Belady` must be offline-optimal over the **full future read stream (all file
  types)** for the invariant to hold in scenarios B/C. This is stricter than the
  brief's "chunk-read stream" phrasing; see Open questions.
* LRU correctness tests (hit/miss order, eviction victim) must pass.

---

## 14. Phases (one per reply, stop after each)

| # | Deliverable |
|---|---|
| 0 | Restate design, assumptions, questions, folder tree, exact interfaces, this document. |
| 1 | `prepare_data.py`, `config.yaml`, `tiers.py`, `store.py`, `logger.py`, `requirements.txt`, pytest tests. Accept: capacity holds, migration bytes exact, logger schema exact. |
| 2 | Virtual simulator + workload + baselines (`AllSlow`, `AllFast`, `LRU`, `LFU`, `ARC`, `Clairvoyant`, `Belady`). `python run_experiments.py --policy lru --scenario A --seed 0` writes metrics + trace. Accept: invariants + LRU tests. |
| 3 | Real-mode tiny CNN through `TieredStore`; virtual-vs-real rank check. |
| 4 | `ContextRulesPolicy` oracle-context upper bound + ablation row. |
| 5 | `features.py` + `features`-based XGBoost classifier; split BY SEED; hold-out scenario (train A+B, test C); accuracy/macro-F1/confusion/importance; `ClassifierPolicy`. |
| 6 | `embeddings.py` Word2Vec + reuse predictor; `EmbeddingPrefetchPolicy`; Scenario C focus; honest Scenario A statement. |
| 7 | `rl_env.py` + `rl_agent.py` (linear/tabular Q-learning; DQN optional); `FullContextTierPolicy`; reward curves. |
| 8 | Full experiments, ≥5 seeds, mean±std; scenarios A–D; sweeps; all metrics to CSV. |
| 9 | `plots.py` publication-quality PDF+PNG with error bars. |
| 10 | Streamlit 4-page website + REPLAY MODE. |
| 11 | Help write Evaluation + Discussion from real tables, including ties/losses and limitations. |

---

## 15. Exact interfaces (declarations; concrete bodies land in their phase)

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Protocol, Sequence

# ---------------- tiers.py ----------------
@dataclass(frozen=True)
class TierSpec:
    tier_id: int                     # 0=fast, 1=mid, 2=slow/origin
    name: str                        # "fast" | "mid" | "slow"
    capacity_bytes: Optional[int]    # None => unlimited (origin only)
    latency_ms: float
    bandwidth_mbps: float

class Tier(Protocol):
    spec: TierSpec
    files: dict[int, int]            # opaque id -> size_bytes
    dirty: set[int]                  # written here, not yet flushed
    pinned: set[int]                 # eviction forbidden
    busy_until_ms: float             # virtual bandwidth serialization

    def used_bytes(self) -> int:
        """Sum of size_bytes currently held."""
    def free_bytes(self) -> Optional[int]:
        """Remaining capacity, or None if unlimited."""
    def holds(self, file_id: int) -> bool:
        """True if this tier holds a copy."""
    def can_fit(self, size_bytes: int) -> bool:
        """True if capacity allows size_bytes (or unlimited)."""

# ---------------- policies.py: actions & events ----------------
@dataclass(frozen=True)
class Action:
    """Base class; never instantiated directly."""

@dataclass(frozen=True)
class Promote(Action):
    file_id: int
    tier: int

@dataclass(frozen=True)
class Prefetch(Action):
    file_id: int
    tier: int

@dataclass(frozen=True)
class Evict(Action):
    file_id: int
    tier: int

@dataclass(frozen=True)
class Pin(Action):
    file_id: int

@dataclass(frozen=True)
class Unpin(Action):
    file_id: int

@dataclass(frozen=True)
class AccessEvent:
    file_id: int          # opaque int
    size_bytes: int
    op: str               # "read" | "write"
    epoch: int
    step: int
    phase: str            # warmup | train | validation | checkpoint
    # DELIBERATELY ABSENT: true_type, name, path

@dataclass(frozen=True)
class TickContext:
    now_ms: float
    epoch: int
    step: int
    phase: str

@dataclass(frozen=True)
class PhaseContext:
    now_ms: float
    epoch: int
    previous_phase: str
    new_phase: str

class PolicyView(Protocol):
    def occupancy(self) -> Mapping[int, int]:
        """tier_id -> bytes used."""
    def free_space(self) -> Mapping[int, Optional[int]]:
        """tier_id -> free bytes (None if unlimited)."""
    def holds(self, file_id: int) -> tuple[int, ...]:
        """Sorted tier ids holding a copy."""
    def size_of(self, file_id: int) -> int:
        """Opaque file size in bytes."""
    @property
    def history(self) -> dict[str, object]:
        """THIS POLICY'S private state only."""

class Oracle(Protocol):
    def upcoming(self, n: int, *, after: Optional[int] = None) -> list[int]:
        """Next n opaque ids in the future read stream."""
    def next_use(self, file_id: int) -> Optional[int]:
        """Index of next use, or None if absent from the stream."""
    def length(self) -> int:
        """Total number of future accesses."""

class Policy(Protocol):
    name: str
    def on_access(self, event: AccessEvent, view: PolicyView) -> list[Action]:
        """React to a read/write; return actions."""
    def on_write(self, event: AccessEvent, view: PolicyView) -> int:
        """Choose landing tier id for a write."""
    def on_tick(self, ctx: TickContext, view: PolicyView) -> list[Action]:
        """Periodic background decisions."""
    def on_phase_change(self, ctx: PhaseContext, view: PolicyView) -> None:
        """Phase transition hook."""

# ---------------- store.py ----------------
@dataclass(frozen=True)
class ReadResult:
    file_id: int
    tier_served: int
    latency_ms: float
    wait_ms: float
    fast_hit: bool

@dataclass(frozen=True)
class MigrationResult:
    file_id: int
    src_tier: Optional[int]
    dst_tier: int
    size_bytes: int
    scheduled_ms: float
    completed_ms: float

class Clock(Protocol):
    def now_ms(self) -> float:
        """Current simulated/wall time in ms."""
    def wait_until(self, t_ms: float) -> None:
        """Block until time t_ms (virtual: advance; real: time.sleep)."""
    def advance(self, dt_ms: float) -> None:
        """Move the clock forward by dt_ms."""

class TieredStore(Protocol):
    def get(self, file_id: int, event: AccessEvent) -> ReadResult:
        """Serve from fastest holding tier; non-mutating."""
    def put(self, event: AccessEvent, landing_tier: int, *, dirty: bool = True) -> None:
        """Write/overwrite a file, choosing a landing tier."""
    def locate(self, file_id: int) -> tuple[int, ...]:
        """Sorted tier ids holding a copy (origin always included)."""
    def promote(self, file_id: int, tier: int) -> Optional[MigrationResult]:
        """Copy up synchronously; None if illegal/no space."""
    def prefetch(self, file_id: int, tier: int) -> Optional[MigrationResult]:
        """Enqueue a bounded background copy up."""
    def evict(self, file_id: int, tier: int) -> bool:
        """Drop a cache copy; False if pinned/dirty/origin."""
    def flush(self, file_id: int) -> Optional[MigrationResult]:
        """Write a dirty cache copy back to the origin."""
    def enforce_capacity(self, tier: int) -> list[Action]:
        """Evict victims until capacity holds (never evicts pinned/dirty)."""
    def tick(self, ctx: TickContext) -> None:
        """Complete due background migrations."""
    @property
    def migration_bytes(self) -> int:
        """Cumulative bytes copied between tiers."""
    @property
    def wasted_prefetch_bytes(self) -> int:
        """Cumulative prefetched bytes evicted unused."""

# ---------------- logger.py ----------------
class Logger:
    TRACE_COLUMNS: tuple[str, ...] = (
        "ts", "run_id", "seed", "scenario", "policy", "file_id", "size_bytes",
        "op", "epoch", "step", "phase", "tier_served", "latency_ms", "wait_ms",
        "fast_hit", "true_type",
    )

    # true_type is injected by the HARNESS from the eval-only side table.
    # No Policy object ever supplies it.
    def log_access(self, *, ts: float, run_id: str, seed: int, scenario: str,
                   policy: str, file_id: int, size_bytes: int, op: str,
                   epoch: int, step: int, phase: str, tier_served: int,
                   latency_ms: float, wait_ms: float, fast_hit: bool,
                   true_type: str) -> None:
        """Append one row conforming to TRACE_COLUMNS."""

    def log_migration(self, *, ts: float, run_id: str, seed: int,
                      scenario: str, policy: str, file_id: int,
                      size_bytes: int, src_tier: Optional[int], dst_tier: int,
                      kind: str) -> None:
        """Append a migration row to the auxiliary migration log."""

    def set_metric(self, name: str, value: float, **tags: object) -> None:
        """Record a scalar metric with run tags."""

    def write_trace(self, path: Path, fmt: str = "parquet") -> None:
        """Write the trace as CSV or Parquet."""

    def write_metrics(self, path: Path) -> None:
        """Write the metrics table as CSV."""
```

---

## 16. Assumptions made in this restatement

1. A plain read never auto-populates a cache; caching is policy-driven via
   actions. This is required for exact migration-byte accounting.
2. `true_type` is joined by the harness/Logger, never passed to a policy.
3. `ContextRulesPolicy` receives declared types via constructor injection
   (privileged channel), like the oracle, not via `AccessEvent`/`view`.
4. `Belady` sees the full future read stream so the Phase-2 invariant can hold
   in B/C (stricter than "chunk-read stream").
5. `compute_ms` is the measured median of 200 pure-compute TinyCNN CPU steps
   on the evaluation machine (`results/real/step_time_benchmark.json`); it is
   still configurable for sensitivity sweeps.
6. `AllFast` is an idealized bound with all files pre-resident in Tier 0 and
   zero migration bytes; it is reported as unachievable.
7. Real-mode hard-example selection uses actual per-sample loss from the
   selected registered model; virtual mode uses the seeded lognormal decay
   model.
8. Tiers are named `fast`, `mid`, `slow` and numbered `0`, `1`, `2`.

---

## 17. Model switch (Phase 12)

`train.py` owns an extensible registry: `cnn` is the historical `TinyCNN` and
`mlp` is `Flatten -> Linear(3072, 256) -> ReLU -> Linear(256, 10)`.  Model
selection is available through `config.model.name`, `--model`, real training,
the rank-check entry point, and the dashboard selector. Unknown names fail with
the registry's available choices.

`python train.py --profile-models` writes
`results/models/profiles.json`. For each model it records parameter count,
median and p95 CPU step time (128 samples; 20 warmups; 200 measured steps),
serialized model-plus-optimizer checkpoint bytes, and observed process peak
RAM. `compute_source: manual` preserves the historical CNN matrix exactly;
`compute_source: profile` copies the selected profile's median step time into
the virtual overlap budget and its checkpoint bytes into `ckpt_size_mb` for a
new model-specific run. Non-baseline model CLI runs are isolated below
`results/models/<model>/`; no existing `results/full/` input is overwritten.

Model-specific classifier, embedding, and RL transfer/retraining comparisons
are recorded only after their own traces and CSVs exist. They must not reuse a
number from the CNN matrix as an MLP result.

---

## 18. Hot / warm / cold temperature (Phase 13)

Scenarios E and F are synthetic stress tests for importance sampling,
curriculum learning, and hard-example mining. E samples a seeded hot/warm/cold
partition non-uniformly; F rotates that partition every
`temperature.drift_every_epochs`. Their access orders are treated as
state-dependent: Clairvoyant receives no seeded oracle in E/F, while Belady
retains the offline full-stream bound.

`true_temp` is appended by the logging harness only. It is absent from
`AccessEvent`, `PolicyView`, policy constructors, opaque ids, and learned
features. `TemperaturePolicy` derives EWMA access-rate buckets from its own
history with rolling quantile thresholds and hysteresis; hot requests target
FAST, warm requests target MID, and cold requests are evicted from caches.
`TemperatureLearnedPolicy` may replace the EWMA bucket with an observable reuse
predictor probability, but receives no truth channel.

Temperature runs and derived tables belong under `results/temperature/`; the
preserved A--D tables under `results/full/` are never overwritten.

---

## 19. Open questions (max 5)

1. What is the actual absolute path to the two `.arrow` files? The brief shows
   `[path]`. Needed for Phase 1.
2. The environment reports Windows (`win32`). Confirm Windows is the target for
   real-mode timings, and provide CPU model + physical cores — or approve
   runtime auto-detection written into every results file.
3. Confirm Belady receives the **full future read stream** (all file types),
   not only chunks, so `AllFast <= Belady <= others <= AllSlow` holds in B/C.
4. Confirm the `ContextRulesPolicy` information channel: constructor-injected
   declared-type map only, never in `AccessEvent`/`view`.
5. CPU model/core count is recorded by the model-profiler environment rather
   than assumed in the paper narrative.
