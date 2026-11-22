"""Policy-side interfaces: actions, events and information-barrier protocols.

No policy implementation exists yet; Phase 2 adds concrete policies. Keeping
these declarations here gives the store and tests a single source of truth.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Optional, Protocol

import numpy as np

from features import FEATURE_NAMES, FeatureExtractor, feature_vector
from tiers import FAST, MID, SLOW


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Action:
    """Base class for all store actions. Never instantiate directly."""


@dataclass(frozen=True)
class Promote(Action):
    """Copy ``file_id`` up into ``tier`` (synchronous, counted as migration)."""

    file_id: int
    tier: int


@dataclass(frozen=True)
class Prefetch(Action):
    """Copy ``file_id`` up into ``tier`` (background, counted as migration)."""

    file_id: int
    tier: int


@dataclass(frozen=True)
class Evict(Action):
    """Drop the cached copy of ``file_id`` from ``tier`` (free)."""

    file_id: int
    tier: int


@dataclass(frozen=True)
class Pin(Action):
    """Forbid eviction of ``file_id``."""

    file_id: int


@dataclass(frozen=True)
class Unpin(Action):
    """Allow eviction of ``file_id`` again."""

    file_id: int


# --------------------------------------------------------------------------- #
# Events and contexts - DELIBERATELY free of true_type / name / path
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AccessEvent:
    """A read or write of an opaque file id.

    Attributes:
        file_id: Opaque integer id.
        size_bytes: Size of the file.
        op: ``"read"`` or ``"write"``.
        epoch: Epoch index.
        step: Step index within the epoch.
        phase: One of ``warmup``, ``train``, ``validation``, ``checkpoint``.
    """

    file_id: int
    size_bytes: int
    op: str
    epoch: int
    step: int
    phase: str


@dataclass(frozen=True)
class TickContext:
    """Context passed to :meth:`Policy.on_tick`."""

    now_ms: float
    epoch: int
    step: int
    phase: str


@dataclass(frozen=True)
class PhaseContext:
    """Context passed to :meth:`Policy.on_phase_change`."""

    now_ms: float
    epoch: int
    previous_phase: str
    new_phase: str


# --------------------------------------------------------------------------- #
# Information-barrier protocols
# --------------------------------------------------------------------------- #
class PolicyView(Protocol):
    """The ONLY state a policy may inspect.

    Exposes tier occupancy/free space, tier membership of a file, file size,
    and the policy's own private history. No names, no true types, no oracle.
    """

    def occupancy(self) -> Mapping[int, int]:
        """Map tier id to bytes used."""

    def free_space(self) -> Mapping[int, Optional[int]]:
        """Map tier id to free bytes (``None`` means unlimited)."""

    def holds(self, file_id: int) -> tuple[int, ...]:
        """Return sorted tier ids holding a copy of ``file_id``."""

    def size_of(self, file_id: int) -> int:
        """Return the opaque file size in bytes."""

    @property
    def history(self) -> dict[str, object]:
        """Return this policy's private mutable history dictionary."""


class Oracle(Protocol):
    """Seeded future order. Only ClairvoyantPolicy and BeladyPolicy get this."""

    def upcoming(self, n: int, *, after: Optional[int] = None) -> list[int]:
        """Return the next ``n`` opaque ids in the future read stream."""

    def next_use(self, file_id: int) -> Optional[int]:
        """Return the index of the next use of ``file_id``, or ``None``."""

    def length(self) -> int:
        """Return the total number of future accesses."""


class Policy(Protocol):
    """Policy interface shared by every execution mode."""

    name: str

    def on_access(self, event: AccessEvent, view: PolicyView) -> list[Action]:
        """React to a read/write and return actions for the store."""

    def on_write(self, event: AccessEvent, view: PolicyView) -> int:
        """Choose the landing tier id for a write."""

    def on_tick(self, ctx: TickContext, view: PolicyView) -> list[Action]:
        """Make periodic background decisions."""

    def on_phase_change(self, ctx: PhaseContext, view: PolicyView) -> None:
        """React to a workload phase transition."""


# --------------------------------------------------------------------------- #
# Concrete baseline policies (Phase 2)
# --------------------------------------------------------------------------- #
class _PolicyBase:
    """Shared default behaviour: no actions, writes go to the origin."""

    def __init__(self) -> None:
        self.name = self.__class__.__name__
        self._history: dict[str, object] = {}

    @property
    def history(self) -> dict[str, object]:
        """Return this policy's private history."""
        return self._history

    def on_access(self, event: AccessEvent, view: PolicyView) -> list[Action]:
        """Default: do nothing."""
        return []

    def on_write(self, event: AccessEvent, view: PolicyView) -> int:
        """Default: land writes in the origin."""
        return SLOW

    def on_tick(self, ctx: TickContext, view: PolicyView) -> list[Action]:
        """Default: do nothing."""
        return []

    def on_phase_change(self, ctx: PhaseContext, view: PolicyView) -> None:
        """Default: ignore phase changes."""


class AllSlowPolicy(_PolicyBase):
    """Lower bound: never caches, every read hits the origin."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "AllSlow"


class AllFastPolicy(_PolicyBase):
    """Upper bound: all files pre-seeded in Tier 0; ignores capacity."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "AllFast"

    def on_write(self, event: AccessEvent, view: PolicyView) -> int:
        """Land writes in the fast tier."""
        return FAST


class LRUPolicy(_PolicyBase):
    """Cache-on-miss; the store evicts the least-recently-used copy."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "LRU"

    def on_access(self, event: AccessEvent, view: PolicyView) -> list[Action]:
        """Promote to Tier 0 on a miss; do nothing on a hit."""
        if FAST not in view.holds(event.file_id):
            return [Promote(file_id=event.file_id, tier=FAST)]
        return []


class LFUPolicy(_PolicyBase):
    """Cache-on-miss with least-frequently-used victim selection."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "LFU"
        self._counts: dict[int, int] = {}

    def on_access(self, event: AccessEvent, view: PolicyView) -> list[Action]:
        """Track frequency and evict the least frequent cached file on a miss."""
        file_id = event.file_id
        self._counts[file_id] = self._counts.get(file_id, 0) + 1
        if FAST in view.holds(file_id):
            return []

        actions: list[Action] = []
        free = view.free_space()[FAST]
        if free is not None and free < event.size_bytes:
            # One authoritative snapshot of FAST residency; iterating _counts
            # keeps the original deterministic victim choice on ties.
            fast_files = view.fast_resident_files()
            candidates = [
                cached_id for cached_id in self._counts if cached_id in fast_files
            ]
            if candidates:
                victim = min(candidates, key=lambda item: self._counts[item])
                actions.append(Evict(file_id=victim, tier=FAST))
        actions.append(Promote(file_id=file_id, tier=FAST))
        return actions


class ARCPolicy(_PolicyBase):
    """Byte-based Adaptive Replacement Cache (T1/T2 plus B1/B2 ghosts)."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "ARC"
        self.t1: OrderedDict[int, int] = OrderedDict()
        self.t2: OrderedDict[int, int] = OrderedDict()
        self.b1: OrderedDict[int, int] = OrderedDict()
        self.b2: OrderedDict[int, int] = OrderedDict()
        self.p: float = 0.0

    @staticmethod
    def _bytes(items: OrderedDict[int, int]) -> int:
        return sum(items.values())

    def _trim_ghosts(self, capacity: int) -> None:
        while self._bytes(self.b1) + self._bytes(self.b2) > capacity:
            if self.b1:
                self.b1.popitem(last=False)
            elif self.b2:
                self.b2.popitem(last=False)
            else:
                break

    def on_access(self, event: AccessEvent, view: PolicyView) -> list[Action]:
        """Update ARC state and emit eviction/promotion actions."""
        file_id = event.file_id
        size = view.size_of(file_id)
        free = view.free_space()[FAST]
        capacity = None if free is None else view.occupancy()[FAST] + free

        if FAST in view.holds(file_id):
            if file_id in self.t1:
                victim_size = self.t1.pop(file_id)
                self.t2[file_id] = victim_size
            elif file_id in self.b1:
                self.b1.pop(file_id)
                self.t2[file_id] = size
            elif file_id in self.b2:
                self.b2.pop(file_id)
                self.t2[file_id] = size
            elif file_id not in self.t2:
                self.t2[file_id] = size
            return []

        actions: list[Action] = []
        if file_id in self.b1:
            ghost_size = self.b1.pop(file_id)
            self.p += ghost_size
            self.t2[file_id] = size
        elif file_id in self.b2:
            self.b2.pop(file_id)
            self.p -= size
            self.t2[file_id] = size
        else:
            self.t1[file_id] = size

        if capacity is not None:
            self.p = max(0.0, min(float(capacity), self.p))
            cached_bytes = self._bytes(self.t1) + self._bytes(self.t2)
            while cached_bytes + size > capacity and (self.t1 or self.t2):
                if self.t1 and (self._bytes(self.t1) > self.p or not self.t2):
                    victim, victim_size = self.t1.popitem(last=False)
                    self.b1[victim] = victim_size
                elif self.t2:
                    victim, victim_size = self.t2.popitem(last=False)
                    self.b2[victim] = victim_size
                else:
                    break
                cached_bytes -= victim_size
                actions.append(Evict(file_id=victim, tier=FAST))
            self._trim_ghosts(capacity)
        actions.append(Promote(file_id=file_id, tier=FAST))
        return actions


class ClairvoyantPolicy(_PolicyBase):
    """NoPFS-style exact prefetching from the seeded future chunk order.

    In Scenario C the order is state-dependent, so the seeded oracle is
    deliberately withheld and the policy falls back to doing nothing.
    """

    def __init__(self, oracle: Optional[Oracle] = None, prefetch_window: int = 3) -> None:
        super().__init__()
        self.name = "Clairvoyant"
        self.oracle = oracle
        self.prefetch_window = prefetch_window
        self._pending: set[int] = set()

    def on_access(self, event: AccessEvent, view: PolicyView) -> list[Action]:
        """Promote unprefetched misses, mark consumption, refill the window."""
        self._pending.discard(event.file_id)
        actions: list[Action] = []
        if FAST not in view.holds(event.file_id):
            actions.append(Promote(file_id=event.file_id, tier=FAST))
        return actions + self._refill(view)

    def on_tick(self, ctx: TickContext, view: PolicyView) -> list[Action]:
        """Refill the prefetch window in the background."""
        return self._refill(view)

    def _refill(self, view: PolicyView) -> list[Action]:
        if self.oracle is None:
            return []
        free = view.free_space()[FAST]
        budget = float("inf") if free is None else max(0.0, free * 0.8)
        actions: list[Action] = []
        for file_id in self.oracle.upcoming_chunks(self.prefetch_window * 3):
            if len(actions) >= self.prefetch_window or budget <= 0:
                break
            if file_id in self._pending or FAST in view.holds(file_id):
                continue
            size = view.size_of(file_id)
            if size > budget:
                continue
            actions.append(Prefetch(file_id=file_id, tier=FAST))
            self._pending.add(file_id)
            budget -= size
        return actions


class BeladyPolicy(_PolicyBase):
    """Offline-optimal eviction for the full read stream (all file types)."""

    def __init__(self, oracle: Oracle, prefetch_window: int = 3) -> None:
        super().__init__()
        self.name = "Belady"
        self.oracle = oracle
        self.prefetch_window = prefetch_window
        self._seen: set[int] = set()
        self._pending: set[int] = set()

    def on_access(self, event: AccessEvent, view: PolicyView) -> list[Action]:
        """On a miss, evict the cached file with the farthest next use.

        Among equally-distant victims we prefer one whose size alone frees
        enough space, avoiding a second (unnecessary) eviction.
        """
        file_id = event.file_id
        self._seen.add(file_id)
        self._pending.discard(file_id)
        if FAST in view.holds(file_id):
            return []

        actions: list[Action] = []
        free = view.free_space()[FAST]
        if free is not None and free < event.size_bytes:
            required = event.size_bytes - free
            # One snapshot of FAST residency and sizes replaces an O(n) pair of
            # holds()/size_of() calls per candidate. Iterating self._seen keeps
            # the original deterministic order (and victim choice on ties).
            fast_files = view.fast_resident_files()
            candidates = [
                (cached_id, fast_files[cached_id])
                for cached_id in self._seen
                if cached_id in fast_files
            ]
            enough = [pair for pair in candidates if pair[1] >= required]
            pool = enough or candidates

            def key(pair: tuple[int, int]) -> tuple[int, int, int]:
                distance = self.oracle.next_use(pair[0])
                far = (1, distance) if distance is not None else (0, -1)
                return (far[0], far[1], pair[1])

            if pool:
                victim = max(pool, key=key)
                actions.append(Evict(file_id=victim[0], tier=FAST))
        if file_id not in self._pending:
            actions.append(Promote(file_id=file_id, tier=FAST))
        return actions

    def on_tick(self, ctx: TickContext, view: PolicyView) -> list[Action]:
        """Prefetch the exact upcoming reads into free fast space."""
        free = view.free_space()[FAST]
        budget = float("inf") if free is None else max(0.0, free * 0.8)
        actions: list[Action] = []
        for file_id in self.oracle.upcoming(self.prefetch_window * 3):
            if len(actions) >= self.prefetch_window or budget <= 0:
                break
            if file_id in self._pending or FAST in view.holds(file_id):
                continue
            size = view.size_of(file_id)
            if size > budget:
                continue
            actions.append(Prefetch(file_id=file_id, tier=FAST))
            self._pending.add(file_id)
            budget -= size
        return actions


class ContextRulesPolicy(_PolicyBase):
    """Oracle-context upper bound driven by an injected declared-type map.

    This is a deliberately privileged policy: the harness supplies
    ``file_id -> declared_type`` at construction time, analogous to the seeded
    oracle. Neither ``AccessEvent`` nor ``PolicyView`` exposes the map. The
    ``no_context`` ablation receives an empty map, so its rules degenerate to
    plain cache-on-miss LRU.

    Rules:
        * ``metadata`` / ``hot_cache``: promote once and pin permanently in fast.
        * ``validation``: promote once and pin permanently in mid, keeping the
          fast tier free for training chunks without paying origin latency.
        * everything else: promote on a miss.
        * writes: always land in the origin.
    """

    def __init__(
        self,
        declared_types: Optional[Mapping[int, str]] = None,
        *,
        no_context: bool = False,
    ) -> None:
        super().__init__()
        self.name = "ContextRulesNoCtx" if no_context else "ContextRules"
        self.declared_types: dict[int, str] = (
            {} if no_context else dict(declared_types or {})
        )
        self._pinned: set[int] = set()

    def on_access(self, event: AccessEvent, view: PolicyView) -> list[Action]:
        """Apply the declared-type rules to one read."""
        kind = self.declared_types.get(event.file_id)
        if kind in {"metadata", "hot_cache"}:
            if event.file_id in self._pinned or FAST in view.holds(event.file_id):
                return []
            self._pinned.add(event.file_id)
            return [
                Promote(file_id=event.file_id, tier=FAST),
                Pin(file_id=event.file_id),
            ]
        if kind == "validation":
            holders = view.holds(event.file_id)
            if event.file_id in self._pinned or MID in holders or FAST in holders:
                return []
            self._pinned.add(event.file_id)
            return [
                Promote(file_id=event.file_id, tier=MID),
                Pin(file_id=event.file_id),
            ]
        if FAST not in view.holds(event.file_id):
            return [Promote(file_id=event.file_id, tier=FAST)]
        return []

    def on_write(self, event: AccessEvent, view: PolicyView) -> int:
        """Land all writes, including checkpoints, in the origin."""
        del event, view
        return SLOW


class ClassifierPolicy(_PolicyBase):
    """Use an injected file-type classifier instead of declared types.

    The model is trained offline by ``classifier.py`` on observable features.
    This policy never receives true types, names, oracle order or declared
    types; it only sees the model's prediction for the current event.
    """

    def __init__(self, model: Optional[object] = None) -> None:
        super().__init__()
        self.name = "Classifier"
        self.model = model
        self.extractor = FeatureExtractor()

    def _predicted_type(self, event: AccessEvent, view: PolicyView) -> str:
        if self.model is None:
            return "unknown"
        features = self.extractor.extract(event, view)
        vector = np.asarray([feature_vector(features)], dtype=np.float32)
        prediction = self.model.predict(vector)[0]
        return str(prediction)

    def on_access(self, event: AccessEvent, view: PolicyView) -> list[Action]:
        """Apply placement rules based on the inferred file type."""
        kind = self._predicted_type(event, view)
        if kind in {"metadata", "hot_cache"}:
            if FAST in view.holds(event.file_id):
                return []
            return [
                Promote(file_id=event.file_id, tier=FAST),
                Pin(file_id=event.file_id),
            ]
        if kind == "validation":
            holders = view.holds(event.file_id)
            if MID in holders or FAST in holders:
                return []
            return [
                Promote(file_id=event.file_id, tier=MID),
                Pin(file_id=event.file_id),
            ]
        if FAST not in view.holds(event.file_id):
            return [Promote(file_id=event.file_id, tier=FAST)]
        return []

    def on_write(self, event: AccessEvent, view: PolicyView) -> int:
        """Land all writes in the origin."""
        del event, view
        return SLOW


class EmbeddingPrefetchPolicy(_PolicyBase):
    """Prefetch files whose learned embedding predicts near-term reuse.

    The reuse predictor is injected by the harness; this policy never sees
    true types, declared types or the oracle. It promotes the current file when
    its predicted reuse exceeds a threshold, and speculatively prefetches a few
    already-seen files with similar embeddings that are also predicted reusable.
    """

    def __init__(self, predictor: Optional[object] = None, *, threshold: float = 0.5) -> None:
        super().__init__()
        self.name = "Embedding"
        self.predictor = predictor
        self.threshold = threshold
        self._extractor = None
        self._seen: set[int] = set()
        self._pinned: set[int] = set()

    def _setup_extractor(self) -> None:
        if self._extractor is None:
            from features import FeatureExtractor

            self._extractor = FeatureExtractor()

    def _score(self, event: AccessEvent, view: PolicyView) -> float:
        self._setup_extractor()
        if self.predictor is None:
            return 0.0
        try:
            return float(
                self.predictor.predict_proba(
                    self.predictor.features(event, view, self._extractor)
                )
            )
        except Exception:
            return 0.0

    def on_access(self, event: AccessEvent, view: PolicyView) -> list[Action]:
        """Keep likely-reused files in fast and prefetch similar seen files."""
        score = self._score(event, view)
        self._seen.add(event.file_id)
        actions: list[Action] = []
        if FAST not in view.holds(event.file_id):
            actions.append(Promote(file_id=event.file_id, tier=FAST))
        if (
            score >= self.threshold
            and event.file_id not in self._pinned
            and (FAST in view.holds(event.file_id) or actions)
        ):
            self._pinned.add(event.file_id)
            actions.append(Pin(file_id=event.file_id))
        return actions


# --------------------------------------------------------------------------- #
# RL action space (Phase 7)
# --------------------------------------------------------------------------- #
RL_ACTION_NAMES: tuple[str, ...] = (
    "NOOP",
    "PROMOTE_FAST",
    "PROMOTE_MID",
)

# PIN_FAST / PREFETCH_FAST were evaluated as RL actions and removed: they yield
# the same immediate reward as PROMOTE_FAST but have hidden long-term costs
# (pinned bytes make eviction impossible; speculative prefetches can be dropped
# or thrash the fast tier). Pinning remains available to the classifier and
# reuse components of FullContextTierPolicy, and to ContextRulesPolicy.


def rl_action_to_actions(
    action_index: int,
    file_id: int,
    view: PolicyView,
) -> list[Action]:
    """Map one discrete RL action to concrete store actions for ``file_id``.

    This is the single source of truth shared by the training environment and
    :class:`FullContextTierPolicy`, so learned behaviour cannot drift between
    train and inference. The RL agent never sees ``true_type``, names, the
    oracle or declared types; only ``view`` and the current ``AccessEvent``.
    """
    index = int(action_index)
    if index < 0:  # Python negative indexing must not select a real action.
        name = "NOOP"
    else:
        try:
            name = RL_ACTION_NAMES[index]
        except (ValueError, TypeError, IndexError):
            name = "NOOP"

    holders = view.holds(file_id)
    if name == "PROMOTE_FAST":
        return [] if FAST in holders else [Promote(file_id=file_id, tier=FAST)]
    if name == "PROMOTE_MID":
        return [] if MID in holders else [Promote(file_id=file_id, tier=MID)]
    return []


class FullContextTierPolicy(_PolicyBase):
    """Combine classifier, reuse predictor and RL decisions (Phase 7).

    This is the "full learned context" policy. It composes the three learned
    signals trained in Phases 5-7, all of which use only policy-observable data:

    1. the Phase 5 classifier predicts a file type from observable features;
       metadata/hot-cache files go to fast and are pinned, validation files go
       to mid and are pinned;
    2. for the remaining files the Phase 7 RL agent chooses a discrete action
       via :func:`rl_action_to_actions`;
    3. the Phase 6 reuse predictor can pin a file that is about to be reused.

    It deliberately receives NO oracle and NO declared-type map. The privileged
    oracle-context channel stays exclusive to ``ContextRulesPolicy`` per the
    information barrier in ``docs/DESIGN.md``.
    """

    def __init__(
        self,
        agent: Optional[object] = None,
        classifier: Optional[object] = None,
        predictor: Optional[object] = None,
        *,
        reuse_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.name = "FullContextTier"
        self.agent = agent
        self.classifier = classifier
        self.predictor = predictor
        self.reuse_threshold = float(reuse_threshold)
        # Each component owns its extractor so per-file access counters are not
        # advanced multiple times for a single event.
        self._rl_extractor = FeatureExtractor()
        self._clf_extractor = FeatureExtractor()
        self._reuse_extractor = FeatureExtractor()
        self._pinned: set[int] = set()

    def _state(self, event: AccessEvent, view: PolicyView) -> list[float]:
        """Build the RL observation from policy-visible features only."""
        return feature_vector(self._rl_extractor.extract(event, view))

    def _predicted_type(self, event: AccessEvent, view: PolicyView) -> str:
        if self.classifier is None:
            return "unknown"
        features = self._clf_extractor.extract(event, view)
        vector = np.asarray([feature_vector(features)], dtype=np.float32)
        try:
            return str(self.classifier.predict(vector)[0])
        except Exception:
            return "unknown"

    def _reuse_score(self, event: AccessEvent, view: PolicyView) -> float:
        if self.predictor is None:
            return 0.0
        try:
            vector = self.predictor.features(event, view, self._reuse_extractor)
            return float(self.predictor.predict_proba(vector))
        except Exception:
            return 0.0

    def _rl_actions(self, event: AccessEvent, view: PolicyView) -> list[Action]:
        if self.agent is None:
            # No learned agent: degrade to plain cache-on-miss LRU.
            return rl_action_to_actions(1, event.file_id, view)
        state = self._state(event, view)
        try:
            action_index = int(self.agent.select_action(state, greedy=True))
        except Exception:
            action_index = 0
        return rl_action_to_actions(action_index, event.file_id, view)

    def on_access(self, event: AccessEvent, view: PolicyView) -> list[Action]:
        """Apply classifier rules, then RL actions, then reuse-based pinning."""
        file_id = event.file_id
        kind = self._predicted_type(event, view)

        if kind in {"metadata", "hot_cache"}:
            if file_id in self._pinned or FAST in view.holds(file_id):
                return []
            self._pinned.add(file_id)
            return [Promote(file_id=file_id, tier=FAST), Pin(file_id=file_id)]

        if kind == "validation":
            holders = view.holds(file_id)
            if file_id in self._pinned or MID in holders or FAST in holders:
                return []
            self._pinned.add(file_id)
            return [Promote(file_id=file_id, tier=MID), Pin(file_id=file_id)]

        actions = self._rl_actions(event, view)
        if (
            self.predictor is not None
            and file_id not in self._pinned
            and self._reuse_score(event, view) >= self.reuse_threshold
            and any(isinstance(action, (Promote, Prefetch)) for action in actions)
        ):
            self._pinned.add(file_id)
            actions.append(Pin(file_id=file_id))
        return actions

    def on_write(self, event: AccessEvent, view: PolicyView) -> int:
        """Land writes in the origin, consistent with the other learned policies."""
        del event, view
        return SLOW


class TemperaturePolicy(_PolicyBase):
    """Observable hot/warm/cold placement from an EWMA access-rate estimate.

    This policy only observes opaque ids, arrival order, sizes and cache state.
    ``true_temp`` never enters its constructor, event, view, or history.
    """

    def __init__(self, half_life: float = 32.0, hysteresis_ticks: int = 3) -> None:
        super().__init__()
        self.name = "Temperature"
        self.half_life = max(1.0, float(half_life))
        self.hysteresis_ticks = max(1, int(hysteresis_ticks))
        self._tick = 0
        self._last_seen: dict[int, int] = {}
        self._score: dict[int, float] = {}
        self._candidate: dict[int, str] = {}
        self._stable: dict[int, int] = {}

    def _update(self, file_id: int) -> None:
        previous = self._last_seen.get(file_id, self._tick)
        decay = 0.5 ** ((self._tick - previous) / self.half_life)
        self._score[file_id] = self._score.get(file_id, 0.0) * decay + 1.0
        self._last_seen[file_id] = self._tick

    def _label(self, file_id: int) -> str:
        values = sorted(self._score.values())
        if not values:
            return "cold"
        hot_cut = values[max(0, int(0.9 * (len(values) - 1)))]
        warm_cut = values[max(0, int(0.6 * (len(values) - 1)))]
        value = self._score.get(file_id, 0.0)
        return "hot" if value >= hot_cut else "warm" if value >= warm_cut else "cold"

    def on_access(self, event: AccessEvent, view: PolicyView) -> list[Action]:
        self._tick += 1
        self._update(event.file_id)
        proposed = self._label(event.file_id)
        if self._candidate.get(event.file_id) == proposed:
            self._stable[event.file_id] = self._stable.get(event.file_id, 0) + 1
        else:
            self._candidate[event.file_id] = proposed
            self._stable[event.file_id] = 1
        if self._stable[event.file_id] < self.hysteresis_ticks:
            return []
        holders = view.holds(event.file_id)
        if proposed == "hot" and FAST not in holders:
            return [Promote(event.file_id, FAST)]
        if proposed == "warm" and FAST not in holders and MID not in holders:
            return [Promote(event.file_id, MID)]
        if proposed == "cold":
            actions: list[Action] = []
            if FAST in holders:
                actions.append(Evict(event.file_id, FAST))
            if MID in holders:
                actions.append(Evict(event.file_id, MID))
            return actions
        return []


class TemperatureLearnedPolicy(TemperaturePolicy):
    """Temperature placement using an observable reuse-predictor probability."""

    def __init__(self, predictor: Optional[object] = None, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.name = "TemperatureLearned"
        self.predictor = predictor
        self._extractor = FeatureExtractor()

    def _label(self, file_id: int) -> str:
        # Fall back to the EWMA bucket until a predictor can score the access.
        return super()._label(file_id)

    def on_access(self, event: AccessEvent, view: PolicyView) -> list[Action]:
        self._tick += 1
        self._update(event.file_id)
        proposed = super()._label(event.file_id)
        if self.predictor is not None:
            try:
                probability = float(
                    self.predictor.predict_proba(
                        self.predictor.features(event, view, self._extractor)
                    )
                )
                proposed = "hot" if probability >= 0.66 else "warm" if probability >= 0.33 else "cold"
            except Exception:
                pass
        self._candidate[event.file_id] = proposed
        self._stable[event.file_id] = self._stable.get(event.file_id, 0) + 1
        if self._stable[event.file_id] < self.hysteresis_ticks:
            return []
        holders = view.holds(event.file_id)
        if proposed == "hot" and FAST not in holders:
            return [Promote(event.file_id, FAST)]
        if proposed == "warm" and FAST not in holders and MID not in holders:
            return [Promote(event.file_id, MID)]
        if proposed == "cold":
            return [Evict(event.file_id, tier) for tier in (FAST, MID) if tier in holders]
        return []


POLICY_CLASSES: dict[str, type] = {
    "allslow": AllSlowPolicy,
    "allfast": AllFastPolicy,
    "lru": LRUPolicy,
    "lfu": LFUPolicy,
    "arc": ARCPolicy,
    "clairvoyant": ClairvoyantPolicy,
    "belady": BeladyPolicy,
    "context_rules": ContextRulesPolicy,
    "context_rules_noctx": ContextRulesPolicy,
    "classifier": ClassifierPolicy,
    "embedding": EmbeddingPrefetchPolicy,
    "full_context": FullContextTierPolicy,
    "temperature": TemperaturePolicy,
    "temperature_learned": TemperatureLearnedPolicy,
}
