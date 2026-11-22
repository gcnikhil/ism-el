"""Thread-safe tiered store with exact migration accounting.

Virtual-time semantics:

* A read is non-mutating and is served from the fastest tier holding a copy.
* Only explicit ``promote`` / ``prefetch`` / ``flush`` operations move bytes.
* Migration bandwidth is reserved through each tier's ``busy_until_ms``; a read
  waits only for the *remaining* transfer time of an in-flight migration.
* Background prefetches use a bounded queue and a daemon worker that registers
  scheduled jobs; jobs materialize when the virtual clock reaches completion.

All mutable state is guarded by a single re-entrant lock.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from policies import AccessEvent, Evict
from tiers import FAST, SLOW, CapacityError, Tier


# --------------------------------------------------------------------------- #
# Clock
# --------------------------------------------------------------------------- #
@dataclass
class VirtualClock:
    """Deterministic simulated clock in milliseconds."""

    real: bool = False
    _now_ms: float = 0.0
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False
    )

    def now_ms(self) -> float:
        """Return the current simulated time."""
        with self._lock:
            return self._now_ms

    def advance(self, dt_ms: float) -> None:
        """Move the clock forward by ``dt_ms``."""
        if dt_ms < 0:
            raise ValueError("Cannot move a clock backwards")
        with self._lock:
            self._now_ms += dt_ms

    def wait_until(self, t_ms: float) -> None:
        """Advance the clock to ``t_ms`` if it is in the future."""
        with self._lock:
            if t_ms > self._now_ms:
                self._now_ms = t_ms


# --------------------------------------------------------------------------- #
# Results and internal jobs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReadResult:
    """Outcome of one :meth:`TieredStore.get` call."""

    file_id: int
    tier_served: int
    latency_ms: float
    wait_ms: float
    fast_hit: bool


@dataclass(frozen=True)
class MigrationResult:
    """Outcome of one scheduled copy between tiers."""

    file_id: int
    src_tier: Optional[int]
    dst_tier: int
    size_bytes: int
    scheduled_ms: float
    completed_ms: float
    kind: str


@dataclass
class _MigrationJob:
    """Internal background-migration record."""

    job_id: int
    file_id: int
    src_tier: int
    dst_tier: int
    size_bytes: int
    kind: str
    prefetch: bool
    scheduled_ms: float = 0.0
    completed_ms: float = 0.0


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class TieredStore:
    """Simulated storage hierarchy with one permanent origin tier."""

    def __init__(
        self,
        tiers: dict[int, Tier],
        clock: VirtualClock,
        *,
        max_queue: int = 64,
    ) -> None:
        """Create a store.

        Args:
            tiers: Mapping of tier id to :class:`Tier`; must include 0, 1, 2.
            clock: Clock used for all timing decisions.
            max_queue: Bounded capacity of the background migration queue.
        """
        if set(tiers) != {0, 1, 2}:
            raise ValueError("TieredStore requires tier ids {0, 1, 2}")
        self.tiers = dict(tiers)
        self.clock = clock
        self._lock = threading.RLock()
        self._queue: queue.Queue[Optional[_MigrationJob]] = queue.Queue(
            maxsize=max_queue
        )
        self._inflight: dict[int, _MigrationJob] = {}
        self._completed_jobs: set[int] = set()
        self._next_job_id = 0
        self._migration_bytes = 0
        self._wasted_prefetch_bytes = 0
        self._last_access: dict[tuple[int, int], float] = {}
        self._arrived_prefetch: dict[int, tuple[int, int]] = {}
        self._consumed_jobs: set[int] = set()
        self._cancelled_jobs: set[int] = set()
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop, name="contexttier-migration", daemon=True
        )
        self._worker.start()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Stop the background worker (idempotent)."""
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=2.0)

    def __enter__(self) -> "TieredStore":
        """Return self for context-manager use."""
        return self

    def __exit__(self, *_: object) -> None:
        """Stop the worker on context exit."""
        self.close()

    def drain(self) -> None:
        """Block until every queued job has been registered by the worker."""
        self._queue.join()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        """Best-effort worker cleanup if the caller forgot ``close``."""
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def migration_bytes(self) -> int:
        """Cumulative bytes copied between tiers (exact)."""
        with self._lock:
            return self._migration_bytes

    @property
    def wasted_prefetch_bytes(self) -> int:
        """Cumulative prefetched bytes evicted before being read."""
        with self._lock:
            return self._wasted_prefetch_bytes

    def locate(self, file_id: int) -> tuple[int, ...]:
        """Return sorted tier ids holding a copy of ``file_id``."""
        with self._lock:
            return tuple(
                sorted(tier_id for tier_id, tier in self.tiers.items() if tier.holds(file_id))
            )

    def fast_resident(self, file_id: int) -> bool:
        """Return whether Tier 0 holds the file *at the current instant*.

        This snapshot must be taken when a read arrives, before the policy can
        issue a demand promotion for that same access. It materialises any
        migrations whose virtual transfer already completed, so a prefetch that
        landed earlier counts as resident, while one still in flight does not.
        """
        with self._lock:
            self._materialize_due()
            return self.tiers[FAST].holds(file_id)

    def register(self, file_id: int, size_bytes: int) -> None:
        """Create the permanent origin copy of a file (not a migration)."""
        with self._lock:
            origin = self.tiers[SLOW]
            if not origin.holds(file_id):
                origin.add(file_id, size_bytes)
            else:
                origin.files[file_id] = size_bytes
            self._last_access[(SLOW, file_id)] = self.clock.now_ms()

    # ------------------------------------------------------------------ #
    # Core operations
    # ------------------------------------------------------------------ #
    def get(
        self,
        file_id: int,
        event: Optional[AccessEvent] = None,
        *,
        arrival_fast: Optional[bool] = None,
    ) -> ReadResult:
        """Read a file from its fastest holding tier.

        The operation does not mutate placement. If a faster prefetch is still
        in flight, the simulated clock waits only for its remaining transfer.

        ``arrival_fast`` is the Tier-0 residency snapshot taken *before* any
        demand promotion for this access. When omitted, the snapshot is taken
        at method entry (still before any waiting for an in-flight prefetch).
        ``fast_hit`` therefore always describes arrival-time residency and is
        distinct from ``tier_served``, which is the tier that finally served
        the read.
        """
        del event  # accepted for interface symmetry; placement is untouched
        with self._lock:
            self._materialize_due()
            if arrival_fast is None:
                arrival_fast = self.tiers[FAST].holds(file_id)
            holders = self._holders(file_id)
            wait_ms = 0.0

            pending = [
                job
                for job in self._inflight.values()
                if job.file_id == file_id
                and job.job_id not in self._cancelled_jobs
                and job.dst_tier < min(holders)
            ]
            if pending:
                eta = min(job.completed_ms for job in pending)
                delta = max(0.0, eta - self.clock.now_ms())
                if delta > 0:
                    self.clock.wait_until(eta)
                    wait_ms += delta
                    self._materialize_due()
                    holders = self._holders(file_id)

            tier_id = min(holders)
            tier = self.tiers[tier_id]
            # Bandwidth reservations only delay reads of the specific file
            # whose prefetch is in flight (handled above). Other migrations on
            # the same tier must not block an already-arrived copy.
            size = tier.files[file_id]
            latency_ms = tier.spec.latency_ms + size / (tier.spec.bandwidth_mbps * 1024.0)
            self.clock.advance(latency_ms)
            self._last_access[(tier_id, file_id)] = self.clock.now_ms()

            for job_id, (arrived_id, _) in list(self._arrived_prefetch.items()):
                if arrived_id == file_id:
                    self._consumed_jobs.add(job_id)
                    del self._arrived_prefetch[job_id]

            return ReadResult(
                file_id=file_id,
                tier_served=tier_id,
                latency_ms=latency_ms,
                wait_ms=wait_ms,
                fast_hit=bool(arrival_fast),
            )

    def put(
        self,
        event: AccessEvent,
        landing_tier: int,
        *,
        dirty: bool = True,
    ) -> None:
        """Write a file, choosing its landing tier via the policy decision."""
        with self._lock:
            size = event.size_bytes
            origin = self.tiers[SLOW]
            if not origin.holds(event.file_id):
                origin.add(event.file_id, size)
            else:
                origin.files[event.file_id] = size

            tier = self.tiers[landing_tier]
            if tier.holds(event.file_id):
                tier.files[event.file_id] = size
                if landing_tier != SLOW and dirty:
                    tier.dirty.add(event.file_id)
            else:
                self._ensure_space(landing_tier, size)
                tier.add(event.file_id, size, dirty=(landing_tier != SLOW and dirty))
            self._last_access[(landing_tier, event.file_id)] = self.clock.now_ms()

    def promote(self, file_id: int, tier: int) -> Optional[MigrationResult]:
        """Copy a file up synchronously; return ``None`` if already resident."""
        with self._lock:
            if tier == SLOW or self.tiers[tier].holds(file_id):
                return None
            size = self._size_of(file_id)
            source = self._fastest_holder(file_id)
            self._ensure_space(tier, size)
            job = self._new_job(file_id, source, tier, size, "promote", False)
            self._schedule(job)
            self.clock.wait_until(job.completed_ms)
            self._commit(job)
            return self._result(job)

    def prefetch(self, file_id: int, tier: int) -> Optional[MigrationResult]:
        """Enqueue a bounded background copy up; return ``None`` if dropped."""
        with self._lock:
            if tier == SLOW or self.tiers[tier].holds(file_id):
                return None
            if self._queue.full():
                return None
            size = self._size_of(file_id)
            source = self._fastest_holder(file_id)
            self._ensure_space(tier, size)
            job = self._new_job(file_id, source, tier, size, "prefetch", True)
            self._schedule(job)
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                return None
            # Register synchronously as well: virtual time can advance past a
            # job's completion before the worker thread is scheduled. The
            # worker skips jobs that are already registered or completed.
            self._inflight[job.job_id] = job
            return self._result(job)

    def evict(self, file_id: int, tier: int) -> bool:
        """Drop a cached copy; refuse origin, pinned and dirty files."""
        with self._lock:
            if tier == SLOW:
                return False
            target = self.tiers[tier]
            if not target.holds(file_id):
                return False
            if file_id in target.pinned or file_id in target.dirty:
                return False
            size = target.remove(file_id)
            self._last_access.pop((tier, file_id), None)

            for job_id, (arrived_id, arrived_tier) in list(self._arrived_prefetch.items()):
                if arrived_id == file_id and arrived_tier == tier:
                    if job_id not in self._consumed_jobs:
                        self._wasted_prefetch_bytes += size
                    del self._arrived_prefetch[job_id]

            for job in self._inflight.values():
                if (
                    job.file_id == file_id
                    and job.dst_tier == tier
                    and job.prefetch
                    and job.job_id not in self._consumed_jobs
                    and job.job_id not in self._cancelled_jobs
                ):
                    self._cancelled_jobs.add(job.job_id)
                    self._wasted_prefetch_bytes += job.size_bytes
            return True

    def flush(self, file_id: int) -> Optional[MigrationResult]:
        """Write dirty cached copies back to the origin (counted as migration)."""
        with self._lock:
            dirty_tiers = [
                tier_id
                for tier_id, tier in self.tiers.items()
                if tier_id != SLOW and tier.holds(file_id) and file_id in tier.dirty
            ]
            if not dirty_tiers:
                return None
            source = min(dirty_tiers)
            size = self.tiers[source].files[file_id]
            job = self._new_job(file_id, source, SLOW, size, "flush", False)
            self._schedule(job)
            self.clock.wait_until(job.completed_ms)
            self._commit(job)
            for tier_id in dirty_tiers:
                self.tiers[tier_id].dirty.discard(file_id)
            return self._result(job)

    def enforce_capacity(self, tier: int) -> list[Evict]:
        """Evict eligible victims until ``tier`` is within capacity."""
        if tier == SLOW:
            return []
        with self._lock:
            return self._ensure_space(tier, 0)

    def tick(self, ctx: object) -> None:
        """Materialize any background migrations that have completed."""
        del ctx
        with self._lock:
            self._materialize_due()

    # ------------------------------------------------------------------ #
    # Internals (all called while holding self._lock)
    # ------------------------------------------------------------------ #
    def _worker_loop(self) -> None:
        """Register queued jobs; completion is driven by the clock."""
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                with self._lock:
                    if (
                        job.job_id not in self._inflight
                        and job.job_id not in self._completed_jobs
                    ):
                        self._inflight[job.job_id] = job
            finally:
                self._queue.task_done()

    def _holders(self, file_id: int) -> list[int]:
        holders = [
            tier_id for tier_id, tier in self.tiers.items() if tier.holds(file_id)
        ]
        if not holders:
            raise KeyError(f"Unknown file_id {file_id}; call register() first")
        return holders

    def _fastest_holder(self, file_id: int) -> int:
        return min(self._holders(file_id))

    def _size_of(self, file_id: int) -> int:
        holders = self._holders(file_id)
        return self.tiers[holders[0]].files[file_id]

    def _new_job(
        self,
        file_id: int,
        src: int,
        dst: int,
        size: int,
        kind: str,
        prefetch: bool,
    ) -> _MigrationJob:
        job = _MigrationJob(
            job_id=self._next_job_id,
            file_id=file_id,
            src_tier=src,
            dst_tier=dst,
            size_bytes=size,
            kind=kind,
            prefetch=prefetch,
        )
        self._next_job_id += 1
        return job

    def _schedule(self, job: _MigrationJob) -> None:
        """Reserve bandwidth on both endpoints and count the copied bytes."""
        source = self.tiers[job.src_tier]
        destination = self.tiers[job.dst_tier]
        now = self.clock.now_ms()
        start = max(now, source.busy_until_ms, destination.busy_until_ms)
        transfer_ms = source.transfer_ms(job.size_bytes, destination)
        completion = start + transfer_ms
        source.busy_until_ms = completion
        destination.busy_until_ms = completion
        job.scheduled_ms = start
        job.completed_ms = completion
        self._migration_bytes += job.size_bytes

    def _result(self, job: _MigrationJob) -> MigrationResult:
        return MigrationResult(
            file_id=job.file_id,
            src_tier=job.src_tier,
            dst_tier=job.dst_tier,
            size_bytes=job.size_bytes,
            scheduled_ms=job.scheduled_ms,
            completed_ms=job.completed_ms,
            kind=job.kind,
        )

    def _materialize_due(self) -> None:
        now = self.clock.now_ms()
        for job_id, job in list(self._inflight.items()):
            if job.completed_ms > now:
                continue
            if job_id not in self._cancelled_jobs:
                self._commit(job)
            del self._inflight[job_id]
            self._completed_jobs.add(job_id)

    def _commit(self, job: _MigrationJob) -> None:
        destination = self.tiers[job.dst_tier]
        if not destination.holds(job.file_id):
            if not destination.can_fit(job.size_bytes):
                self._ensure_space(job.dst_tier, job.size_bytes)
            destination.add(job.file_id, job.size_bytes)
        self._last_access[(job.dst_tier, job.file_id)] = self.clock.now_ms()
        arrived_allowed = (
            job.job_id not in self._consumed_jobs
            and job.job_id not in self._cancelled_jobs
        )
        if job.prefetch and arrived_allowed:
            self._arrived_prefetch[job.job_id] = (job.file_id, job.dst_tier)

    def _ensure_space(self, tier: int, size_bytes: int) -> list[Evict]:
        """Evict the least-recently-used eligible victims until ``size`` fits."""
        target = self.tiers[tier]
        capacity = target.spec.capacity_bytes
        if capacity is None:
            return []
        required = max(0, target.used_bytes() + size_bytes - capacity)
        if required == 0:
            return []

        candidates = [
            (self._last_access.get((tier, file_id), 0.0), file_id, size)
            for file_id, size in target.files.items()
            if file_id not in target.pinned and file_id not in target.dirty
        ]
        candidates.sort(key=lambda entry: (entry[0], entry[1]))

        actions: list[Evict] = []
        freed = 0
        for _, file_id, size in candidates:
            if freed >= required:
                break
            if self.evict(file_id, tier):
                freed += size
                actions.append(Evict(file_id=file_id, tier=tier))
        if freed < required:
            raise CapacityError(
                f"Tier {tier} cannot free {required} bytes: "
                "remaining files are pinned or dirty"
            )
        return actions

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        """Return a compact representation of tier occupancies."""
        usage = {tier_id: tier.used_bytes() for tier_id, tier in self.tiers.items()}
        return f"TieredStore(usage={usage}, migrations={self.migration_bytes})"


# --------------------------------------------------------------------------- #
# Real-mode clock and payload-carrying store (Phase 3)
# --------------------------------------------------------------------------- #
@dataclass
class RealClock:
    """Wall-clock implementation with ``time.sleep`` realizing tier delays."""

    real: bool = True
    _origin_ms: float = field(default_factory=lambda: time.perf_counter() * 1000.0)

    def now_ms(self) -> float:
        """Return elapsed wall time in milliseconds."""
        return time.perf_counter() * 1000.0 - self._origin_ms

    def advance(self, dt_ms: float) -> None:
        """Realize a delay; busy-wait below 2 ms to avoid sleep granularity."""
        if dt_ms <= 0:
            return
        if dt_ms < 2.0:
            target = time.perf_counter() + dt_ms / 1000.0
            while time.perf_counter() < target:
                pass
        else:
            time.sleep(dt_ms / 1000.0)

    def wait_until(self, t_ms: float) -> None:
        """Wait until wall time ``t_ms`` is reached."""
        remaining = t_ms - self.now_ms()
        if remaining > 0:
            self.advance(remaining)


class PayloadStore(TieredStore):
    """A TieredStore that also carries real bytes for cached copies.

    The origin reads from real files on disk; cache tiers hold ``bytes``
    payloads. Byte movement is still accounted by the parent class.
    """

    def __init__(
        self,
        tiers: dict[int, Tier],
        clock: RealClock,
        origin_paths: dict[int, Path],
        *,
        max_queue: int = 64,
    ) -> None:
        """Create a payload store with opaque-id to origin-path mapping."""
        super().__init__(tiers, clock, max_queue=max_queue)
        self.origin_paths = dict(origin_paths)
        self.payloads: dict[int, bytes] = {}
        self._payload_lock = threading.RLock()

    def read_payload(self, file_id: int, tier_served: int) -> bytes:
        """Return the bytes served by a read from ``tier_served``."""
        if tier_served != SLOW:
            with self._payload_lock:
                if file_id in self.payloads:
                    return self.payloads[file_id]
        path = self.origin_paths[file_id]
        return path.read_bytes()

    def write_payload(self, file_id: int, data: bytes) -> None:
        """Attach payload bytes to a freshly written cached file."""
        with self._payload_lock:
            self.payloads[file_id] = data

    def _commit(self, job: object) -> None:
        """Materialize a migration and cache the corresponding payload."""
        super()._commit(job)  # type: ignore[arg-type]
        migration = job  # _MigrationJob is private but structurally known here
        file_id = int(getattr(migration, "file_id"))
        dst_tier = int(getattr(migration, "dst_tier"))
        if dst_tier == SLOW:
            return
        path = self.origin_paths.get(file_id)
        if path is None or not path.is_file():
            return
        with self._payload_lock:
            if file_id not in self.payloads:
                self.payloads[file_id] = path.read_bytes()

    def evict(self, file_id: int, tier: int) -> bool:
        """Evict a copy and drop its payload once no cache holds it."""
        result = super().evict(file_id, tier)
        if result and not any(
            self.tiers[cache].holds(file_id) for cache in (FAST, 1)
        ):
            with self._payload_lock:
                self.payloads.pop(file_id, None)
        return result
