"""Tier records and tier construction for ContextTier.

Tier 2 is the permanent origin; tiers 0 and 1 are finite caches. All state is
plain Python data so virtual mode never touches real storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

FAST: int = 0
MID: int = 1
SLOW: int = 2
MB: int = 1024 * 1024


@dataclass(frozen=True)
class TierSpec:
    """Immutable hardware-style description of one tier."""

    tier_id: int
    name: str
    capacity_bytes: Optional[int]  # None => unlimited (origin only)
    latency_ms: float
    bandwidth_mbps: float  # MB/s, where 1 MB = 1024**2 bytes


class CapacityError(RuntimeError):
    """Raised when a tier cannot accept bytes after all legal evictions."""


@dataclass
class Tier:
    """Mutable state of one storage tier."""

    spec: TierSpec
    files: dict[int, int] = field(default_factory=dict)  # opaque id -> size
    dirty: set[int] = field(default_factory=set)  # needs flush before eviction
    pinned: set[int] = field(default_factory=set)  # eviction forbidden
    busy_until_ms: float = 0.0  # bandwidth serialization on the virtual clock

    def used_bytes(self) -> int:
        """Return the sum of sizes of all files currently held."""
        return sum(self.files.values())

    def free_bytes(self) -> Optional[int]:
        """Return remaining capacity, or ``None`` for the unlimited origin."""
        capacity = self.spec.capacity_bytes
        if capacity is None:
            return None
        return max(0, capacity - self.used_bytes())

    def holds(self, file_id: int) -> bool:
        """Return ``True`` if this tier holds a copy of ``file_id``."""
        return file_id in self.files

    def can_fit(self, size_bytes: int) -> bool:
        """Return ``True`` if ``size_bytes`` fits (or the tier is unlimited)."""
        free = self.free_bytes()
        return free is None or size_bytes <= free

    def add(self, file_id: int, size_bytes: int, *, dirty: bool = False) -> None:
        """Add or overwrite a file in this tier.

        Raises:
            CapacityError: If the file does not fit.
        """
        if not self.holds(file_id) and not self.can_fit(size_bytes):
            raise CapacityError(
                f"Tier {self.spec.tier_id} ({self.spec.name}) cannot fit "
                f"{size_bytes} bytes"
            )
        self.files[file_id] = size_bytes
        if dirty:
            self.dirty.add(file_id)
        else:
            self.dirty.discard(file_id)

    def remove(self, file_id: int) -> int:
        """Remove a file and return its size (does not check pin/dirty)."""
        size = self.files.pop(file_id)
        self.dirty.discard(file_id)
        return size

    def transfer_ms(self, size_bytes: int, other: "Tier") -> float:
        """Return transfer time between this tier and ``other`` in ms."""
        bandwidth = min(self.spec.bandwidth_mbps, other.spec.bandwidth_mbps)
        return size_bytes / (bandwidth * 1024.0)


def build_tiers(tier_config: list[dict]) -> dict[int, Tier]:
    """Build the tier dictionary from the ``tiers`` block of ``config.yaml``.

    Args:
        tier_config: List of tier dictionaries with keys ``id``, ``name``,
            ``capacity_mb`` (``None`` for unlimited), ``latency_ms`` and
            ``bandwidth_mbps``.

    Returns:
        Mapping from tier id to :class:`Tier`.

    Raises:
        ValueError: If ids are not exactly ``{0, 1, 2}`` or the origin is finite.
    """
    tiers: dict[int, Tier] = {}
    for entry in tier_config:
        capacity_mb = entry.get("capacity_mb")
        capacity_bytes = None if capacity_mb is None else int(float(capacity_mb) * MB)
        spec = TierSpec(
            tier_id=int(entry["id"]),
            name=str(entry["name"]),
            capacity_bytes=capacity_bytes,
            latency_ms=float(entry["latency_ms"]),
            bandwidth_mbps=float(entry["bandwidth_mbps"]),
        )
        tiers[spec.tier_id] = Tier(spec)

    if set(tiers) != {FAST, MID, SLOW}:
        raise ValueError("Tier ids must be exactly {0, 1, 2}")
    if tiers[SLOW].spec.capacity_bytes is not None:
        raise ValueError("Tier 2 (origin) must have unlimited capacity")
    return tiers
