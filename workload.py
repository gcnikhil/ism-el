"""Deterministic workload generation for scenarios A-F.

The whole access stream is generated up front so that:

* the simulator can hand the full read stream to the Belady oracle, and
* Scenario C can make hard-example oversampling depend on decaying synthetic
  difficulty (training state), not on the shuffle seed alone.

Policies never see this module's difficulty data or the manifest's true types.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from config import resolve_path
from policies import AccessEvent

WARMUP_STEPS: int = 10


@dataclass(frozen=True)
class Workload:
    """A pre-generated sequence of training steps."""

    scenario: str
    steps: list[list[AccessEvent]]
    read_stream: list[int]
    chunk_ids: frozenset[int]
    metadata: dict
    temperature_by_epoch: dict[tuple[int, int], str]

    def temperature_for(self, event: AccessEvent) -> str:
        """Return evaluation-only temperature without exposing it to policies."""
        return self.temperature_by_epoch.get((event.epoch, event.file_id), "cold")


class StreamOracle:
    """Future read order over opaque ids, with a simulator-driven cursor."""

    def __init__(self, read_stream: Iterable[int], chunk_ids: Iterable[int]) -> None:
        """Index a read stream for clairvoyant/offline policies."""
        self.stream = list(read_stream)
        self.chunk_ids = frozenset(chunk_ids)
        self._positions: dict[int, list[int]] = defaultdict(list)
        for index, file_id in enumerate(self.stream):
            self._positions[file_id].append(index)
        self._cursor = 0

    def observe(self, file_id: int) -> None:
        """Advance the cursor past the current occurrence of ``file_id``."""
        for position in self._positions.get(file_id, []):
            if position >= self._cursor:
                self._cursor = position + 1
                return
        self._cursor += 1

    def upcoming(self, n: int, *, after: int | None = None) -> list[int]:
        """Return the next ``n`` opaque ids in the read stream."""
        start = self._cursor if after is None else after
        return self.stream[start : start + n]

    def upcoming_chunks(self, n: int, *, after: int | None = None) -> list[int]:
        """Return the next ``n`` ids that belong to the seeded chunk stream."""
        start = self._cursor if after is None else after
        result: list[int] = []
        for file_id in self.stream[start:]:
            if file_id in self.chunk_ids:
                result.append(file_id)
                if len(result) == n:
                    break
        return result

    def next_use(self, file_id: int) -> int | None:
        """Return relative distance to the next use, or ``None`` if absent."""
        for position in self._positions.get(file_id, []):
            if position >= self._cursor:
                return position - self._cursor
        return None

    def length(self) -> int:
        """Return the total number of reads in the stream."""
        return len(self.stream)


def load_manifest(config: dict) -> dict:
    """Load the evaluation-only manifest produced by ``prepare_data.py``."""
    path = resolve_path(config["data"]["manifest"])
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Run `python prepare_data.py` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def files_by_type(manifest: dict) -> dict[str, list[dict]]:
    """Group manifest entries by their evaluation-only true type."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in manifest["files"]:
        grouped[entry["true_type"]].append(entry)
    return dict(grouped)


def _sample_order(
    epoch: int,
    chunk_count: int,
    difficulty: np.ndarray,
    chunk_size: int,
    config: dict,
    rng: np.random.Generator,
) -> list[int]:
    """Build one epoch's chunk order, optionally oversampling hard chunks."""
    order = rng.permutation(chunk_count).tolist()
    if epoch == 0:
        return order

    difficulty_cfg = config["difficulty"]
    fraction = float(difficulty_cfg["oversample_fraction"])
    hard_count = max(1, int(difficulty.size * fraction))
    hard_samples = np.argpartition(difficulty, -hard_count)[-hard_count:]
    hard_chunks = sorted({int(index) // chunk_size for index in hard_samples})
    # Each hard chunk appears once extra; this depends on decaying difficulty,
    # i.e. on training state, so the shuffle seed alone cannot predict it.
    order.extend(hard_chunks)
    rng.shuffle(order)
    return order


def _temperature_groups(
    chunk_ids: list[int], config: dict, seed: int, epoch: int, scenario: str
) -> dict[int, str]:
    """Assign synthetic hot/warm/cold labels; F rotates them at configured epochs."""
    settings = config["temperature"]
    rotation = epoch // max(1, int(settings["drift_every_epochs"])) if scenario == "F" else 0
    rng = np.random.default_rng(seed * 1009 + rotation)
    ordered = rng.permutation(chunk_ids).tolist()
    hot_end = max(1, int(len(ordered) * float(settings["hot_frac"])))
    warm_end = min(len(ordered), hot_end + int(len(ordered) * float(settings["warm_frac"])))
    return {
        file_id: "hot" if index < hot_end else "warm" if index < warm_end else "cold"
        for index, file_id in enumerate(ordered)
    }


def _temperature_order(
    chunk_ids: list[int], groups: dict[int, str], config: dict, rng: np.random.Generator
) -> list[int]:
    """Sample one epoch with configured aggregate hot/warm/cold shares."""
    settings = config["temperature"]
    shares = {"hot": float(settings["hot_share"]), "warm": float(settings["warm_share"])}
    shares["cold"] = max(0.0, 1.0 - shares["hot"] - shares["warm"])
    weights = np.asarray([shares[groups[file_id]] for file_id in chunk_ids], dtype=np.float64)
    weights /= weights.sum()
    return rng.choice(chunk_ids, size=len(chunk_ids), replace=True, p=weights).astype(int).tolist()


def generate_workload(
    config: dict,
    scenario: str,
    seed: int,
    epochs: int | None = None,
    chunk_limit: int | None = None,
) -> Workload:
    """Generate the complete workload for one scenario and seed.

    Scenarios:

    * ``A``: shuffled training chunks + per-step metadata only.
    * ``B``: A + checkpoints + an end-of-epoch validation burst.
    * ``C``: B + state-dependent hard-example oversampling (order not seeded).
    * ``D``: B with a fast tier large enough to hold everything (handled by
       the simulator); the access stream matches B.
    * ``E``: synthetic non-uniform hot/warm/cold chunk sampling.
    * ``F``: E with a rotating hot set (synthetic temperature drift).
    """
    if scenario not in {"A", "B", "C", "D", "E", "F"}:
        raise ValueError(f"Unknown scenario: {scenario}")

    manifest = load_manifest(config)
    grouped = files_by_type(manifest)
    chunks = grouped["chunk"]
    if chunk_limit is not None:
        chunks = chunks[:chunk_limit]
    validation = grouped["validation"]
    metadata = grouped["metadata"][0]
    checkpoints = grouped["checkpoint"]

    chunk_size = int(config["chunk_size_samples"])
    batch_size = int(config["batch_size"])
    steps_per_epoch = math.ceil((len(chunks) * chunk_size) / batch_size)
    chunks_per_step = math.ceil(batch_size / chunk_size)
    checkpoint_every = max(1, int(config["checkpoint_every_steps"]))
    total_epochs = int(config["epochs"] if epochs is None else epochs)

    if scenario == "C":
        difficulty_path = resolve_path(config["data"]["difficulty"])
        if not difficulty_path.is_file():
            from prepare_data import build_difficulty
            difficulty = build_difficulty(
                difficulty_path,
                len(chunks) * chunk_size,
                config["difficulty"],
                int(config["seeds"][0]),
            )
        else:
            difficulty = np.load(difficulty_path).astype(np.float32).copy()
    else:
        difficulty = np.zeros(len(chunks) * chunk_size, dtype=np.float32)

    rng = np.random.default_rng(seed)
    steps: list[list[AccessEvent]] = []
    read_stream: list[int] = []
    checkpoint_cursor = 0
    include_extras = scenario in {"B", "C", "D", "E", "F"}
    temperature_by_epoch: dict[tuple[int, int], str] = {}

    for epoch in range(total_epochs):
        if scenario in {"E", "F"}:
            ids = [int(entry["id"]) for entry in chunks]
            groups = _temperature_groups(ids, config, seed, epoch, scenario)
            temperature_by_epoch.update(
                {(epoch, file_id): label for file_id, label in groups.items()}
            )
            # Event construction below expects indexes into ``chunks``.
            index_by_id = {int(entry["id"]): index for index, entry in enumerate(chunks)}
            order = [index_by_id[file_id] for file_id in _temperature_order(ids, groups, config, rng)]
        else:
            order = _sample_order(epoch, len(chunks), difficulty, chunk_size, config, rng)

        for step in range(steps_per_epoch):
            phase = "warmup" if epoch == 0 and step < WARMUP_STEPS else "train"
            accesses: list[AccessEvent] = []
            for offset in range(chunks_per_step):
                chunk_index = order[(step * chunks_per_step + offset) % len(order)]
                entry = chunks[chunk_index]
                accesses.append(
                    AccessEvent(
                        file_id=int(entry["id"]),
                        size_bytes=int(entry["size_bytes"]),
                        op="read",
                        epoch=epoch,
                        step=step,
                        phase=phase,
                    )
                )
            accesses.append(
                AccessEvent(
                    file_id=int(metadata["id"]),
                    size_bytes=int(metadata["size_bytes"]),
                    op="read",
                    epoch=epoch,
                    step=step,
                    phase=phase,
                )
            )
            if include_extras and (step + 1) % checkpoint_every == 0:
                entry = checkpoints[checkpoint_cursor % len(checkpoints)]
                checkpoint_cursor += 1
                accesses.append(
                    AccessEvent(
                        file_id=int(entry["id"]),
                        size_bytes=int(entry["size_bytes"]),
                        op="write",
                        epoch=epoch,
                        step=step,
                        phase="checkpoint",
                    )
                )
            steps.append(accesses)
            read_stream.extend(event.file_id for event in accesses if event.op == "read")

        if include_extras:
            validation_accesses = [
                AccessEvent(
                    file_id=int(entry["id"]),
                    size_bytes=int(entry["size_bytes"]),
                    op="read",
                    epoch=epoch,
                    step=steps_per_epoch,
                    phase="validation",
                )
                for entry in validation
            ]
            steps.append(validation_accesses)
            read_stream.extend(event.file_id for event in validation_accesses)

        if scenario == "C":
            decay = float(config["difficulty"]["decay_per_seen"])
            for chunk_index in order:
                start = chunk_index * chunk_size
                end = min(start + chunk_size, difficulty.size)
                difficulty[start:end] *= decay

    return Workload(
        scenario=scenario,
        steps=steps,
        read_stream=read_stream,
        chunk_ids=frozenset(int(entry["id"]) for entry in chunks),
        metadata=manifest,
        temperature_by_epoch=temperature_by_epoch,
    )
