"""Real-mode dataset helpers: opaque chunk payloads to PyTorch tensors.

Only a handful of chunks are decoded at a time, keeping Python memory well
below the 3 GB budget.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from config import resolve_path
from workload import load_manifest

COUNT_STRUCT = struct.Struct("<I")
LABEL_STRUCT = struct.Struct("<q")
IMAGE_SHAPE = (32, 32, 3)
IMAGE_BYTES = 32 * 32 * 3


@dataclass(frozen=True)
class ChunkRecord:
    """Evaluation-side description of one opaque chunk file."""

    file_id: int
    path: Path
    size_bytes: int
    logical_index: int
    true_type: str


class ChunkTable:
    """Maps opaque file ids to real chunk files (harness-side only)."""

    def __init__(self, config: dict) -> None:
        """Build the id-to-path table from the prepared manifest."""
        manifest = load_manifest(config)
        self.entries: dict[int, ChunkRecord] = {}
        for item in manifest["files"]:
            if item["true_type"] not in {"chunk", "validation"} or not item["path"]:
                continue
            self.entries[int(item["id"])] = ChunkRecord(
                file_id=int(item["id"]),
                path=resolve_path(item["path"]),
                size_bytes=int(item["size_bytes"]),
                logical_index=int(item["logical_index"]),
                true_type=str(item["true_type"]),
            )
        self.all_ids = sorted(self.entries, key=lambda file_id: self.entries[file_id].logical_index)

    def train_ids(self, limit: int | None = None) -> list[int]:
        """Return opaque ids of training chunks, optionally truncated."""
        ids = [
            file_id
            for file_id in self.all_ids
            if self.entries[file_id].true_type == "chunk"
        ]
        return ids[:limit] if limit is not None else ids

    def validation_ids(self, limit: int | None = None) -> list[int]:
        """Return opaque ids of validation chunks, optionally truncated."""
        ids = [
            file_id
            for file_id in self.all_ids
            if self.entries[file_id].true_type == "validation"
        ]
        return ids[:limit] if limit is not None else ids


def decode_chunk(payload: bytes) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a prepared chunk's bytes into normalized images and labels.

    Args:
        payload: Binary chunk produced by ``prepare_data.write_chunk``.

    Returns:
        Tuple of ``(images, labels)`` with shapes ``(N, 3, 32, 32)`` float32
        normalized to ``[0, 1]`` and ``(N,)`` int64.
    """
    count = COUNT_STRUCT.unpack_from(payload, 0)[0]
    images = np.empty((count, *IMAGE_SHAPE), dtype=np.uint8)
    labels = np.empty(count, dtype=np.int64)
    offset = COUNT_STRUCT.size
    for index in range(count):
        labels[index] = LABEL_STRUCT.unpack_from(payload, offset)[0]
        offset += LABEL_STRUCT.size
        flat = np.frombuffer(
            payload, dtype=np.uint8, count=IMAGE_BYTES, offset=offset
        )
        images[index] = flat.reshape(IMAGE_SHAPE)
        offset += IMAGE_BYTES
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255.0)
    return tensor, torch.from_numpy(labels.copy())


def load_chunk(table: ChunkTable, file_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Read and decode one chunk from its origin file."""
    return decode_chunk(table.entries[file_id].path.read_bytes())


def read_json(path: str | Path) -> dict:
    """Read a UTF-8 JSON file (used by the real-mode rank report)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
