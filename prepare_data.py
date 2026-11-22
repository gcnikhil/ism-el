"""Prepare the HuggingFace CIFAR-10 Arrow files into ContextTier artifacts.

The script:

1. prints the Arrow schema and one decoded sample (confirmation step),
2. decodes images lazily into ``uint8`` chunks of ``chunk_size_samples``,
3. writes a seeded synthetic difficulty array for virtual-mode hard examples,
4. builds a manifest whose integer ids are seeded-shuffled (opaque: no type or
   ordering information is encoded in an id), and
5. writes one metadata file and a small hot-cache file set.

Memory use stays bounded: at most one chunk of decoded samples is resident.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import struct
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image

from config import load_config, resolve_path

IMAGE_SHAPE: tuple[int, int, int] = (32, 32, 3)
IMAGE_BYTES: int = 32 * 32 * 3
LABEL_STRUCT = struct.Struct("<q")  # int64 labels, per schema
COUNT_STRUCT = struct.Struct("<I")


def open_arrow(path: Path) -> tuple[object, str]:
    """Open an Arrow file as an IPC stream, falling back to IPC file format.

    Args:
        path: Path to the ``.arrow`` file.

    Returns:
        Tuple of ``(reader, kind)`` where ``kind`` is ``"stream"`` or ``"file"``.

    Raises:
        RuntimeError: If neither reader can open the file.
    """
    try:
        return ipc.open_stream(str(path)), "stream"
    except (pa.ArrowInvalid, OSError, ValueError) as stream_error:
        try:
            return ipc.open_file(str(path)), "file"
        except Exception as file_error:  # noqa: BLE001 - report both failures
            raise RuntimeError(
                f"Could not open {path} as IPC stream or IPC file"
            ) from file_error


def iter_batches(path: Path) -> Iterator[pa.RecordBatch]:
    """Yield record batches from an Arrow file lazily."""
    reader, kind = open_arrow(path)
    if kind == "stream":
        yield from reader
    else:
        for index in range(reader.num_record_batches):
            yield reader.get_batch(index)


def decode_sample(row: dict) -> tuple[np.ndarray, int]:
    """Decode one Arrow row into a ``uint8`` RGB array and an int label."""
    payload = row["img"]["bytes"]
    with Image.open(io.BytesIO(payload)) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return array, int(row["label"])


def inspect_arrow(path: Path) -> None:
    """Print the schema and one decoded sample as the required sanity check."""
    reader, kind = open_arrow(path)
    print(f"Arrow reader for {path.name}: {kind}")
    print("Schema:")
    print(reader.schema)
    first_batch = next(iter_batches(path))
    row = first_batch.slice(0, 1).to_pylist()[0]
    array, label = decode_sample(row)
    print(
        "Decoded sample: "
        f"label={label}, shape={array.shape}, dtype={array.dtype}, "
        f"png_bytes={len(row['img']['bytes'])}"
    )


def write_chunk(path: Path, samples: list[tuple[np.ndarray, int]]) -> int:
    """Write decoded samples to a compact binary chunk file.

    Format: ``uint32 count`` followed by ``int64 label`` + ``uint8[3072]`` per
    sample. Returns the exact number of bytes written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(COUNT_STRUCT.pack(len(samples)))
        for array, label in samples:
            handle.write(LABEL_STRUCT.pack(label))
            handle.write(array.tobytes())
    return COUNT_STRUCT.size + len(samples) * (LABEL_STRUCT.size + IMAGE_BYTES)


def build_chunks(
    arrow_path: Path,
    out_dir: Path,
    chunk_size: int,
    prefix: str,
) -> list[dict]:
    """Decode an Arrow file into chunk files, one chunk held in memory at once."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    descriptors: list[dict] = []
    buffer: list[tuple[np.ndarray, int]] = []
    chunk_index = 0

    def flush_buffer() -> None:
        nonlocal buffer, chunk_index
        if not buffer:
            return
        filename = f"{prefix}_{chunk_index:05d}.bin"
        path = out_dir / filename
        size = write_chunk(path, buffer)
        descriptors.append(
            {
                "path": path,
                "size_bytes": size,
                "samples": len(buffer),
                "display_name": f"{prefix}_chunk_{chunk_index:05d}",
                "logical_index": chunk_index,
            }
        )
        chunk_index += 1
        buffer = []

    for batch in iter_batches(arrow_path):
        for row in batch.to_pylist():
            array, label = decode_sample(row)
            buffer.append((array, label))
            if len(buffer) == chunk_size:
                flush_buffer()
    flush_buffer()
    return descriptors


def build_difficulty(path: Path, n_samples: int, cfg: dict, seed: int) -> np.ndarray:
    """Draw seeded lognormal difficulties and save them as float32.

    The per-seen decay is applied at runtime by the workload generator, not
    here; this file stores the initial difficulty of each sample.
    """
    rng = np.random.default_rng(seed)
    scores = rng.lognormal(
        mean=float(cfg["lognormal_mu"]),
        sigma=float(cfg["lognormal_sigma"]),
        size=n_samples,
    ).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, scores)
    return scores


def write_metadata(
    path: Path,
    schema_text: str,
    train_descriptors: list[dict],
    validation_descriptors: list[dict],
    difficulty_cfg: dict,
    difficulty_seed: int,
    config: dict,
) -> int:
    """Write the single metadata file and return its exact size in bytes."""
    payload = {
        "dataset": "CIFAR-10 (HuggingFace Arrow)",
        "schema": schema_text,
        "train_rows": sum(item["samples"] for item in train_descriptors),
        "validation_rows": sum(item["samples"] for item in validation_descriptors),
        "train_chunks": len(train_descriptors),
        "validation_chunks": len(validation_descriptors),
        "chunk_size_samples": int(config["chunk_size_samples"]),
        "difficulty_model": {
            "distribution": "seeded lognormal with per-seen decay",
            "seed": difficulty_seed,
            **{key: float(value) for key, value in difficulty_cfg.items()},
        },
        "note": (
            "Scaled-down emulation. Difficulty decay and hard-example "
            "oversampling are applied by the workload generator at runtime."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def build_hot_cache(
    out_dir: Path,
    source_chunks: list[dict],
    difficulties: np.ndarray,
    chunk_size: int,
    count: int,
) -> list[dict]:
    """Copy the highest-difficulty chunks into a small hot-cache file set."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_mean_difficulty = []
    for descriptor in source_chunks:
        start = descriptor["logical_index"] * chunk_size
        end = min(start + chunk_size, difficulties.size)
        chunk_mean_difficulty.append(float(difficulties[start:end].mean()))

    ranked = sorted(
        range(len(source_chunks)),
        key=lambda index: chunk_mean_difficulty[index],
        reverse=True,
    )[:count]

    descriptors: list[dict] = []
    for rank, source_index in enumerate(ranked):
        source_path: Path = source_chunks[source_index]["path"]
        destination = out_dir / f"hot_{rank:04d}.bin"
        shutil.copyfile(source_path, destination)
        descriptors.append(
            {
                "path": destination,
                "size_bytes": source_path.stat().st_size,
                "display_name": f"hot_cache_{rank:04d}",
                "logical_index": rank,
            }
        )
    return descriptors


def build_manifest(
    manifest_path: Path,
    descriptors: list[dict],
    id_seed: int,
    ckpt_size_bytes: int,
    checkpoint_slots: int,
) -> dict:
    """Assign seeded opaque ids and write the evaluation-side manifest.

    Ids are a seeded permutation of ``0..N-1`` so id magnitude cannot encode
    file type. The manifest is never passed to a policy.
    """
    rng = np.random.default_rng(id_seed)

    all_descriptors: list[dict] = list(descriptors)
    for slot in range(checkpoint_slots):
        all_descriptors.append(
            {
                "path": None,
                "size_bytes": ckpt_size_bytes,
                "true_type": "checkpoint",
                "display_name": f"checkpoint_{slot:03d}",
                "logical_index": slot,
            }
        )

    # Permute ids over ALL files, including checkpoint slots, so that even
    # the numeric range of an id cannot reveal its true type.
    ids = rng.permutation(len(all_descriptors)).tolist()

    files = []
    for file_id, descriptor in zip(ids, all_descriptors):
        path = descriptor["path"]
        files.append(
            {
                "id": int(file_id),
                "path": None if path is None else path.relative_to(
                    resolve_path(".")
                ).as_posix(),
                "size_bytes": int(descriptor["size_bytes"]),
                "true_type": descriptor["true_type"],
                "display_name": descriptor["display_name"],
                "logical_index": descriptor["logical_index"],
            }
        )

    manifest = {
        "version": 1,
        "id_seed": id_seed,
        "file_count": len(files),
        "files": files,
        "true_type_vocabulary": [
            "chunk",
            "validation",
            "checkpoint",
            "metadata",
            "hot_cache",
        ],
        "warning": "EVALUATION ONLY - never expose true_type or display_name to policies",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    """Run the full preparation pipeline using ``config.yaml``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    train_path = resolve_path(config["data"]["train_arrow"])
    test_path = resolve_path(config["data"]["test_arrow"])
    artifacts_dir = resolve_path(config["data"]["artifacts_dir"])
    manifest_path = resolve_path(config["data"]["manifest"])
    metadata_path = resolve_path(config["data"]["metadata"])
    difficulty_path = resolve_path(config["data"]["difficulty"])
    chunk_size = int(config["chunk_size_samples"])
    seed = int(config["seeds"][0])

    # Required confirmation step: schema + one decoded sample BEFORE processing.
    inspect_arrow(train_path)
    inspect_arrow(test_path)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    train_descriptors = build_chunks(
        train_path, artifacts_dir / "train_chunks", chunk_size, "train"
    )
    validation_descriptors = build_chunks(
        test_path, artifacts_dir / "validation_chunks", chunk_size, "val"
    )

    difficulties = build_difficulty(
        difficulty_path,
        sum(item["samples"] for item in train_descriptors),
        config["difficulty"],
        seed,
    )
    hot_descriptors = build_hot_cache(
        artifacts_dir / "hot_cache",
        train_descriptors,
        difficulties,
        chunk_size,
        int(config["artifacts"]["hot_cache_files"]),
    )

    reader, _ = open_arrow(train_path)
    schema_text = str(reader.schema)
    metadata_size = write_metadata(
        metadata_path,
        schema_text,
        train_descriptors,
        validation_descriptors,
        config["difficulty"],
        seed,
        config,
    )

    descriptors: list[dict] = []
    for descriptor in train_descriptors:
        descriptors.append({**descriptor, "true_type": "chunk"})
    for descriptor in validation_descriptors:
        descriptors.append({**descriptor, "true_type": "validation"})
    for descriptor in hot_descriptors:
        descriptors.append({**descriptor, "true_type": "hot_cache"})
    descriptors.append(
        {
            "path": metadata_path,
            "size_bytes": metadata_size,
            "true_type": "metadata",
            "display_name": "metadata",
            "logical_index": 0,
        }
    )

    manifest = build_manifest(
        manifest_path,
        descriptors,
        int(config["artifacts"]["id_seed"]),
        int(float(config["ckpt_size_mb"]) * 1024 * 1024),
        int(config["artifacts"]["checkpoint_slots"]),
    )

    counts: dict[str, int] = {}
    for item in manifest["files"]:
        counts[item["true_type"]] = counts.get(item["true_type"], 0) + 1
    total_bytes = sum(item["size_bytes"] for item in manifest["files"])
    print(f"Wrote {manifest_path}")
    print(f"Files by true type: {counts}")
    print(f"Total declared bytes: {total_bytes / (1024 * 1024):.2f} MB")
    print("Note: ids are seeded-shuffled and carry no type information.")


if __name__ == "__main__":
    main()
