"""Measure the real CPU tiny-CNN compute step time used by ``train.py``.

This script reproduces the timed compute block from ``train.run_real_training``
(the block that produces ``compute_ms``): it concatenates the decoded chunks
for one step, runs forward + backward + an Adam optimizer step, and records
per-step wall-clock milliseconds with ``time.perf_counter``.

No storage I/O, tier simulation, or logging is included -- this is a *pure
compute* measurement intended to justify ``compute_ms`` in config.yaml.

Note on batch composition: ``train._epoch_steps`` reads
``ceil(batch_size / chunk_size)`` chunks per step.  With ``batch_size=128`` and
``chunk_size_samples=50`` that is 3 chunks = 150 samples, which is the tensor
actually fed to the model.  Both the nominal batch size and the effective
sample count are recorded in the report.

Only two new files are produced by this work:
  * this script
  * results/real/step_time_benchmark.json
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from config import load_config
from train import TinyCNN

SEED = 0
MEASURED_STEPS = 200
WARMUP_STEPS = 10  # matches train.WARMUP_STEPS
OUTPUT_PATH = Path("results/real/step_time_benchmark.json")


def main() -> None:
    """Run the benchmark and write the JSON report."""
    config = load_config("config.yaml")
    batch_size = int(config["batch_size"])
    chunk_size = int(config["chunk_size_samples"])
    chunks_per_step = math.ceil(batch_size / chunk_size)
    effective_samples = chunks_per_step * chunk_size

    # train.py applies this cap before any training; mirror it so the timing
    # reflects the real run environment.
    threads_initial = torch.get_num_threads()
    torch.set_num_threads(min(2, threads_initial))
    threads_used = torch.get_num_threads()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Import the exact architecture from train.py (no redefinition/drift).
    model = TinyCNN()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(reduction="none")

    # Build the decoded chunks the way dataset.decode_chunk does:
    # uint8 (50, 32, 32, 3) -> permute -> float / 255 => (50, 3, 32, 32).
    rng = np.random.default_rng(SEED)
    image_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    for _ in range(chunks_per_step):
        raw = rng.integers(0, 256, size=(chunk_size, 32, 32, 3), dtype=np.uint8)
        images = torch.from_numpy(raw).permute(0, 3, 1, 2).float().div_(255.0)
        labels = torch.from_numpy(rng.integers(0, 10, size=chunk_size, dtype=np.int64))
        image_chunks.append(images)
        label_chunks.append(labels)

    def compute_step() -> float:
        """Time one forward+backward+optimizer step; mirrors train.py exactly."""
        start = time.perf_counter()
        images = torch.cat(image_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        per_sample = criterion(logits, labels)
        loss = per_sample.mean()
        loss.backward()
        optimizer.step()
        return (time.perf_counter() - start) * 1000.0

    for _ in range(WARMUP_STEPS):
        compute_step()

    per_step_ms = [compute_step() for _ in range(MEASURED_STEPS)]
    arr = np.asarray(per_step_ms, dtype=np.float64)

    report = {
        "measurement": {
            "description": (
                "Pure CPU forward+backward+Adam step from train.py timed with "
                "time.perf_counter(); no storage I/O or tier simulation."
            ),
            "median_ms": float(np.median(arr)),
            "mean_ms": float(np.mean(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "steps_measured": MEASURED_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "per_step_ms": per_step_ms,
        },
        "configuration": {
            "batch_size": batch_size,
            "chunk_size_samples": chunk_size,
            "chunks_per_step": chunks_per_step,
            "effective_batch_samples": effective_samples,
            "image_shape_uint8": [32, 32, 3],
            "model_input_shape": [effective_samples, 3, 32, 32],
            "model": (
                "TinyCNN from train.py (conv1 3->16 k3 p1; "
                "conv2 16->32 k3 p1; flatten; fc 32768->10)"
            ),
            "optimizer": "Adam(lr=1e-3)",
            "criterion": "CrossEntropyLoss(reduction='none') followed by .mean()",
            "seed": SEED,
            "device": "cpu",
        },
        "environment": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cpu_count": os.cpu_count(),
            "torch_num_threads": threads_used,
            "torch_num_threads_initial": threads_initial,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"median_ms: {report['measurement']['median_ms']!r}")
    print(f"mean_ms:   {report['measurement']['mean_ms']!r}")
    print(f"p95_ms:    {report['measurement']['p95_ms']!r}")
    print(f"p99_ms:    {report['measurement']['p99_ms']!r}")
    print(f"min_ms:    {report['measurement']['min_ms']!r}")
    print(f"max_ms:    {report['measurement']['max_ms']!r}")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
