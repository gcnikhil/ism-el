"""Real-mode CPU training through ``TieredStore`` and the rank check.

A tiny CNN reads real decoded CIFAR chunks through the same tiered store used
by virtual mode. Tier delays are realized with ``time.sleep`` via ``RealClock``.
The training loop measures wall-clock data-wait and writes the exact trace
schema, so real runs are directly comparable to virtual runs.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch import nn

from config import load_config, resolve_path
from dataset import ChunkTable, decode_chunk
from logger import Logger
from policies import (
    AccessEvent,
    PhaseContext,
    POLICY_CLASSES,
    TickContext,
)
from simulator import StoreView, _apply_actions, _make_policy, _tiers_for_run, run_simulation
from store import PayloadStore, RealClock
from workload import StreamOracle, files_by_type, load_manifest

WARMUP_STEPS = 10


class TinyCNN(nn.Module):
    """Two convolution layers plus one fully connected layer (~333K params)."""

    def __init__(self) -> None:
        """Build the small CPU-friendly network."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32 * 32, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of normalized images."""
        hidden = torch.relu(self.conv1(images))
        hidden = torch.relu(self.conv2(hidden))
        return self.fc(self.flatten(hidden))


class TinyMLP(nn.Module):
    """Flattened CIFAR classifier with roughly 0.8M trainable parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(3 * 32 * 32, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.output(torch.relu(self.hidden(self.flatten(images))))


MODEL_REGISTRY: dict[str, Callable[[], nn.Module]] = {
    "cnn": TinyCNN,
    "mlp": TinyMLP,
}


def available_models() -> tuple[str, ...]:
    """Return the stable list of selectable real-training model names."""
    return tuple(sorted(MODEL_REGISTRY))


def build_model(name: str) -> nn.Module:
    """Construct a registered model, with an actionable unknown-name error."""
    normalized = str(name).lower()
    try:
        return MODEL_REGISTRY[normalized]()
    except KeyError as error:
        choices = ", ".join(available_models())
        raise ValueError(f"Unknown model {name!r}. Available models: {choices}") from error


def model_name(config: dict[str, Any]) -> str:
    """Read and validate the selected model from a parsed configuration."""
    selected = str(config.get("model", {}).get("name", "cnn")).lower()
    # Validate here so virtual-only invocations get the same useful error.
    build_model(selected)
    return selected


def apply_model_selection(
    config: dict[str, Any],
    *,
    name: str | None = None,
    compute_source: str | None = None,
) -> dict[str, Any]:
    """Return a copied config with model/profile-derived virtual knobs applied.

    ``manual`` leaves the historical ``compute_ms`` and ``ckpt_size_mb`` values
    untouched.  ``profile`` reads those values from ``profiles.json``.  This
    deliberately avoids mutating a caller's shared config during sweeps.
    """
    selected = str(name or config.get("model", {}).get("name", "cnn")).lower()
    build_model(selected)
    updated = copy.deepcopy(config)
    block = dict(updated.get("model", {}))
    source = str(compute_source or block.get("compute_source", "manual")).lower()
    if source not in {"manual", "profile"}:
        raise ValueError("model.compute_source must be 'manual' or 'profile'")
    block.update({"name": selected, "compute_source": source})
    updated["model"] = block
    # Learned artefacts are model-specific.  Preserve legacy CNN paths exactly;
    # an MLP invocation cannot silently load a classifier/embedding/RL agent
    # trained from CNN traces.
    if selected != "cnn":
        for component in ("classifier", "embeddings", "rl"):
            settings = dict(updated.get(component, {}))
            if "model_path" in settings:
                settings["model_path"] = (
                    f"results/models/{selected}/{component}/{Path(str(settings['model_path'])).name}"
                )
            if "report_dir" in settings:
                settings["report_dir"] = f"results/models/{selected}/{component}"
            updated[component] = settings
    if source == "profile":
        profile_path = resolve_path(block.get("profile_path", "results/models/profiles.json"))
        if not profile_path.is_file():
            raise FileNotFoundError(
                f"Model profile not found: {profile_path}. Run `python train.py --profile-models` first."
            )
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        if "profiles" in payload:
            validate_profiles(payload)
        profiles = payload.get("profiles", payload)
        if selected not in profiles:
            raise ValueError(f"Profile {selected!r} missing from {profile_path}")
        profile = profiles[selected]
        updated["compute_ms"] = float(profile["median_step_ms"])
        updated["ckpt_size_mb"] = float(profile["checkpoint_bytes"]) / (1024 * 1024)
    return updated


def checkpoint_bytes(model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """Return serialized weights plus optimizer state bytes without a temp file."""
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
    return buffer.tell()


def profile_models(
    *,
    batch_size: int = 128,
    warmup_steps: int = 20,
    timed_steps: int = 200,
) -> dict[str, Any]:
    """Measure all registry entries on CPU and return a JSON-safe profile payload."""
    if batch_size <= 0 or warmup_steps < 0 or timed_steps <= 0:
        raise ValueError("batch_size/timed_steps must be positive and warmup_steps non-negative")
    try:
        import psutil
    except ImportError:  # pragma: no cover - requirements include psutil in normal runs
        psutil = None
    torch.set_num_threads(min(2, torch.get_num_threads()))
    process = psutil.Process() if psutil is not None else None
    profiles: dict[str, dict[str, float | int]] = {}
    for name in available_models():
        model = build_model(name)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        images = torch.rand(batch_size, 3, 32, 32)
        labels = torch.randint(0, 10, (batch_size,))

        def step() -> None:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        for _ in range(warmup_steps):
            step()
        samples: list[float] = []
        peak_ram = process.memory_info().rss if process is not None else 0
        for _ in range(timed_steps):
            before = time.perf_counter()
            step()
            samples.append((time.perf_counter() - before) * 1000.0)
            if process is not None:
                peak_ram = max(peak_ram, process.memory_info().rss)
        profiles[name] = {
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "median_step_ms": float(np.median(samples)),
            "p95_step_ms": float(np.percentile(samples, 95)),
            "checkpoint_bytes": checkpoint_bytes(model, optimizer),
            "peak_ram_bytes": int(peak_ram),
        }
    return {
        "schema_version": 1,
        "device": "cpu",
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "profiles": profiles,
    }


def validate_profiles(payload: dict[str, Any]) -> None:
    """Validate the portable model-profile schema before it is consumed."""
    required_top = {"schema_version", "device", "batch_size", "warmup_steps", "timed_steps", "profiles"}
    missing = required_top.difference(payload)
    if missing:
        raise ValueError(f"Invalid model profile: missing top-level keys {sorted(missing)}")
    profiles = payload["profiles"]
    if not isinstance(profiles, dict):
        raise ValueError("Invalid model profile: profiles must be an object")
    required_profile = {
        "parameter_count", "median_step_ms", "p95_step_ms", "checkpoint_bytes", "peak_ram_bytes"
    }
    for name in available_models():
        if name not in profiles:
            raise ValueError(f"Invalid model profile: missing registry model {name!r}")
        missing = required_profile.difference(profiles[name])
        if missing:
            raise ValueError(f"Invalid model profile {name!r}: missing keys {sorted(missing)}")
        if float(profiles[name]["median_step_ms"]) <= 0 or int(profiles[name]["checkpoint_bytes"]) <= 0:
            raise ValueError(f"Invalid model profile {name!r}: timings and checkpoint bytes must be positive")


@dataclass
class RealResult:
    """Outcome of one real-mode run."""

    metrics: dict[str, float]
    logger: Logger
    store: PayloadStore
    clock: RealClock


def _epoch_order(
    epoch: int,
    chunk_count: int,
    rng: np.random.Generator,
    config: dict,
    scenario: str,
    losses: np.ndarray | None,
    chunk_size: int,
) -> list[int]:
    """Build an epoch's chunk order, using real losses for Scenario C."""
    order = rng.permutation(chunk_count).tolist()
    if scenario != "C" or epoch == 0 or losses is None:
        return order
    if losses.size < chunk_count * chunk_size:
        return order
    per_chunk = losses[: chunk_count * chunk_size].reshape(chunk_count, chunk_size)
    mean_loss = per_chunk.mean(axis=1)
    hard_count = max(1, int(chunk_size * float(config["difficulty"]["oversample_fraction"])))
    hard_count = min(hard_count, chunk_count)
    hard = np.argsort(mean_loss)[-hard_count:].tolist()
    order.extend(hard)
    rng.shuffle(order)
    return order


def _epoch_steps(
    epoch: int,
    order: list[int],
    chunk_ids: list[int],
    metadata_id: int,
    checkpoint_ids: list[int],
    val_ids: list[int],
    config: dict,
    scenario: str,
    checkpoint_cursor: int,
) -> tuple[list[list[AccessEvent]], int]:
    """Generate the read/write events for one epoch (chunk-level shuffle)."""
    chunk_size = int(config["chunk_size_samples"])
    batch_size = int(config["batch_size"])
    steps_per_epoch = math.ceil(len(chunk_ids) * chunk_size / batch_size)
    chunks_per_step = math.ceil(batch_size / chunk_size)
    checkpoint_every = max(1, int(config["checkpoint_every_steps"]))
    include_extras = scenario in {"B", "C", "D"}
    steps: list[list[AccessEvent]] = []

    sizes = {file_id: 0 for file_id in chunk_ids}

    for step in range(steps_per_epoch):
        phase = "warmup" if epoch == 0 and step < WARMUP_STEPS else "train"
        accesses: list[AccessEvent] = []
        for offset in range(chunks_per_step):
            chunk_index = order[(step * chunks_per_step + offset) % len(order)]
            file_id = chunk_ids[chunk_index]
            accesses.append(
                AccessEvent(
                    file_id=file_id,
                    size_bytes=sizes[file_id],
                    op="read",
                    epoch=epoch,
                    step=step,
                    phase=phase,
                )
            )
        accesses.append(
            AccessEvent(
                file_id=metadata_id,
                size_bytes=sizes.get(metadata_id, 0),
                op="read",
                epoch=epoch,
                step=step,
                phase=phase,
            )
        )
        if include_extras and (step + 1) % checkpoint_every == 0:
            file_id = checkpoint_ids[checkpoint_cursor % len(checkpoint_ids)]
            checkpoint_cursor += 1
            accesses.append(
                AccessEvent(
                    file_id=file_id,
                    size_bytes=sizes.get(file_id, 0),
                    op="write",
                    epoch=epoch,
                    step=step,
                    phase="checkpoint",
                )
            )
        steps.append(accesses)

    if include_extras:
        steps.append(
            [
                AccessEvent(
                    file_id=file_id,
                    size_bytes=0,
                    op="read",
                    epoch=epoch,
                    step=steps_per_epoch,
                    phase="validation",
                )
                for file_id in val_ids
            ]
        )
    return steps, checkpoint_cursor


def run_real_training(
    config: dict,
    policy_name: str,
    scenario: str,
    seed: int,
    *,
    epochs: int | None = None,
    subset_images: int | None = None,
) -> RealResult:
    """Train the selected registered model through a real tiered store.

    Args:
        config: Parsed config dictionary.
        policy_name: Lower-case policy key.
        scenario: Scenario label.
        seed: RNG seed.
        epochs: Optional epoch override.
        subset_images: Optional image-count override for the CPU subset.

    Returns:
        A :class:`RealResult` with wall-clock metrics and a full trace.
    """
    config = apply_model_selection(config)
    selected_model = model_name(config)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    policy_name = policy_name.lower()
    if policy_name not in POLICY_CLASSES:
        raise ValueError(f"Unknown policy: {policy_name}")

    manifest = load_manifest(config)
    truth = {int(entry["id"]): str(entry["true_type"]) for entry in manifest["files"]}
    sizes = {int(entry["id"]): int(entry["size_bytes"]) for entry in manifest["files"]}
    grouped = files_by_type(manifest)
    metadata_id = int(grouped["metadata"][0]["id"])
    checkpoint_ids = [int(entry["id"]) for entry in grouped["checkpoint"]]

    table = ChunkTable(config)
    chunk_size = int(config["chunk_size_samples"])
    configured_subset = config.get("real_subset_images")
    limit = subset_images if subset_images is not None else configured_subset
    chunk_limit = math.ceil(limit / chunk_size) if limit else None
    train_ids = table.train_ids(limit=chunk_limit)
    val_ids = table.validation_ids(
        limit=int(config["real_validation_chunks"])
        if config.get("real_validation_chunks")
        else None
    )
    total_epochs = int(config["real_epochs"] if epochs is None else epochs)

    tiers = _tiers_for_run(config, scenario, policy_name, manifest)
    clock = RealClock()
    origin_paths = {
        int(entry["id"]): resolve_path(entry["path"])
        for entry in manifest["files"]
        if entry["path"]
    }
    store = PayloadStore(
        tiers, clock, origin_paths, max_queue=int(config["max_prefetch_queue"])
    )
    logger = Logger(
        run_id=f"real-{scenario}-{policy_name}-seed{seed}",
        seed=seed,
        scenario=scenario,
        policy=policy_name,
    )
    for file_id, size in sizes.items():
        store.register(file_id, size)
    if policy_name == "allfast" or scenario == "D":
        for file_id, size in sizes.items():
            store.tiers[0].add(file_id, size)

    rng = np.random.default_rng(seed)
    chunk_ids_set = set(train_ids)
    val_ids_set = set(val_ids)
    oracle_stream: list[int] = []
    for epoch in range(total_epochs):
        order = _epoch_order(epoch, len(train_ids), rng, config, scenario, None, chunk_size)
        steps_e, _ = _epoch_steps(
            epoch, order, train_ids, metadata_id, checkpoint_ids, val_ids,
            config, scenario, 0,
        )
        for accesses in steps_e:
            oracle_stream.extend(
                event.file_id for event in accesses if event.op == "read"
            )

    oracle: StreamOracle | None = None
    if policy_name in {"clairvoyant", "belady"}:
        oracle = StreamOracle(oracle_stream, chunk_ids_set)
    declared_types = (
        {int(entry["id"]): str(entry["true_type"]) for entry in manifest["files"]}
        if policy_name.startswith("context_rules")
        else None
    )
    classifier_model = None
    if policy_name == "classifier":
        from classifier import load_model

        classifier_model = load_model(config)
    elif policy_name == "embedding":
        from embeddings import load_predictor

        classifier_model = load_predictor(config)
    rl_agent = reuse_predictor = None
    if policy_name == "full_context":
        from simulator import _load_full_context_models

        rl_agent, classifier_model, reuse_predictor = _load_full_context_models(
            config
        )
    policy = _make_policy(
        config,
        policy_name,
        scenario,
        oracle,
        declared_types,
        classifier_model,
        rl_agent,
        reuse_predictor,
    )
    view = StoreView(store, getattr(policy, "history", {}))

    model = build_model(selected_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(reduction="none")
    sample_losses = np.zeros(len(train_ids) * chunk_size, dtype=np.float32)

    total_data_wait_ms = 0.0
    total_samples = 0
    reads = 0
    fast_hits = 0
    tier_served: dict[int, int] = {0: 0, 1: 0, 2: 0}
    latencies: list[float] = []
    current_phase = "warmup"
    checkpoint_payload = bytes(int(float(config["ckpt_size_mb"]) * 1024 * 1024))

    try:
        checkpoint_cursor = 0
        for epoch in range(total_epochs):
            order = _epoch_order(
                epoch, len(train_ids), rng, config, scenario,
                sample_losses if epoch > 0 else None, chunk_size,
            )
            steps, checkpoint_cursor = _epoch_steps(
                epoch, order, train_ids, metadata_id, checkpoint_ids, val_ids,
                config, scenario, checkpoint_cursor,
            )
            model.train()
            for accesses in steps:
                epoch_index = accesses[0].epoch
                step_index = accesses[0].step
                data_time_ms = 0.0
                images_batch: list[torch.Tensor] = []
                labels_batch: list[torch.Tensor] = []
                sample_id_batch: list[np.ndarray] = []

                for event in accesses:
                    if event.phase != current_phase:
                        policy.on_phase_change(
                            PhaseContext(
                                now_ms=clock.now_ms(),
                                epoch=epoch_index,
                                previous_phase=current_phase,
                                new_phase=event.phase,
                            ),
                            view,
                        )
                        current_phase = event.phase

                    start = time.perf_counter()
                    if event.op == "read":
                        # Arrival-time Tier-0 residency, before demand promotion.
                        arrival_fast = store.fast_resident(event.file_id)
                        actions = policy.on_access(event, view)
                        _apply_actions(actions, store, logger, clock)
                        result = store.get(event.file_id, event, arrival_fast=arrival_fast)
                        payload = store.read_payload(event.file_id, result.tier_served)
                        storage_timing_end = time.perf_counter()
                        reads += 1
                        fast_hits += int(result.fast_hit)
                        tier_served[result.tier_served] = tier_served.get(result.tier_served, 0) + 1
                        latencies.append(result.latency_ms)
                        logger.log_access(
                            ts=clock.now_ms(),
                            file_id=event.file_id,
                            size_bytes=event.size_bytes or len(payload),
                            op="read",
                            epoch=event.epoch,
                            step=event.step,
                            phase=event.phase,
                            tier_served=result.tier_served,
                            latency_ms=result.latency_ms,
                            wait_ms=result.wait_ms,
                            fast_hit=result.fast_hit,
                            true_type=truth[event.file_id],
                        )
                        if oracle is not None:
                            oracle.observe(event.file_id)
                        if event.file_id in chunk_ids_set or event.file_id in val_ids_set:
                            images, labels = decode_chunk(payload)
                            if event.phase in {"train", "warmup"}:
                                images_batch.append(images)
                                labels_batch.append(labels)
                                sample_start = table.entries[event.file_id].logical_index * chunk_size
                                sample_id_batch.append(
                                    np.arange(
                                        sample_start, sample_start + images.shape[0], dtype=np.int64
                                    )
                                )
                            elif event.phase == "validation":
                                model.eval()
                                with torch.no_grad():
                                    logits = model(images)
                                del logits
                                model.train()
                        data_time_ms += (storage_timing_end - start) * 1000.0
                    else:
                        landing = int(policy.on_write(event, view))
                        store.put(event, landing, dirty=(landing != 2))
                        store.write_payload(event.file_id, checkpoint_payload)
                        clock.advance(
                            sizes[event.file_id] / (tiers[landing].spec.bandwidth_mbps * 1024.0)
                            + tiers[landing].spec.latency_ms
                        )
                        logger.log_access(
                            ts=clock.now_ms(),
                            file_id=event.file_id,
                            size_bytes=sizes[event.file_id],
                            op="write",
                            epoch=event.epoch,
                            step=event.step,
                            phase=event.phase,
                            tier_served=landing,
                            latency_ms=0.0,
                            wait_ms=0.0,
                            fast_hit=False,
                            true_type=truth[event.file_id],
                        )
                        data_time_ms += (time.perf_counter() - start) * 1000.0

                tick_ctx = TickContext(
                    now_ms=clock.now_ms(),
                    epoch=epoch_index,
                    step=step_index,
                    phase=current_phase,
                )
                store.tick(tick_ctx)
                _apply_actions(policy.on_tick(tick_ctx, view), store, logger, clock)

                if current_phase in {"train", "warmup"} and images_batch:
                    compute_start = time.perf_counter()
                    images = torch.cat(images_batch, dim=0)
                    labels = torch.cat(labels_batch, dim=0)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    per_sample = criterion(logits, labels)
                    loss = per_sample.mean()
                    loss.backward()
                    optimizer.step()
                    compute_ms = (time.perf_counter() - compute_start) * 1000.0
                    del compute_start, compute_ms  # training is for realism; overlap uses config
                    batch_loss = per_sample.detach().numpy()
                    sample_ids = np.concatenate(sample_id_batch)
                    valid = sample_ids < sample_losses.size
                    sample_losses[sample_ids[valid]] = batch_loss[valid]
                    total_samples += int(labels.numel())
                    # Use the configured accelerator-overlap budget in both
                    # modes so data-wait has the same definition everywhere.
                    total_data_wait_ms += max(
                        0.0, data_time_ms - float(config["compute_ms"])
                    )

            for tier_id in (0, 1):
                for file_id in sorted(store.tiers[tier_id].dirty):
                    store.flush(file_id)
    finally:
        store.close()

    elapsed_ms = clock.now_ms()
    latency_array = np.asarray(latencies, dtype=np.float64)
    metrics: dict[str, float] = {
        "data_wait_ms": total_data_wait_ms,
        "data_wait_pct": 100.0 * total_data_wait_ms / elapsed_ms if elapsed_ms else 0.0,
        "throughput": total_samples / (elapsed_ms / 1000.0) if elapsed_ms else 0.0,
        "fast_hit_rate": fast_hits / reads if reads else 0.0,
        "mid_hit_rate": tier_served.get(1, 0) / reads if reads else 0.0,
        "slow_hit_rate": tier_served.get(2, 0) / reads if reads else 0.0,
        "migration_bytes": float(store.migration_bytes),
        "wasted_prefetch_bytes": float(store.wasted_prefetch_bytes),
        "latency_mean_ms": float(latency_array.mean()) if latency_array.size else 0.0,
        "latency_p95_ms": float(np.percentile(latency_array, 95)) if latency_array.size else 0.0,
        "latency_p99_ms": float(np.percentile(latency_array, 99)) if latency_array.size else 0.0,
        "elapsed_ms": elapsed_ms,
        "read_count": float(reads),
        "sample_count": float(total_samples),
        "mode": float(1.0),  # 1 == real, 0 == virtual
    }
    for name, value in metrics.items():
        logger.set_metric(name, value)
    return RealResult(metrics=metrics, logger=logger, store=store, clock=clock)


def _ranks(values: np.ndarray) -> np.ndarray:
    """Return average ranks (1-based) with ties handled."""
    order = np.argsort(values)
    ranked = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index
        while end + 1 < len(values) and values[order[end + 1]] == values[order[index]]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranked[order[position]] = average
        index = end + 1
    return ranked


def spearman_rho(a: Iterable[float], b: Iterable[float]) -> float:
    """Compute Spearman's rank correlation between two metric lists."""
    left = np.asarray(list(a), dtype=np.float64)
    right = np.asarray(list(b), dtype=np.float64)
    if left.size != right.size or left.size < 2:
        raise ValueError("Need equal-length vectors of at least two elements")
    return float(np.corrcoef(_ranks(left), _ranks(right))[0, 1])


def rank_check(
    config: dict,
    policies: Iterable[str],
    scenario: str = "A",
    seed: int = 0,
    subset_images: int | None = None,
    epochs: int | None = None,
) -> dict:
    """Compare virtual and real policy rankings on the same small workload.

    Returns a dictionary with both metric vectors, Spearman rho and a
    pairwise-order agreement fraction.
    """
    config = apply_model_selection(config)
    limit = subset_images if subset_images is not None else config.get("real_subset_images")
    chunk_limit = math.ceil(limit / chunk_size_from(config)) if limit else None
    names = list(policies)
    virtual = [
        run_simulation(
            config, name, scenario, seed, epochs=epochs, chunk_limit=chunk_limit
        ).metrics["data_wait_ms"]
        for name in names
    ]
    real = [
        run_real_training(
            config, name, scenario, seed, epochs=epochs, subset_images=limit
        ).metrics["data_wait_ms"]
        for name in names
    ]
    # Rho is undefined when either mode produces a complete tie (Scenario D).
    rho = (
        spearman_rho(virtual, real)
        if len(set(virtual)) > 1 and len(set(real)) > 1
        else None
    )
    agreements = 0
    comparisons = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            comparisons += 1
            if (virtual[i] - virtual[j]) * (real[i] - real[j]) >= 0:
                agreements += 1
    return {
        "policies": names,
        "virtual_data_wait_ms": virtual,
        "real_data_wait_ms": real,
        "spearman_rho": rho,
        "pairwise_agreement": agreements / comparisons if comparisons else 1.0,
    }


def chunk_size_from(config: dict) -> int:
    """Return the configured chunk size in samples."""
    return int(config["chunk_size_samples"])


def main() -> None:
    """Run one real-mode training run or the virtual-vs-real rank check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="lru", choices=sorted(POLICY_CLASSES))
    parser.add_argument("--scenario", default="A", choices=["A", "B", "C", "D", "E", "F"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="results/real")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rank-check", action="store_true")
    parser.add_argument("--model", choices=available_models(), default=None)
    parser.add_argument(
        "--compute-source", choices=("manual", "profile"), default=None,
        help="Use manual config values or this machine's profile-derived compute/checkpoint values.",
    )
    parser.add_argument(
        "--profile-models", action="store_true",
        help="Profile every registered CPU model and write results/models/profiles.json.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.profile_models:
        payload = profile_models()
        validate_profiles(payload)
        output = resolve_path("results/models/profiles.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"profiles: {output}")
        for name, profile in payload["profiles"].items():
            print(
                f"  {name}: parameters={profile['parameter_count']:,} "
                f"median={profile['median_step_ms']:.3f} ms "
                f"p95={profile['p95_step_ms']:.3f} ms "
                f"checkpoint={profile['checkpoint_bytes']:,} bytes "
                f"peak_ram={profile['peak_ram_bytes']:,} bytes"
            )
        return
    config = apply_model_selection(
        config, name=args.model, compute_source=args.compute_source
    )
    if args.rank_check:
        names = ["allslow", "lru", "clairvoyant", "belady"]
        report = rank_check(
            config, names, scenario=args.scenario, seed=args.seed,
            subset_images=args.subset, epochs=args.epochs,
        )
        output_base = Path(args.output_dir)
        if args.model is not None and args.model != "cnn":
            output_base = resolve_path("results/models") / args.model / "real"
        output = output_base / f"rank_{args.scenario}_seed{args.seed}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"rank check: {report}")
        print(f"report: {output}")
        return

    result = run_real_training(
        config, args.policy, args.scenario, args.seed,
        epochs=args.epochs, subset_images=args.subset,
    )
    output_base = Path(args.output_dir)
    if args.model is not None and args.model != "cnn":
        output_base = resolve_path("results/models") / args.model / "real"
    output = output_base / args.scenario
    result.logger.write_trace(output / f"{args.policy}_seed{args.seed}_trace.csv", fmt="csv")
    result.logger.write_metrics(output / f"{args.policy}_seed{args.seed}_metrics.csv")
    result.logger.write_migration_log(output / f"{args.policy}_seed{args.seed}_migrations.csv")
    print(
        f"policy={args.policy} scenario={args.scenario} seed={args.seed} "
        f"model={model_name(config)} (real)"
    )
    for name in ("data_wait_ms", "throughput", "fast_hit_rate", "migration_bytes", "elapsed_ms"):
        print(f"  {name}: {result.metrics[name]:.4f}")


if __name__ == "__main__":
    main()
