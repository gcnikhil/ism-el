"""Phase 6: Word2Vec access embeddings and a next-reuse predictor.

The model treats opaque file ids as "words" in the observed access stream. It
learns representations from *past* sequences only; no true types, names or
future oracle order are available at inference. The reuse predictor combines
each file's embedding with the observable features from ``features.py`` to
estimate whether that file will be read again within a short window.

Scenario C is the focus: there the access order depends on training state, so
seed-based clairvoyance is not valid. Scenario A is seed-determined by design,
so learned methods are not expected to beat the NoPFS-style upper bound there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from config import load_config, resolve_path
from features import FEATURE_NAMES, FeatureExtractor
from simulator import run_simulation

def reuse_features(embedding_size: int) -> list[str]:
    """Return feature column names for a given embedding dimensionality."""
    return [f"emb_{index}" for index in range(embedding_size)] + FEATURE_NAMES


REUSE_FEATURES = reuse_features(16)


class ReusePredictor:
    """Predict short-horizon reuse probability from embedding + observables."""

    def __init__(
        self,
        model: XGBClassifier,
        word2vec: Word2Vec,
        embedding_size: int,
    ) -> None:
        """Bind the trained predictor to its embedding model."""
        self.model = model
        self.word2vec = word2vec
        self.embedding_size = embedding_size

    def embedding_for(self, file_id: int) -> np.ndarray:
        """Return the learned vector for an opaque id, or zeros if unseen."""
        token = str(file_id)
        if token in self.word2vec.wv:
            return np.asarray(self.word2vec.wv[token], dtype=np.float32)
        return np.zeros(self.embedding_size, dtype=np.float32)

    def features(self, event, view, extractor: FeatureExtractor) -> np.ndarray:
        """Build the joint embedding/observable feature vector."""
        observables = extractor.extract(event, view)
        embedding = self.embedding_for(event.file_id)
        return np.asarray(
            list(embedding) + [observables[name] for name in FEATURE_NAMES],
            dtype=np.float32,
        )

    def predict_proba(self, features: np.ndarray) -> float:
        """Return P(file will be reused within the configured window)."""
        probabilities = self.model.predict_proba(
            np.asarray([features], dtype=np.float32)
        )
        return float(probabilities[0][1])

    def save(self, model_path: Path) -> None:
        """Persist the predictor and Word2Vec model."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(model_path))
        self.word2vec.save(str(model_path.with_suffix(".word2vec")))
        Path(str(model_path) + ".meta.json").write_text(
            json.dumps({"embedding_size": self.embedding_size}), encoding="utf-8"
        )


def load_predictor(config: dict) -> ReusePredictor:
    """Load a trained reuse predictor from the configured path."""
    path = resolve_path(config["embeddings"]["model_path"])
    if not path.is_file():
        raise FileNotFoundError(
            f"Embedding model not found: {path}. Run `python embeddings.py` first."
        )
    meta = json.loads(Path(str(path) + ".meta.json").read_text(encoding="utf-8"))
    model = XGBClassifier()
    model.load_model(str(path))
    word2vec = Word2Vec.load(str(path.with_suffix(".word2vec")))
    return ReusePredictor(model, word2vec, int(meta["embedding_size"]))


def access_sequences(
    config: dict,
    scenarios: Iterable[str],
    seeds: Iterable[int],
) -> list[list[str]]:
    """Build opaque-id sequences from generated workloads (harness side only)."""
    from workload import generate_workload

    sequences: list[list[str]] = []
    for scenario in scenarios:
        for seed in seeds:
            workload = generate_workload(config, scenario, seed)
            sequences.append([str(file_id) for file_id in workload.read_stream])
    return sequences


def train_word2vec(config: dict, sequences: list[list[str]]) -> Word2Vec:
    """Train a skip-gram Word2Vec model on observed access sequences."""
    params = config["embeddings"]
    return Word2Vec(
        sentences=sequences,
        vector_size=int(params["embedding_size"]),
        window=int(params["window"]),
        min_count=1,
        sg=1,
        workers=2,
        seed=0,
        epochs=int(params["word2vec_epochs"]),
    )


def collect_reuse_dataset(
    config: dict,
    scenarios: Iterable[str],
    seeds: Iterable[int],
    word2vec: Word2Vec,
    embedding_size: int,
) -> pd.DataFrame:
    """Collect joint features and offline reuse labels from baseline runs."""
    from workload import generate_workload

    params = config["embeddings"]
    horizon = int(params["reuse_horizon"])
    rows: list[dict] = []

    for scenario in scenarios:
        for seed in seeds:
            workload = generate_workload(config, scenario, seed)
            stream = workload.read_stream
            positions: dict[int, list[int]] = {}
            for index, file_id in enumerate(stream):
                positions.setdefault(file_id, []).append(index)
            stream_cursor = 0
            extractor = FeatureExtractor()

            def on_access(event, view, result) -> None:
                nonlocal stream_cursor
                del result
                observables = extractor.extract(event, view)
                token = str(event.file_id)
                embedding = (
                    np.asarray(word2vec.wv[token], dtype=np.float32)
                    if token in word2vec.wv
                    else np.zeros(embedding_size, dtype=np.float32)
                )
                row: dict[str, float] = {
                    f"emb_{index}": float(value)
                    for index, value in enumerate(embedding)
                }
                row.update(observables)

                # Find this occurrence in the stream and label short-horizon reuse.
                occurrence = next(
                    (
                        index
                        for index in positions.get(event.file_id, [])
                        if index >= stream_cursor
                    ),
                    None,
                )
                if occurrence is not None:
                    row["reused"] = int(
                        any(
                            occurrence < other <= occurrence + horizon
                            for other in positions[event.file_id]
                        )
                    )
                    stream_cursor = occurrence + 1
                else:
                    row["reused"] = 0

                row["scenario"] = scenario
                row["seed"] = seed
                rows.append(row)

            run_simulation(config, "lru", scenario, seed, on_access=on_access)

    return pd.DataFrame(rows)


def train_reuse_model(config: dict, frame: pd.DataFrame) -> XGBClassifier:
    """Fit a binary XGBoost reuse classifier."""
    model = XGBClassifier(
        n_estimators=int(config["embeddings"]["n_estimators"]),
        max_depth=4,
        learning_rate=0.1,
        objective="binary:logistic",
        tree_method="hist",
        random_state=0,
        n_jobs=2,
    )
    columns = reuse_features(int(config["embeddings"]["embedding_size"]))
    model.fit(frame[columns].to_numpy(dtype=np.float32), frame["reused"])
    return model


def evaluate_reuse_model(
    predictor: ReusePredictor, frame: pd.DataFrame
) -> dict:
    """Return accuracy and ROC-AUC for short-horizon reuse prediction."""
    columns = reuse_features(predictor.embedding_size)
    features = frame[columns].to_numpy(dtype=np.float32)
    probabilities = predictor.model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(frame["reused"], predictions)),
        "roc_auc": float(roc_auc_score(frame["reused"], probabilities)),
        "row_count": int(len(frame)),
        "positive_rate": float(frame["reused"].mean()),
    }


def evaluate_policy(config: dict, predictor: ReusePredictor) -> dict:
    """Compare the embedding policy with baselines on scenarios C and A."""
    seeds = [int(seed) for seed in config["embeddings"]["focus_seeds"]]
    keys = ("data_wait_ms", "throughput", "fast_hit_rate", "migration_bytes")
    report: dict[str, dict] = {}
    for scenario in ("C", "A"):
        policies = ["embedding", "lru", "context_rules"]
        if scenario == "A":
            # Seed-determined order makes clairvoyance valid only in A.
            policies.append("clairvoyant")
        scenario_report: dict[str, dict] = {}
        for name in policies:
            totals = {key: 0.0 for key in keys}
            for seed in seeds:
                result = run_simulation(
                    config,
                    name,
                    scenario,
                    seed,
                    classifier_model=predictor,
                )
                for key in keys:
                    totals[key] += float(result.metrics[key])
            scenario_report[name] = {
                key: value / len(seeds) for key, value in totals.items()
            }
        report[scenario] = scenario_report
    return report


def train_and_evaluate(config: dict) -> dict:
    """Run the full Word2Vec + reuse pipeline and write its report."""
    params = config["embeddings"]
    train_scenarios = [str(item) for item in params["train_scenarios"]]
    train_seeds = [int(seed) for seed in params["train_seeds"]]
    test_scenario = str(params["test_scenario"])
    test_seeds = [int(seed) for seed in params["test_seeds"]]
    embedding_size = int(params["embedding_size"])

    sequences = access_sequences(config, train_scenarios, train_seeds)
    word2vec = train_word2vec(config, sequences)
    train_frame = collect_reuse_dataset(
        config, train_scenarios, train_seeds, word2vec, embedding_size
    )
    model = train_reuse_model(config, train_frame)
    predictor = ReusePredictor(model, word2vec, embedding_size)

    test_frame = collect_reuse_dataset(
        config, [test_scenario], test_seeds, word2vec, embedding_size
    )
    report = {
        "split": {
            "train_scenarios": train_scenarios,
            "train_seeds": train_seeds,
            "test_scenario": test_scenario,
            "test_seeds": test_seeds,
            "reuse_horizon": int(params["reuse_horizon"]),
        },
        "train": {"row_count": int(len(train_frame))},
        "test": evaluate_reuse_model(predictor, test_frame),
        "policies": evaluate_policy(config, predictor),
    }

    report_dir = resolve_path(params["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    predictor.save(resolve_path(params["model_path"]))
    (report_dir / "embedding_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {"reuse_horizon": [params["reuse_horizon"]], "accuracy": [report["test"]["accuracy"]]}
    ).to_csv(report_dir / "reuse_metrics.csv", index=False)
    return report


def main() -> None:
    """Train, evaluate and print the Phase 6 embedding report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    report = train_and_evaluate(config)
    print("split:", report["split"])
    print(f"test accuracy: {report['test']['accuracy']:.4f}")
    print(f"test ROC-AUC: {report['test']['roc_auc']:.4f}")
    print("policies:", report["policies"])
    print("Note (Scenario A): seed-determined, so NoPFS-style clairvoyance is the")
    print("  valid upper bound there; learned methods are not expected to win.")
    print(f"report: {resolve_path(config['embeddings']['report_dir'])}")


if __name__ == "__main__":
    main()
