"""Phase 6 tests: Word2Vec features, reuse prediction and embedding policy."""

from __future__ import annotations

import json

import numpy as np

from config import load_config
from embeddings import (
    REUSE_FEATURES,
    ReusePredictor,
    access_sequences,
    train_and_evaluate,
    train_word2vec,
)
from features import FEATURE_NAMES
from policies import AccessEvent, EmbeddingPrefetchPolicy, Pin, Promote
from tiers import FAST, MID, SLOW


class FakeView:
    def __init__(self) -> None:
        self.history: dict[str, object] = {}

    def occupancy(self) -> dict[int, int]:
        return {FAST: 100, MID: 100, SLOW: 0}

    def free_space(self) -> dict[int, int | None]:
        return {FAST: 900, MID: 900, SLOW: None}

    def holds(self, file_id: int) -> tuple[int, ...]:
        return (SLOW,)

    def size_of(self, file_id: int) -> int:
        return 154004


def event(file_id: int) -> AccessEvent:
    return AccessEvent(
        file_id=file_id,
        size_bytes=154004,
        op="read",
        epoch=0,
        step=1,
        phase="train",
    )


def test_access_sequences_and_word2vec(tmp_path) -> None:
    config = load_config()
    config["embeddings"] = dict(
        config["embeddings"], word2vec_epochs=2, embedding_size=8
    )
    sequences = access_sequences(config, ["A"], [0])
    assert sequences and all(isinstance(token, str) for token in sequences[0])
    model = train_word2vec(config, sequences)
    assert model.wv.vector_size == 8
    assert len(model.wv) > 0


def test_reuse_feature_order() -> None:
    assert REUSE_FEATURES[:16] == [f"emb_{index}" for index in range(16)]
    assert REUSE_FEATURES[16:] == FEATURE_NAMES


class FakePredictor:
    def __init__(self, score: float) -> None:
        self.score = score

    def features(self, event, view, extractor) -> np.ndarray:
        return np.zeros(36, dtype=np.float32)

    def predict_proba(self, features) -> float:
        assert len(features) == 36
        return self.score


def test_embedding_policy_promotes_and_pins_high_reuse() -> None:
    policy = EmbeddingPrefetchPolicy(FakePredictor(0.9))
    actions = policy.on_access(event(7), FakeView())
    assert actions[0] == Promote(file_id=7, tier=FAST)
    assert Pin(file_id=7) in actions


def test_embedding_policy_promotes_without_pin_when_reuse_low() -> None:
    policy = EmbeddingPrefetchPolicy(FakePredictor(0.1))
    actions = policy.on_access(event(7), FakeView())
    assert actions == [Promote(file_id=7, tier=FAST)]


def test_full_embedding_pipeline_on_small_split(tmp_path) -> None:
    config = load_config()
    config["epochs"] = 1
    config["embeddings"] = {
        "train_scenarios": ["A", "B"],
        "train_seeds": [0],
        "test_scenario": "C",
        "test_seeds": [1],
        "focus_seeds": [1],
        "embedding_size": 8,
        "window": 5,
        "word2vec_epochs": 2,
        "reuse_horizon": 5,
        "n_estimators": 10,
        "model_path": str(tmp_path / "model.json"),
        "report_dir": str(tmp_path / "report"),
    }
    report = train_and_evaluate(config)
    assert set(report) >= {"split", "test", "policies"}
    assert 0.0 <= report["test"]["accuracy"] <= 1.0
    assert 0.0 <= report["test"]["roc_auc"] <= 1.0
    assert {"C", "A"} <= set(report["policies"])
    assert "clairvoyant" in report["policies"]["A"]
    assert (tmp_path / "model.json").is_file()
    report_file = tmp_path / "report" / "embedding_report.json"
    assert json.loads(report_file.read_text(encoding="utf-8"))["test"] == report["test"]
