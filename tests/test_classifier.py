"""Phase 5 tests: observable features, XGBoost pipeline and ClassifierPolicy."""

from __future__ import annotations

import json

from config import load_config
from features import FEATURE_NAMES, FeatureExtractor
from policies import AccessEvent, ClassifierPolicy, Pin, Promote
from tiers import FAST, MID, SLOW


class FakeView:
    def __init__(self) -> None:
        self.history: dict[str, object] = {}

    def occupancy(self) -> dict[int, int]:
        return {FAST: 100, MID: 200, SLOW: 0}

    def free_space(self) -> dict[int, int | None]:
        return {FAST: 900, MID: 800, SLOW: None}

    def holds(self, file_id: int) -> tuple[int, ...]:
        return (FAST,) if file_id == 1 else (SLOW,)

    def size_of(self, file_id: int) -> int:
        return 154004


def event(file_id: int, **kwargs) -> AccessEvent:
    defaults = dict(
        file_id=file_id,
        size_bytes=154004,
        op="read",
        epoch=0,
        step=5,
        phase="train",
    )
    defaults.update(kwargs)
    return AccessEvent(**defaults)


def test_feature_extractor_uses_only_observable_fields() -> None:
    extractor = FeatureExtractor()
    features = extractor.extract(event(1), FakeView())
    assert list(features) == FEATURE_NAMES
    assert features["in_fast"] == 1.0
    assert features["phase_train"] == 1.0
    assert features["read_count_log"] > 0.0
    assert not any("true" in name or "type" in name for name in features)


def test_feature_extractor_tracks_repeated_access() -> None:
    extractor = FeatureExtractor()
    view = FakeView()
    extractor.extract(event(1), view)
    second = extractor.extract(event(1, step=10), view)
    assert second["read_count_log"] > 0.0
    assert second["step_gap_log"] > 0.0


class FakeModel:
    """Return a fixed predicted type for ClassifierPolicy tests."""

    def __init__(self, prediction: str) -> None:
        self.prediction = prediction

    def predict(self, features) -> list[str]:
        assert len(features[0]) == len(FEATURE_NAMES)
        return [self.prediction]


def test_classifier_policy_uses_predicted_type() -> None:
    policy = ClassifierPolicy(FakeModel("validation"))
    view = FakeView()
    assert policy.on_access(event(2), view) == [
        Promote(file_id=2, tier=MID),
        Pin(file_id=2),
    ]

    policy = ClassifierPolicy(FakeModel("metadata"))
    assert policy.on_access(event(2), FakeView())[:1] == [
        Promote(file_id=2, tier=FAST)
    ]


def test_classifier_policy_without_model_falls_back() -> None:
    policy = ClassifierPolicy(None)
    assert policy.on_access(event(2), FakeView()) == [
        Promote(file_id=2, tier=FAST)
    ]


def test_full_classifier_pipeline_on_small_split(tmp_path) -> None:
    from classifier import train_and_evaluate

    config = load_config()
    config["epochs"] = 1
    config["classifier"] = {
        "train_seeds": [0],
        "test_seeds": [1],
        "train_scenarios": ["A", "B"],
        "test_scenario": "C",
        "model_path": str(tmp_path / "model.json"),
        "report_dir": str(tmp_path / "report"),
        "n_estimators": 10,
        "max_depth": 2,
    }
    report = train_and_evaluate(config)

    assert set(report) >= {"split", "test", "classifier_policy", "test_no_phase_op"}
    assert 0.0 <= report["test"]["accuracy"] <= 1.0
    assert 0.0 <= report["test"]["macro_f1"] <= 1.0
    assert (tmp_path / "model.json").is_file()
    report_path = tmp_path / "report" / "classification_report.json"
    assert json.loads(report_path.read_text(encoding="utf-8"))["test"]["accuracy"] == (
        report["test"]["accuracy"]
    )
