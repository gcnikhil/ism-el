"""Phase 5: feature-based XGBoost file-type classifier.

Training uses features extracted by the harness from baseline runs on
``train_scenarios`` with ``train_seeds``. Evaluation holds out an entire
scenario (default C) *and* separate seeds. The evaluation-only true types are
used solely as labels here; ``ClassifierPolicy`` only sees the trained model's
predictions at runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from config import load_config, resolve_path
from features import FEATURE_NAMES, FeatureExtractor
from simulator import run_simulation


class LabelClassifier:
    """Wrap a numeric-label XGBoost model so policies receive type strings."""

    def __init__(self, model: XGBClassifier, classes: Iterable[str]) -> None:
        """Bind the booster to its string-label mapping."""
        self.model = model
        self.classes = np.asarray(list(classes))

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return predicted file-type strings for a feature matrix."""
        indices = np.asarray(self.model.predict(features), dtype=int)
        return self.classes[indices]

    def save(self, path: Path) -> None:
        """Persist the booster and its label mapping."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        Path(str(path) + ".labels.json").write_text(
            json.dumps(self.classes.tolist()), encoding="utf-8"
        )


def load_model(config: dict) -> LabelClassifier:
    """Load a trained classifier from the configured model path."""
    path = resolve_path(config["classifier"]["model_path"])
    if not path.is_file():
        raise FileNotFoundError(
            f"Classifier model not found: {path}. Run `python classifier.py --train` first."
        )
    model = XGBClassifier()
    model.load_model(str(path))
    classes = json.loads(Path(str(path) + ".labels.json").read_text(encoding="utf-8"))
    return LabelClassifier(model, classes)


def collect_dataset(
    config: dict,
    scenarios: Iterable[str],
    seeds: Iterable[int],
    *,
    baseline: str = "lru",
) -> pd.DataFrame:
    """Extract observable features from baseline runs on the given scenarios/seeds."""
    manifest_path = resolve_path(config["data"]["manifest"])
    truth = {
        int(entry["id"]): str(entry["true_type"])
        for entry in json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    }
    rows: list[dict] = []

    for scenario in scenarios:
        for seed in seeds:
            extractor = FeatureExtractor()

            def on_access(event, view, result) -> None:
                del result
                row = extractor.extract(event, view)
                row["label"] = truth[event.file_id]
                row["scenario"] = scenario
                row["seed"] = seed
                rows.append(row)

            run_simulation(config, baseline, scenario, seed, on_access=on_access)

    return pd.DataFrame(rows)


def train_classifier(
    config: dict,
    frame: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> LabelClassifier:
    """Fit an XGBoost multi-class classifier on the observable feature table."""
    columns = feature_names or FEATURE_NAMES
    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame["label"].astype(str))
    params = config["classifier"]
    model = XGBClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        tree_method="hist",
        random_state=0,
        n_jobs=2,
    )
    model.fit(frame[columns].to_numpy(dtype=np.float32), labels)
    return LabelClassifier(model, encoder.classes_)


def evaluate_classifier(
    classifier: LabelClassifier,
    frame: pd.DataFrame,
    labels: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict:
    """Return accuracy, macro-F1, confusion matrix and gain-based importance."""
    columns = feature_names or FEATURE_NAMES
    predictions = classifier.predict(frame[columns].to_numpy(dtype=np.float32))
    truth = frame["label"].astype(str).to_numpy()
    matrix = confusion_matrix(truth, predictions, labels=labels)
    gain = classifier.model.get_booster().get_score(importance_type="gain")
    importance = {
        FEATURE_NAMES[int(name[1:])]: float(value)
        for name, value in gain.items()
        if name.startswith("f") and int(name[1:]) < len(FEATURE_NAMES)
    }
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "labels": list(labels),
        "confusion_matrix": matrix.tolist(),
        "feature_importance_gain": importance,
        "row_count": int(len(frame)),
        "label_counts": {
            str(label): int((truth == label).sum()) for label in labels
        },
    }


def evaluate_policy(config: dict, classifier: LabelClassifier) -> dict:
    """Run ClassifierPolicy on the held-out scenario and average the seeds."""
    params = config["classifier"]
    scenario = str(params["test_scenario"])
    seeds = [int(seed) for seed in params["test_seeds"]]
    keys = (
        "data_wait_ms",
        "throughput",
        "fast_hit_rate",
        "migration_bytes",
    )
    totals = {key: 0.0 for key in keys}
    for seed in seeds:
        result = run_simulation(
            config,
            "classifier",
            scenario,
            seed,
            classifier_model=classifier,
        )
        for key in keys:
            totals[key] += float(result.metrics[key])
    return {
        "scenario": scenario,
        "seeds": seeds,
        **{key: value / len(seeds) for key, value in totals.items()},
    }


def train_and_evaluate(config: dict) -> dict:
    """Run the full train/hold-out pipeline and return the report dictionary."""
    params = config["classifier"]
    train_scenarios = [str(item) for item in params["train_scenarios"]]
    train_seeds = [int(seed) for seed in params["train_seeds"]]
    test_scenario = str(params["test_scenario"])
    test_seeds = [int(seed) for seed in params["test_seeds"]]

    train_frame = collect_dataset(config, train_scenarios, train_seeds)
    classifier = train_classifier(config, train_frame)
    test_frame = collect_dataset(config, [test_scenario], test_seeds)
    labels = np.asarray(sorted(train_frame["label"].astype(str).unique()))

    # Transparency ablation: drop phase/op features, which are application
    # context rather than inferred behaviour, and re-evaluate on the same split.
    no_phase_columns = [
        name
        for name in FEATURE_NAMES
        if not name.startswith("phase_") and name != "is_write"
    ]
    no_phase_classifier = train_classifier(config, train_frame, no_phase_columns)
    no_phase_eval = evaluate_classifier(
        no_phase_classifier, test_frame, labels, no_phase_columns
    )

    report = {
        "split": {
            "train_scenarios": train_scenarios,
            "train_seeds": train_seeds,
            "test_scenario": test_scenario,
            "test_seeds": test_seeds,
        },
        "train": {
            "row_count": int(len(train_frame)),
            "label_counts": {
                str(label): int(count)
                for label, count in train_frame["label"].value_counts().items()
            },
        },
        "test": evaluate_classifier(classifier, test_frame, labels),
        "test_no_phase_op": {
            "features": no_phase_columns,
            "accuracy": no_phase_eval["accuracy"],
            "macro_f1": no_phase_eval["macro_f1"],
        },
        "classifier_policy": evaluate_policy(config, classifier),
    }

    report_dir = resolve_path(config["classifier"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    classifier.save(resolve_path(config["classifier"]["model_path"]))
    (report_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        report["test"]["confusion_matrix"],
        index=labels,
        columns=labels,
    ).to_csv(report_dir / "confusion_matrix.csv")
    pd.DataFrame(
        sorted(
            report["test"]["feature_importance_gain"].items(),
            key=lambda item: item[1],
            reverse=True,
        ),
        columns=["feature", "gain"],
    ).to_csv(report_dir / "feature_importance.csv", index=False)
    return report


def main() -> None:
    """Train, evaluate and print the Phase 5 classifier report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    report = train_and_evaluate(load_config(args.config))
    test = report["test"]
    print("split:", report["split"])
    print(f"accuracy: {test['accuracy']:.4f}")
    print(f"macro_f1: {test['macro_f1']:.4f}")
    print("confusion labels:", test["labels"])
    for row in test["confusion_matrix"]:
        print("  ", row)
    print("no-phase/op ablation:", report["test_no_phase_op"])
    print("top features:", list(report["test"]["feature_importance_gain"])[:5])
    print("classifier policy:", report["classifier_policy"])
    print(f"report: {resolve_path(load_config(args.config)['classifier']['report_dir'])}")


if __name__ == "__main__":
    main()
