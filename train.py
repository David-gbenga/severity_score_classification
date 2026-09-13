from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from phso_severity.config import RANDOM_SEED, TARGET
from phso_severity.features import categorical_columns, engineer_features
from phso_severity.metrics import classification_metrics
from phso_severity.model import build_model
from phso_severity.risk import apply_severe_escalation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/training_dataset.csv")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()

    out = Path(args.artifacts)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train)
    if TARGET not in df:
        raise ValueError(f"Training data must contain {TARGET}")

    y = df[TARGET].astype(int)
    x_raw = df.drop(columns=[TARGET])
    x = engineer_features(x_raw)
    cats = categorical_columns(x)

    # Out-of-fold probabilities give a less optimistic estimate than one train/validation split.
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_SEED)
    # CatBoost's sklearn clone integration is stable, but categorical columns must be passed by index/name.
    oof = np.zeros((len(x), 6), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(x, y), start=1):
        model = build_model(random_seed=RANDOM_SEED + fold)
        model.fit(x.iloc[tr_idx], y.iloc[tr_idx], cat_features=cats)
        oof[va_idx] = model.predict_proba(x.iloc[va_idx])

    classes = np.array([1, 2, 3, 4, 5, 6])
    base_pred = classes[np.argmax(oof, axis=1)]

    # Business-cost calibration: missing a severe case is treated as 5x more costly
    # than progressing a low-severity case. Threshold is selected only from OOF predictions.
    threshold_candidates = np.arange(0.20, 0.61, 0.025)
    threshold_results = []
    for threshold in threshold_candidates:
        candidate_pred = apply_severe_escalation(oof, classes, float(threshold))
        candidate_metrics = classification_metrics(y, candidate_pred)
        risk_cost = 5 * candidate_metrics["severe_false_negatives"] + candidate_metrics["severe_false_positives"]
        threshold_results.append((risk_cost, -candidate_metrics["macro_f1"], float(threshold), candidate_metrics))

    _, _, selected_threshold, risk_metrics = min(threshold_results, key=lambda item: (item[0], item[1]))
    risk_pred = apply_severe_escalation(oof, classes, selected_threshold)

    metrics = {
        "validation_strategy": f"{args.folds}-fold stratified out-of-fold",
        "business_cost_assumption": "severe false negative = 5x low-severity false positive",
        "severe_escalation_threshold": selected_threshold,
        "base_multiclass": classification_metrics(y, base_pred),
        "risk_adjusted": risk_metrics,
    }
    (out / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    final_model = build_model(random_seed=RANDOM_SEED)
    final_model.fit(x, y, cat_features=cats)
    final_model.save_model(out / "severity_model.cbm")

    metadata = {
        "feature_columns": list(x.columns),
        "categorical_columns": cats,
        "classes": [1, 2, 3, 4, 5, 6],
        "severe_escalation_threshold": selected_threshold,
    }
    joblib.dump(metadata, out / "model_metadata.joblib")

    importance = pd.DataFrame({
        "feature": x.columns,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(out / "feature_importance.csv", index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
