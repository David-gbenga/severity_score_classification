from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from phso_severity.config import CASE_ID
from phso_severity.features import engineer_features
from phso_severity.risk import apply_severe_escalation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/test_dataset.csv")
    parser.add_argument("--model", default="artifacts/severity_model.cbm")
    parser.add_argument("--metadata", default="artifacts/model_metadata.joblib")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    if CASE_ID not in raw:
        raise ValueError(f"Input data must contain {CASE_ID}")
    case_reference = raw[CASE_ID].copy()

    x = engineer_features(raw)
    metadata = joblib.load(args.metadata)
    expected = metadata["feature_columns"]
    missing = sorted(set(expected) - set(x.columns))
    extra = sorted(set(x.columns) - set(expected))
    if missing or extra:
        raise ValueError(f"Feature schema mismatch. Missing={missing}; extra={extra}")
    x = x[expected]

    model = CatBoostClassifier()
    model.load_model(args.model)
    probabilities = model.predict_proba(x)
    classes = np.asarray(metadata["classes"], dtype=int)
    predictions = apply_severe_escalation(
        probabilities,
        classes,
        float(metadata["severe_escalation_threshold"]),
    )

    output = pd.DataFrame({CASE_ID: case_reference, "PredictedSeverityScore": predictions})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} predictions to {args.output}")


if __name__ == "__main__":
    main()
