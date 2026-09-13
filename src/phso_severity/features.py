from __future__ import annotations

import pandas as pd

from .config import CASE_ID, DATE_COLUMN, LEAKAGE_RISK_COLUMNS


def engineer_features(df: pd.DataFrame, *, drop_case_id: bool = True) -> pd.DataFrame:
    """Create reproducible, pre-triage features and remove leakage-risk columns."""
    x = df.copy()

    if DATE_COLUMN in x.columns:
        # Source dates are supplied as DD/MM/YYYY.
        dt = pd.to_datetime(x[DATE_COLUMN], format="%d/%m/%Y", errors="coerce")
        x["CaseCreatedYear"] = dt.dt.year
        x["CaseCreatedMonth"] = dt.dt.month
        x["CaseCreatedQuarter"] = dt.dt.quarter
        x["CaseCreatedDayOfWeek"] = dt.dt.dayofweek
        x = x.drop(columns=[DATE_COLUMN])

    x = x.drop(columns=[c for c in LEAKAGE_RISK_COLUMNS if c in x.columns], errors="ignore")
    if drop_case_id:
        x = x.drop(columns=[CASE_ID], errors="ignore")

    # CatBoost requires categorical missing values to be represented explicitly.
    for col in x.select_dtypes(include=["object", "string", "category"]).columns:
        x[col] = x[col].fillna("__MISSING__").astype(str)

    return x


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=["object", "string", "category"]).columns)
