from __future__ import annotations

from catboost import CatBoostClassifier

from .config import MODEL_PARAMS


def build_model(**overrides) -> CatBoostClassifier:
    params = dict(MODEL_PARAMS)
    params.update(overrides)
    return CatBoostClassifier(**params)
