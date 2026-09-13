from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(y_true, y_pred) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    severe_true = y_true >= 4
    severe_pred = y_pred >= 4
    tn, fp, fn, tp = confusion_matrix(severe_true, severe_pred, labels=[False, True]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "mean_absolute_class_error": float(np.mean(np.abs(y_true - y_pred))),
        "severe_recall": float(recall_score(severe_true, severe_pred, zero_division=0)),
        "severe_precision": float(precision_score(severe_true, severe_pred, zero_division=0)),
        "severe_false_negatives": int(fn),
        "severe_false_positives": int(fp),
        "severe_true_positives": int(tp),
        "severe_true_negatives": int(tn),
    }
