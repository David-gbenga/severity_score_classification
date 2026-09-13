from __future__ import annotations

import numpy as np


def apply_severe_escalation(
    probabilities: np.ndarray,
    classes: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Risk overlay: escalate low argmax predictions when total P(severity >= 4) is high enough."""
    probabilities = np.asarray(probabilities)
    classes = np.asarray(classes, dtype=int)
    prediction = classes[np.argmax(probabilities, axis=1)].copy()

    severe_mask = classes >= 4
    severe_probability = probabilities[:, severe_mask].sum(axis=1)
    escalate = (prediction < 4) & (severe_probability >= threshold)

    if np.any(escalate):
        severe_classes = classes[severe_mask]
        severe_probabilities = probabilities[escalate][:, severe_mask]
        prediction[escalate] = severe_classes[np.argmax(severe_probabilities, axis=1)]

    return prediction.astype(int)
