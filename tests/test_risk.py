import numpy as np

from phso_severity.risk import apply_severe_escalation


def test_severe_escalation_moves_low_prediction_across_triage_boundary():
    classes = np.array([1, 2, 3, 4, 5, 6])
    probabilities = np.array([[0.05, 0.15, 0.55, 0.15, 0.07, 0.03]])
    pred = apply_severe_escalation(probabilities, classes, threshold=0.20)
    assert pred[0] == 4


def test_no_escalation_below_threshold():
    classes = np.array([1, 2, 3, 4, 5, 6])
    probabilities = np.array([[0.05, 0.15, 0.65, 0.08, 0.05, 0.02]])
    pred = apply_severe_escalation(probabilities, classes, threshold=0.20)
    assert pred[0] == 3
