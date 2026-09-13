from pathlib import Path

RANDOM_SEED = 42
TARGET = "SeverityScore"
CASE_ID = "CaseReference"
SEVERE_CUTOFF = 4
# Selected by 3-fold out-of-fold validation using a 5:1 false-negative:false-positive cost ratio.
SEVERE_ESCALATION_THRESHOLD = 0.20

# These fields appear downstream of, or strongly entangled with, the triage decision.
# They are excluded from the deployable model to reduce target leakage risk.
LEAKAGE_RISK_COLUMNS = [
    "InvestigationRoute",
    "OmbudsmanInvestigationRequired",
    "PredictedRemedyBand",
    "ExpectedFinancialRedressGBP",
]

DATE_COLUMN = "CaseCreatedDate"
DATE_FEATURES = [
    "CaseCreatedYear",
    "CaseCreatedMonth",
    "CaseCreatedQuarter",
    "CaseCreatedDayOfWeek",
]

MODEL_PARAMS = {
    "iterations": 260,
    "depth": 6,
    "learning_rate": 0.08,
    "loss_function": "MultiClass",
    "l2_leaf_reg": 5,
    "random_seed": RANDOM_SEED,
    "verbose": False,
    "thread_count": -1,
}
