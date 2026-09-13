import pandas as pd

from phso_severity.features import engineer_features


def test_feature_engineering_removes_identifier_and_leakage_fields():
    df = pd.DataFrame({
        "CaseReference": ["X-1"],
        "CaseCreatedDate": ["01/09/2026"],
        "InvestigationRoute": ["Escalated"],
        "OmbudsmanInvestigationRequired": ["Yes"],
        "PredictedRemedyBand": ["F"],
        "ExpectedFinancialRedressGBP": [1000.0],
        "AgeBand": [None],
    })
    x = engineer_features(df)
    assert "CaseReference" not in x
    assert "InvestigationRoute" not in x
    assert "CaseCreatedMonth" in x
    assert x.loc[0, "AgeBand"] == "__MISSING__"
