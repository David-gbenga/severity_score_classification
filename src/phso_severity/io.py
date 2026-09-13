from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
