from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from ftwin_core.domain.models import LedgerEntry


REQUIRED_COLUMNS = {"date", "amount", "category", "kind"}


def load_ledger(csv_path: str | Path) -> List[LedgerEntry]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in ledger CSV: {missing}")

    df["date"] = pd.to_datetime(df["date"]).dt.date
    entries: List[LedgerEntry] = []
    for _, row in df.iterrows():
        entries.append(
            LedgerEntry(
                date=row["date"],
                amount=float(row["amount"]),
                category=str(row["category"]),
                kind=str(row["kind"]).lower(),
            )
        )
    return entries

