from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Dict, Any

import numpy as np
import pandas as pd

from ftwin_core.domain.models import LedgerEntry


def _memory_path() -> Path:
    return Path.home() / ".ftwin_memory.json"


def train_user_models(entries: Iterable[LedgerEntry]) -> Dict[str, Any]:
    """
    Lightweight "training" step:
    - Compute per-category monthly mean/std for income and expense.
    - Estimate savings rate from positive net months.
    - Estimate lifestyle creep: share of income growth that turned into expense growth (rough proxy).
    Stores results to ~/.ftwin_memory.json
    """
    rows = [
        {"date": e.date, "amount": e.amount, "category": e.category, "kind": e.kind}
        for e in entries
    ]
    if not rows:
        raise ValueError("No ledger entries provided")

    df = pd.DataFrame(rows)
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()

    # Monthly aggregates
    monthly = df.groupby(["month", "kind"])["amount"].sum().unstack(fill_value=0)
    monthly["net"] = monthly.get("income", 0) - monthly.get("expense", 0)

    # Savings rate on positive income months
    pos_income = monthly[monthly.get("income", 0) > 0]
    if not pos_income.empty:
        savings_rate = float((pos_income["net"] / pos_income["income"].replace(0, np.nan)).clip(lower=0, upper=1).mean())
    else:
        savings_rate = 0.0

    # Lifestyle creep rough proxy: correlation between income changes and expense changes
    income_diff = monthly.get("income", pd.Series(dtype=float)).diff().fillna(0)
    expense_diff = monthly.get("expense", pd.Series(dtype=float)).diff().fillna(0)
    if (income_diff.abs() > 0).sum() > 0:
        creep = float(np.cov(income_diff, expense_diff)[0, 1] / (income_diff.var() + 1e-8))
        creep = max(min(creep, 1.0), 0.0)
    else:
        creep = 0.0

    # Per-category stats
    cat_stats: Dict[str, Dict[str, float]] = defaultdict(dict)
    cat_df = df.copy()
    cat_df["month"] = pd.to_datetime(cat_df["date"]).dt.to_period("M")
    by_cat = cat_df.groupby(["category", "month"])["amount"].sum()
    for cat, series in by_cat.groupby(level=0):
        vals = series.values
        cat_stats[cat] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    memory = {
        "savings_rate": savings_rate,
        "lifestyle_creep": creep,
        "categories": cat_stats,
        "months_observed": len(monthly),
    }

    _memory_path().write_text(json.dumps(memory, indent=2))
    return memory

