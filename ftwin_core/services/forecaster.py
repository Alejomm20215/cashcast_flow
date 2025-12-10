from __future__ import annotations

import datetime as dt
import math
from typing import Iterable, List

import numpy as np
import pandas as pd

from ftwin_core.domain.models import ForecastConfig, ForecastResult, LedgerEntry, MonthlyForecast


def _month_start(date: dt.date) -> dt.date:
    return dt.date(date.year, date.month, 1)


def forecast_cashflow(entries: Iterable[LedgerEntry], config: ForecastConfig) -> ForecastResult:
    """
    Naive monthly forecaster:
    - Aggregates historical income/expense by month.
    - Uses mean monthly income/expense.
    - Applies inflation to expenses only.
    """
    rows = [
        {"date": e.date, "amount": e.amount, "kind": e.kind, "category": e.category}
        for e in entries
    ]
    if not rows:
        raise ValueError("No ledger entries provided")

    df = pd.DataFrame(rows)
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    monthly = df.groupby(["month", "kind"])["amount"].sum().unstack(fill_value=0)

    mean_income = monthly.get("income", pd.Series(dtype=float)).mean() if not monthly.empty else 0.0
    mean_expense = monthly.get("expense", pd.Series(dtype=float)).mean() if not monthly.empty else 0.0

    inflation_monthly = math.pow(1 + config.annual_inflation, 1 / 12) - 1

    today = dt.date.today()
    start_month = _month_start(today.replace(day=1))
    forecasts: List[MonthlyForecast] = []

    for i in range(config.months):
        month_date = (pd.Period(start_month, freq="M") + i).to_timestamp().date()
        expense_adj = mean_expense * math.pow(1 + inflation_monthly, i)
        forecasts.append(
            MonthlyForecast(
                month=month_date,
                income=float(mean_income),
                expense=float(expense_adj),
            )
        )

    return ForecastResult(config=config, monthly=forecasts)

