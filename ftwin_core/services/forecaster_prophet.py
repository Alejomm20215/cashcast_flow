from __future__ import annotations

import datetime as dt
import math
from typing import Iterable, List

import pandas as pd

from ftwin_core.domain.models import ForecastConfig, ForecastResult, LedgerEntry, MonthlyForecast


def forecast_cashflow_prophet(entries: Iterable[LedgerEntry], config: ForecastConfig) -> ForecastResult:
    """
    Prophet-based monthly forecaster:
    - Fits separate Prophet models for income and expense monthly totals.
    - Applies inflation to expenses over horizon.
    """
    try:
        from prophet import Prophet  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError("prophet is not installed; install prophet>=1.1") from exc

    rows = [
        {"date": e.date, "amount": e.amount, "kind": e.kind, "category": e.category}
        for e in entries
    ]
    if not rows:
        raise ValueError("No ledger entries provided")

    df = pd.DataFrame(rows)
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()

    def _fit_and_forecast(kind: str) -> List[float]:
        subset = df[df["kind"] == kind]
        if subset.empty:
            return [0.0] * config.months
        monthly = subset.groupby("month")["amount"].sum().reset_index()
        monthly = monthly.rename(columns={"month": "ds", "amount": "y"})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(monthly)
        future = model.make_future_dataframe(periods=config.months, freq="MS", include_history=False)
        forecast = model.predict(future)
        return forecast["yhat"].clip(lower=0).tolist()

    income_pred = _fit_and_forecast("income")
    expense_pred = _fit_and_forecast("expense")

    inflation_monthly = math.pow(1 + config.annual_inflation, 1 / 12) - 1
    expense_pred_adj = [val * math.pow(1 + inflation_monthly, idx) for idx, val in enumerate(expense_pred)]

    today = dt.date.today()
    start_month = dt.date(today.year, today.month, 1)
    months = []
    for i in range(config.months):
        m_date = (pd.Period(start_month, freq="M") + i).to_timestamp().date()
        months.append(
            MonthlyForecast(
                month=m_date,
                income=float(income_pred[i]) if i < len(income_pred) else 0.0,
                expense=float(expense_pred_adj[i]) if i < len(expense_pred_adj) else 0.0,
            )
        )

    return ForecastResult(config=config, monthly=months)

