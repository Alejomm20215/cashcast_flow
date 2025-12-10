from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from ftwin_core.domain.models import ForecastConfig, ForecastResult, LedgerEntry, MonthlyForecast
from ftwin_core.services import forecaster, forecaster_prophet

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ModelResult:
    name: str
    series: List[float]
    mae: float


def _evaluate_mae(actual: np.ndarray, preds: np.ndarray) -> float:
    if len(actual) == 0:
        return np.inf
    return float(np.mean(np.abs(actual - preds[: len(actual)])))


def _fit_arima(series: pd.Series, periods: int) -> List[float]:
    if len(series) < 4:
        return [series.mean()] * periods
    try:
        model = ARIMA(series, order=(1, 0, 1))
        res = model.fit()
        fc = res.forecast(steps=periods)
        return fc.clip(lower=0).tolist()
    except Exception:
        return [series.mean()] * periods


def forecast_cashflow_ensemble(entries: Iterable[LedgerEntry], config: ForecastConfig) -> ForecastResult:
    """
    Ensemble forecaster: naive (mean), Prophet, ARIMA. Weighted by inverse MAE on history.
    """
    rows = [
        {"date": e.date, "amount": e.amount, "kind": e.kind, "category": e.category}
        for e in entries
    ]
    if not rows:
        raise ValueError("No ledger entries provided")

    df = pd.DataFrame(rows)
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby(["month", "kind"])["amount"].sum().unstack(fill_value=0)

    income_hist = monthly.get("income", pd.Series(dtype=float))
    expense_hist = monthly.get("expense", pd.Series(dtype=float))

    # Naive baseline
    naive_income = [income_hist.mean()] * config.months
    naive_expense = [expense_hist.mean()] * config.months

    # Prophet
    try:
        prophet_income = forecaster_prophet._fit_and_forecast_series(income_hist, config.months)  # type: ignore[attr-defined]
    except Exception:
        prophet_income = naive_income
    try:
        prophet_expense = forecaster_prophet._fit_and_forecast_series(expense_hist, config.months)  # type: ignore[attr-defined]
    except Exception:
        prophet_expense = naive_expense

    # ARIMA
    arima_income = _fit_arima(income_hist, config.months)
    arima_expense = _fit_arima(expense_hist, config.months)

    # Compute MAE on history using one-step-ahead where possible
    hist_idx = monthly.index
    hist_income = income_hist.values
    hist_expense = expense_hist.values

    naive_mae_income = _evaluate_mae(hist_income, np.array([income_hist.mean()] * len(hist_income)))
    naive_mae_expense = _evaluate_mae(hist_expense, np.array([expense_hist.mean()] * len(hist_expense)))

    prophet_mae_income = _evaluate_mae(hist_income, np.array(prophet_income[: len(hist_income)]))
    prophet_mae_expense = _evaluate_mae(hist_expense, np.array(prophet_expense[: len(hist_expense)]))

    arima_mae_income = _evaluate_mae(hist_income, np.array(arima_income[: len(hist_income)]))
    arima_mae_expense = _evaluate_mae(hist_expense, np.array(arima_expense[: len(hist_expense)]))

    def weights(maes: List[float]) -> List[float]:
        inv = np.array([1 / m if m > 0 else 1 for m in maes], dtype=float)
        if inv.sum() == 0:
            return [1 / len(maes)] * len(maes)
        return (inv / inv.sum()).tolist()

    w_income = weights([naive_mae_income, prophet_mae_income, arima_mae_income])
    w_expense = weights([naive_mae_expense, prophet_mae_expense, arima_mae_expense])

    income_fc = np.average(
        np.vstack([naive_income, prophet_income, arima_income]),
        axis=0,
        weights=w_income,
    ).tolist()
    expense_fc = np.average(
        np.vstack([naive_expense, prophet_expense, arima_expense]),
        axis=0,
        weights=w_expense,
    ).tolist()

    today = pd.Timestamp("today").normalize()
    start_month = pd.Timestamp(today.year, today.month, 1)
    months: List[MonthlyForecast] = []
    for i in range(config.months):
        m_date = (start_month.to_period("M") + i).to_timestamp().date()
        months.append(
            MonthlyForecast(
                month=m_date,
                income=float(max(income_fc[i], 0.0)),
                expense=float(max(expense_fc[i], 0.0)),
            )
        )

    return ForecastResult(config=config, monthly=months)

