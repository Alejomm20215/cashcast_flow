from __future__ import annotations

import copy
from typing import List

from ftwin_core.domain.models import ForecastResult, MonthlyForecast, ScenarioDelta


def apply_scenario(forecast: ForecastResult, delta: ScenarioDelta) -> ForecastResult:
    """
    Applies income/expense deltas and optional savings rate override.
    """
    adjusted: List[MonthlyForecast] = []
    income_shift = sum(delta.income_deltas.values())
    expense_shift = sum(delta.expense_deltas.values())

    for month in forecast.monthly:
        new_income = month.income + income_shift
        new_expense = month.expense + expense_shift

        if delta.savings_rate_override is not None:
            net = new_income - new_expense
            if net > 0:
                keep = net * delta.savings_rate_override
                creep = net - keep
                new_expense += creep  # lifestyle creep consumes non-saved portion

        adjusted.append(
            MonthlyForecast(
                month=month.month,
                income=new_income,
                expense=new_expense,
            )
        )

    cloned = copy.deepcopy(forecast.config)
    return ForecastResult(config=cloned, monthly=adjusted)

