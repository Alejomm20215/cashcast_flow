from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ftwin_core.domain.models import ForecastConfig, ScenarioDelta, SimulationConfig


def load_forecast_config(path: str | Path) -> ForecastConfig:
    data = _read_json(path)
    return ForecastConfig(
        months=int(data.get("months", 12)),
        annual_inflation=float(data.get("annual_inflation", 0.02)),
        seed=data.get("seed"),
    )


def load_simulation_config(path: str | Path) -> SimulationConfig:
    data = _read_json(path)
    return SimulationConfig(
        horizon_months=int(data.get("horizon_months", 12)),
        paths=int(data.get("paths", 5000)),
        return_mean=float(data.get("return_mean", 0.07)),
        return_vol=float(data.get("return_vol", 0.15)),
        inflation_mean=float(data.get("inflation_mean", 0.03)),
        inflation_std=float(data.get("inflation_std", 0.01)),
        shock_lambda=float(data.get("shock_lambda", 0.1)),
        shock_mean=float(data.get("shock_mean", 200.0)),
        seed=data.get("seed"),
    )


def load_scenario_delta(path: str | Path) -> ScenarioDelta:
    data = _read_json(path)
    return ScenarioDelta(
        income_deltas=data.get("income_deltas", {}) or {},
        expense_deltas=data.get("expense_deltas", {}) or {},
        savings_rate_override=data.get("savings_rate_override"),
    )


def _read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

