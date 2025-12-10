from __future__ import annotations

import json
from pathlib import Path

from celery import shared_task
from django.conf import settings
from django.db import transaction

from ftwin_core.domain.models import ForecastConfig, SimulationConfig
from ftwin_core.io import config as config_io
from ftwin_core.io import ledger as ledger_io
from ftwin_core.services import forecaster, forecaster_prophet, scenario as scenario_service
from ftwin_core.services import simulator

from .models import SimulationRun


def _write_result(run_id: int, payload: dict) -> Path:
    path = Path(settings.MEDIA_ROOT) / "results" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


@shared_task
def run_simulation(run_id: int):
    try:
        run = SimulationRun.objects.get(id=run_id)
    except SimulationRun.DoesNotExist:
        return

    with transaction.atomic():
        run.status = "running"
        run.error = ""
        run.save(update_fields=["status", "error"])

    try:
        cfg = run.config or {}
        entries = ledger_io.load_ledger(run.ledger_path)

        forecast_conf = ForecastConfig(
            months=int(cfg.get("months", 12)),
            annual_inflation=float(cfg.get("annual_inflation", 0.02)),
            seed=cfg.get("seed"),
        )

        engine = cfg.get("engine", "naive")
        if engine == "prophet":
            forecast = forecaster_prophet.forecast_cashflow_prophet(entries, forecast_conf)
        else:
            forecast = forecaster.forecast_cashflow(entries, forecast_conf)

        if run.kind == "scenario" and run.delta_path:
            delta = config_io.load_scenario_delta(run.delta_path)
            forecast = scenario_service.apply_scenario(forecast, delta)

        sim_conf = SimulationConfig(
            horizon_months=int(cfg.get("months", 12)),
            paths=int(cfg.get("paths", 5000)),
            return_mean=float(cfg.get("return_mean", 0.07)),
            return_vol=float(cfg.get("return_vol", 0.15)),
            inflation_mean=float(cfg.get("inflation_mean", 0.03)),
            inflation_std=float(cfg.get("inflation_std", 0.01)),
            shock_lambda=float(cfg.get("shock_lambda", 0.1)),
            shock_mean=float(cfg.get("shock_mean", 200.0)),
            start_wealth=float(cfg.get("start_wealth", 0.0)),
            goal_target=cfg.get("goal_target"),
            liquidity_floor=cfg.get("liquidity_floor"),
            seed=cfg.get("seed"),
        )

        result = simulator.run_monte_carlo(forecast, sim_conf)
        payload = {
            "percentiles": result.percentiles,
            "config": vars(result.config),
            "goal_success_prob": result.goal_success_prob,
            "liquidity_breach_prob": result.liquidity_breach_prob,
        }
        result_path = _write_result(run.id, payload)

        with transaction.atomic():
            run.result_path = str(result_path)
            run.status = "completed"
            run.save(update_fields=["result_path", "status"])
    except Exception as exc:  # noqa: BLE001
        with transaction.atomic():
            run.status = "failed"
            run.error = str(exc)
            run.save(update_fields=["status", "error"])

