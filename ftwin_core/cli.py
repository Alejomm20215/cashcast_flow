from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

import typer

from ftwin_core.domain.models import ForecastConfig, ForecastResult, SimulationConfig
from ftwin_core.domain.models import MonthlyForecast  # noqa: E402
from ftwin_core.io import config as config_io
from ftwin_core.io import ledger as ledger_io
from ftwin_core.services import forecaster, scenario as scenario_service
from ftwin_core.services import forecaster_prophet
from ftwin_core.services import simulator

app = typer.Typer(help="Financial Twin CLI for forecasting and simulation.")


def _save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _forecast_to_json(result: ForecastResult) -> dict:
    return {
        "config": {
            "months": result.config.months,
            "annual_inflation": result.config.annual_inflation,
            "seed": result.config.seed,
        },
        "monthly": [
            {"month": m.month.isoformat(), "income": m.income, "expense": m.expense}
            for m in result.monthly
        ],
    }


def _forecast_from_json(data: dict) -> ForecastResult:
    cfg = data["config"]
    config = ForecastConfig(
        months=cfg.get("months", 12),
        annual_inflation=cfg.get("annual_inflation", 0.02),
        seed=cfg.get("seed"),
    )
    monthly = []
    for item in data["monthly"]:
        iso = item["month"]
        parts = [int(x) for x in iso.split("-")]
        monthly.append(
            MonthlyForecast(
                month=date(parts[0], parts[1], 1),
                income=item["income"],
                expense=item["expense"],
            )
        )

    return ForecastResult(config=config, monthly=monthly)


def _simulation_to_json(result):
    return {
        "config": vars(result.config),
        "percentiles": result.percentiles,
        "goal_success_prob": result.goal_success_prob,
    }


@app.command()
def forecast(
    ledger: Path = typer.Option(..., help="CSV ledger with date,amount,category,kind"),
    months: int = typer.Option(12, help="Months to forecast"),
    annual_inflation: float = typer.Option(0.02, help="Annual inflation rate for expenses"),
    engine: str = typer.Option("naive", help="Forecaster engine: naive|prophet"),
    out: Optional[Path] = typer.Option(None, help="Output path for baseline forecast JSON"),
):
    """Generate baseline cashflow forecast."""
    entries = ledger_io.load_ledger(ledger)
    config = ForecastConfig(months=months, annual_inflation=annual_inflation)
    if engine == "prophet":
        result = forecaster_prophet.forecast_cashflow_prophet(entries, config)
    else:
        result = forecaster.forecast_cashflow(entries, config)
    payload = _forecast_to_json(result)
    if out:
        _save_json(out, payload)
        typer.echo(f"Baseline forecast written to {out}")
    else:
        typer.echo(json.dumps(payload, indent=2))


def _load_forecast(
    baseline: Optional[Path],
    ledger: Optional[Path],
    months: int,
    annual_inflation: float,
    engine: str = "naive",
):
    if baseline:
        with baseline.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return _forecast_from_json(data)
    if ledger:
        entries = ledger_io.load_ledger(ledger)
        config = ForecastConfig(months=months, annual_inflation=annual_inflation)
        if engine == "prophet":
            return forecaster_prophet.forecast_cashflow_prophet(entries, config)
        return forecaster.forecast_cashflow(entries, config)
    raise typer.BadParameter("Provide either --baseline or --ledger")


@app.command()
def simulate(
    baseline: Optional[Path] = typer.Option(None, help="Baseline forecast JSON"),
    ledger: Optional[Path] = typer.Option(None, help="Ledger CSV (used if baseline not supplied)"),
    months: int = typer.Option(12, help="Months to forecast if ledger is used"),
    annual_inflation: float = typer.Option(0.02, help="Annual inflation rate if ledger is used"),
    engine: str = typer.Option("naive", help="Forecaster engine when using ledger: naive|prophet"),
    paths: int = typer.Option(5000, help="Monte Carlo paths"),
    return_mean: float = typer.Option(0.07, help="Annual return mean"),
    return_vol: float = typer.Option(0.15, help="Annual return volatility"),
    inflation_mean: float = typer.Option(0.03, help="Annual inflation mean for returns"),
    inflation_std: float = typer.Option(0.01, help="Annual inflation std for returns"),
    shock_lambda: float = typer.Option(0.1, help="Poisson lambda for shocks"),
    shock_mean: float = typer.Option(200.0, help="Mean shock cost"),
    seed: Optional[int] = typer.Option(None, help="Random seed"),
    out: Optional[Path] = typer.Option(None, help="Output path for simulation JSON"),
):
    """Run Monte Carlo simulation on baseline."""
    forecast_result = _load_forecast(baseline, ledger, months, annual_inflation, engine)
    sim_config = SimulationConfig(
        horizon_months=months,
        paths=paths,
        return_mean=return_mean,
        return_vol=return_vol,
        inflation_mean=inflation_mean,
        inflation_std=inflation_std,
        shock_lambda=shock_lambda,
        shock_mean=shock_mean,
        seed=seed,
    )
    sim_result = simulator.run_monte_carlo(forecast_result, sim_config)
    payload = _simulation_to_json(sim_result)
    if out:
        _save_json(out, payload)
        typer.echo(f"Simulation written to {out}")
    else:
        typer.echo(json.dumps(payload, indent=2))


@app.command()
def scenario(
    delta: Path = typer.Option(..., help="Scenario delta JSON"),
    baseline: Optional[Path] = typer.Option(None, help="Baseline forecast JSON"),
    ledger: Optional[Path] = typer.Option(None, help="Ledger CSV if baseline not provided"),
    months: int = typer.Option(12, help="Months to forecast if ledger is used"),
    annual_inflation: float = typer.Option(0.02, help="Annual inflation rate if ledger is used"),
    engine: str = typer.Option("naive", help="Forecaster engine when using ledger: naive|prophet"),
    paths: int = typer.Option(5000, help="Monte Carlo paths"),
    return_mean: float = typer.Option(0.07, help="Annual return mean"),
    return_vol: float = typer.Option(0.15, help="Annual return volatility"),
    inflation_mean: float = typer.Option(0.03, help="Annual inflation mean for returns"),
    inflation_std: float = typer.Option(0.01, help="Annual inflation std for returns"),
    shock_lambda: float = typer.Option(0.1, help="Poisson lambda for shocks"),
    shock_mean: float = typer.Option(200.0, help="Mean shock cost"),
    seed: Optional[int] = typer.Option(None, help="Random seed"),
    out: Optional[Path] = typer.Option(None, help="Output path for scenario simulation JSON"),
):
    """Apply scenario deltas then run Monte Carlo."""
    forecast_result = _load_forecast(baseline, ledger, months, annual_inflation, engine)
    delta_obj = config_io.load_scenario_delta(delta)
    adjusted_forecast = scenario_service.apply_scenario(forecast_result, delta_obj)
    sim_config = SimulationConfig(
        horizon_months=months,
        paths=paths,
        return_mean=return_mean,
        return_vol=return_vol,
        inflation_mean=inflation_mean,
        inflation_std=inflation_std,
        shock_lambda=shock_lambda,
        shock_mean=shock_mean,
        seed=seed,
    )
    sim_result = simulator.run_monte_carlo(adjusted_forecast, sim_config)
    payload = _simulation_to_json(sim_result)
    if out:
        _save_json(out, payload)
        typer.echo(f"Scenario simulation written to {out}")
    else:
        typer.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    app()

