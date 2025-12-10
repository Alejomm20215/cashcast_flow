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
        "liquidity_breach_prob": result.liquidity_breach_prob,
    }


# -------------------------------
# Interactive helper (no JSON)
# -------------------------------


def _default_inflation_for_country(country: str) -> float:
    country = country.strip().lower()
    presets = {
        "us": 0.03,
        "usa": 0.03,
        "united states": 0.03,
        "uk": 0.025,
        "united kingdom": 0.025,
        "ca": 0.028,
        "canada": 0.028,
        "mx": 0.04,
        "mexico": 0.04,
        "eu": 0.025,
        "europe": 0.025,
        "co": 0.05,
        "colombia": 0.05,
    }
    return presets.get(country, 0.03)


def _parse_rate(raw: str) -> float:
    """
    Parse a rate string that may contain a percent sign or plain float.
    Accepts "0.08", "8%", or "8" (treated as 8%).
    """
    txt = raw.strip().replace("%", "")
    if not txt:
        return 0.0
    try:
        val = float(txt)
    except ValueError:
        return 0.0
    return val / 100.0 if val > 1 else val


def _build_constant_forecast(income: float, expense: float, months: int, annual_inflation: float):
    monthly = []
    inf_m = (1 + annual_inflation) ** (1 / 12) - 1
    today = date.today()
    start = date(today.year, today.month, 1)
    import pandas as pd  # local import to avoid global dep in prompts

    for i in range(months):
        m_date = (pd.Period(start, freq="M") + i).to_timestamp().date()
        expense_adj = expense * ((1 + inf_m) ** i)
        monthly.append(MonthlyForecast(month=m_date, income=income, expense=expense_adj))
    return ForecastResult(config=ForecastConfig(months=months, annual_inflation=annual_inflation), monthly=monthly)


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
    start_wealth: float = typer.Option(0.0, help="Starting wealth for simulation paths"),
    goal_target: Optional[float] = typer.Option(None, help="Goal target wealth at end of horizon"),
    liquidity_floor: Optional[float] = typer.Option(None, help="Liquidity floor; breach probability is reported"),
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
        start_wealth=start_wealth,
        goal_target=goal_target,
        liquidity_floor=liquidity_floor,
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
    start_wealth: float = typer.Option(0.0, help="Starting wealth for simulation paths"),
    goal_target: Optional[float] = typer.Option(None, help="Goal target wealth at end of horizon"),
    liquidity_floor: Optional[float] = typer.Option(None, help="Liquidity floor; breach probability is reported"),
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
        start_wealth=start_wealth,
        goal_target=goal_target,
        liquidity_floor=liquidity_floor,
        seed=seed,
    )
    sim_result = simulator.run_monte_carlo(adjusted_forecast, sim_config)
    payload = _simulation_to_json(sim_result)
    if out:
        _save_json(out, payload)
        typer.echo(f"Scenario simulation written to {out}")
    else:
        typer.echo(json.dumps(payload, indent=2))


@app.command()
def interactive():
    """
    Interactive mode: simple questions, quick simulation (no JSON needed).
    """
    typer.echo("Financial Twin Quick Simulation\n")

    monthly_income = typer.prompt("Monthly take-home income", type=float)
    monthly_expense = typer.prompt("Monthly spending (all-in)", type=float)
    country = typer.prompt("Country (for inflation)", default="US")
    months = typer.prompt("Months to simulate", default=12, type=int)

    start_wealth_in = typer.prompt("Current savings (type a number, blank = 0)", default="", show_default=False)
    start_wealth = float(start_wealth_in) if start_wealth_in.strip() else 0.0

    expected_return_in = typer.prompt(
        "Expected annual growth on savings/investments (e.g., 0.05 = 5%, blank = 0%)",
        default="",
        show_default=False,
    )
    expected_return = _parse_rate(expected_return_in)

    expected_vol_in = typer.prompt(
        "Expected annual volatility (e.g., 0.10 = 10%, blank = 0%)",
        default="",
        show_default=False,
    )
    expected_vol = _parse_rate(expected_vol_in)

    goal_target = typer.prompt("Goal at end (blank = skip)", default="", show_default=False)
    goal_val = float(goal_target) if goal_target.strip() else None

    liquidity_val = typer.prompt("Liquidity floor (blank = skip)", default="", show_default=False)
    liquidity_floor = float(liquidity_val) if liquidity_val.strip() else None

    inflation = _default_inflation_for_country(country)
    typer.echo(f"Using annual inflation {inflation*100:.2f}% for '{country}'.")

    forecast = _build_constant_forecast(
        income=monthly_income,
        expense=monthly_expense,
        months=months,
        annual_inflation=inflation,
    )

    # Optional growth assumptions from user
    sim_conf = SimulationConfig(
        horizon_months=months,
        paths=2000,
        return_mean=expected_return,
        return_vol=expected_vol,
        shock_lambda=0.0,
        shock_mean=0.0,
        start_wealth=start_wealth,
        goal_target=goal_val,
        liquidity_floor=liquidity_floor,
    )
    result = simulator.run_monte_carlo(forecast, sim_conf)

    # Summary output
    net_monthly = monthly_income - monthly_expense
    p10_final = result.percentiles["p10"][-1]
    p50_final = result.percentiles["p50"][-1]
    p90_final = result.percentiles["p90"][-1]

    typer.secho("\n=== Quick Forecast ===", fg="cyan")
    typer.echo(f"Horizon: {months} months | Paths: {sim_conf.paths}")
    typer.echo(
        f"Assumptions: return {expected_return*100:.1f}%/yr, vol {expected_vol*100:.1f}%/yr, shocks 0%, "
        f"expense inflation {inflation*100:.2f}%/yr"
    )
    typer.echo(f"Avg monthly save/burn: {net_monthly:,.2f}")
    typer.echo(f"Final wealth (approx): P10={p10_final:,.0f}, P50={p50_final:,.0f}, P90={p90_final:,.0f}")

    if goal_val is not None:
        goal_gap = p50_final - goal_val
        typer.echo(f"Goal: {goal_val:,.0f} | Median end wealth: {p50_final:,.0f} | Gap: {goal_gap:,.0f}")
    if result.goal_success_prob is not None:
        prob = result.goal_success_prob * 100
        if prob == 0:
            typer.secho(
                "Goal reach chance: 0% (given current income/spend, inflation, and return assumptions, the goal sits above the forecast path).",
                fg="yellow",
            )
        else:
            typer.echo(f"Goal reach chance: {prob:.1f}%")

    if result.liquidity_breach_prob is not None:
        typer.echo(f"Chance of dropping below floor: {result.liquidity_breach_prob*100:.1f}%")
    typer.secho("----------------", fg="cyan")

    # Guidance
    typer.secho("\nTip:", fg="green")
    if goal_val is not None and goal_val > 0:
        needed_net = (goal_val - start_wealth) / months if months > 0 else 0
        delta = needed_net - net_monthly
        if delta > 0:
            typer.echo(
                f"To be on track for the goal without growth, you'd need about {needed_net:,.0f}/mo net; "
                f"that's {delta:,.0f} more per month than now."
            )
        else:
            typer.echo("Your current net/month is sufficient to hit the goal in a straight-line scenario.")
    else:
        typer.echo("Add a goal to see how far off you are, or adjust return/volatility for growth assumptions.")

    typer.secho("\nDone.\n", fg="cyan")


if __name__ == "__main__":
    app()

