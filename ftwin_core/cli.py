from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

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


def _state_path() -> Path:
    return Path.home() / ".ftwin_interactive.json"


def _load_state() -> dict:
    p = _state_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save_state(payload: dict) -> None:
    try:
        _state_path().write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


def _spark_bar(p10: float, p50: float, p90: float, goal: Optional[float]) -> str:
    """
    Build a tiny bar to visualize spread.
    """
    markers = {"p10": p10, "p50": p50, "p90": p90}
    max_val = max([p90, p50, p10, goal or 0, 1])
    width = 30
    def pos(v: float) -> int:
        return min(width - 1, int((v / max_val) * (width - 1)))
    line = [" "] * width
    line[pos(p10)] = "▏"
    line[pos(p50)] = "▌"
    line[pos(p90)] = "█"
    if goal is not None:
        line[pos(goal)] = "↑"
    return "".join(line)


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
    console = Console()
    console.print("[bold cyan]Financial Twin Quick Simulation[/bold cyan]\n")

    state = _load_state()

    monthly_income = typer.prompt("Monthly take-home income", type=float)
    monthly_expense = typer.prompt("Monthly spending (all-in)", type=float)
    country = typer.prompt("Country (for inflation)", default=state.get("country", "US"))
    months = typer.prompt("Months to simulate", default=state.get("months", 12), type=int)

    start_wealth_in = typer.prompt("Current savings (type a number, blank = 0)", default="", show_default=False)
    start_wealth = float(start_wealth_in) if start_wealth_in.strip() else 0.0

    console.print("\nGrowth presets (optional):")
    console.print("  [green]1[/green] Cash-like (2% return, 0% vol)")
    console.print("  [green]2[/green] Balanced (6% return, 10% vol)")
    console.print("  [green]3[/green] Aggressive (10% return, 18% vol)")
    console.print("  [green]0[/green] Custom (enter your own)")
    preset_choice = typer.prompt("Choose preset (0-3)", default=str(state.get("preset_choice", "0")))

    preset_map = {
        "1": (0.02, 0.0),
        "2": (0.06, 0.10),
        "3": (0.10, 0.18),
    }
    if preset_choice in preset_map:
        expected_return, expected_vol = preset_map[preset_choice]
        console.print(f"Using preset: return {expected_return*100:.1f}%, vol {expected_vol*100:.1f}%")
    else:
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

    # Shocks
    console.print("\nUnexpected expense risk (shocks): 0=None, 1=Low, 2=Med, 3=High")
    shock_choice = typer.prompt("Choose shock level (0-3)", default=str(state.get("shock_choice", "0")))
    shock_map = {
        "0": (0.0, 0.0),
        "1": (0.05, 100.0),
        "2": (0.15, 200.0),
        "3": (0.25, 400.0),
    }
    shock_lambda, shock_mean = shock_map.get(shock_choice, (0.0, 0.0))

    inflation = _default_inflation_for_country(country)
    console.print(f"Using annual inflation [bold]{inflation*100:.2f}%[/bold] for '{country}'.")

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
        shock_lambda=shock_lambda,
        shock_mean=shock_mean,
        start_wealth=start_wealth,
        goal_target=goal_val,
        liquidity_floor=liquidity_floor,
    )
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Running simulation...", total=None)
        result = simulator.run_monte_carlo(forecast, sim_conf)
        progress.update(task, advance=1)

    # Summary output
    net_monthly = monthly_income - monthly_expense
    p10_final = result.percentiles["p10"][-1]
    p50_final = result.percentiles["p50"][-1]
    p90_final = result.percentiles["p90"][-1]

    console.print("\n[bold cyan]== Quick Forecast ==[/bold cyan]")
    console.print(f"Horizon: {months} months | Paths: {sim_conf.paths}")
    console.print(
        f"Assumptions: return [bold]{expected_return*100:.1f}%[/bold]/yr, vol [bold]{expected_vol*100:.1f}%[/bold]/yr, "
        f"shocks λ={shock_lambda:.2f}, mean={shock_mean:,.0f}, expense inflation [bold]{inflation*100:.2f}%[/bold]/yr"
    )
    console.print(f"Avg monthly save/burn: [bold]{net_monthly:,.2f}[/bold]")
    console.print(
        f"Final wealth (approx): P10=[yellow]{p10_final:,.0f}[/yellow], "
        f"P50=[green]{p50_final:,.0f}[/green], P90=[cyan]{p90_final:,.0f}[/cyan]"
    )
    console.print(f"Spread bar: { _spark_bar(p10_final, p50_final, p90_final, goal_val) }")

    if goal_val is not None:
        goal_gap = p50_final - goal_val
        console.print(
            f"Goal: {goal_val:,.0f} | Median end wealth: {p50_final:,.0f} | Gap: "
            f"{('[green]' if goal_gap >=0 else '[red]')}{goal_gap:,.0f}[/]"
        )
    if result.goal_success_prob is not None:
        prob = result.goal_success_prob * 100
        if prob == 0:
            console.print(
                "[yellow]Goal reach chance: 0% (goal sits above the forecast path with current income/spend and growth).[/yellow]"
            )
        else:
            console.print(f"Goal reach chance: [bold]{prob:.1f}%[/bold]")

    if result.liquidity_breach_prob is not None:
        console.print(f"Chance of dropping below floor: [bold]{result.liquidity_breach_prob*100:.1f}%[/bold]")
    console.print("[bold cyan]----------------[/bold cyan]")

    # Guidance
    console.print("\n[green]Tip:[/green]")
    if goal_val is not None and goal_val > 0:
        r_m = expected_return / 12
        annuity = ((1 + r_m) ** months - 1) / r_m if r_m != 0 else months
        deterministic_end = start_wealth * (1 + r_m) ** months + net_monthly * annuity
        gap = goal_val - deterministic_end
        if gap <= 0:
            console.print("Your current plan (median) is enough to reach the goal in expectation.")
        else:
            needed_extra = gap / annuity if annuity > 0 else gap / max(months, 1)
            console.print(
                f"To reach the goal in expectation, add about [bold]{needed_extra:,.0f}/mo[/bold] net "
                f"(or extend horizon / increase return)."
            )
    else:
        console.print("Add a goal to see how far off you are, or adjust return/volatility for growth assumptions.")

    console.print("\n[bold cyan]Done.[/bold cyan]\n")

    # Save state
    _save_state(
        {
            "country": country,
            "months": months,
            "preset_choice": preset_choice,
            "shock_choice": shock_choice,
        }
    )


if __name__ == "__main__":
    app()

