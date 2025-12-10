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
from ftwin_core.services import forecaster, forecaster_ensemble, scenario as scenario_service
from ftwin_core.services import forecaster_prophet
from ftwin_core.services import simulator
from ftwin_core.services import training
from ftwin_core.services import suggestions

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


def _memory_path() -> Path:
    return Path.home() / ".ftwin_memory.json"


def _read_memory() -> Optional[dict]:
    p = _memory_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _generate_suggestions(
    income: float,
    expense: float,
    savings_rate: Optional[float],
    lifestyle_creep: Optional[float],
    horizon_months: int,
    goal: Optional[float],
    occupation: str,
    breakdown: Optional[dict],
    sim_result,
    sim_conf: SimulationConfig,
    inflation: float,
) -> list[str]:
    goal_gap = None
    if goal is not None:
        goal_gap = sim_result.percentiles["p50"][-1] - goal
    ai_tips = []
    try:
        ai_tips = suggestions.generate_ai_suggestions(
            net_monthly=income - expense,
            p10=sim_result.percentiles["p10"][-1],
            p50=sim_result.percentiles["p50"][-1],
            p90=sim_result.percentiles["p90"][-1],
            goal=goal,
            goal_gap=goal_gap,
            horizon_months=horizon_months,
            return_mean=sim_conf.return_mean,
            return_vol=sim_conf.return_vol,
            shock_lambda=sim_conf.shock_lambda,
            shock_mean=sim_conf.shock_mean,
            inflation=inflation,
            savings_rate=savings_rate,
            lifestyle_creep=lifestyle_creep,
            occupation=occupation,
            breakdown=breakdown,
        )
    except Exception:
        ai_tips = []
    return ai_tips


def _prompt_expense_breakdown(console: Console, base_spend: float):
    console.print("[cyan]Enter monthly amounts (leave blank to skip a category).[/cyan]")
    rent = typer.prompt("  Rent/mortgage", default="", show_default=False)
    utilities = typer.prompt("  Utilities (power/water/internet)", default="", show_default=False)
    groceries = typer.prompt("  Groceries", default="", show_default=False)
    dining = typer.prompt("  Eating out / coffee", default="", show_default=False)
    subs = typer.prompt("  Subscriptions (Netflix/Spotify/etc.)", default="", show_default=False)
    transport = typer.prompt("  Transport", default="", show_default=False)
    fun = typer.prompt("  Fun/parties/other", default="", show_default=False)

    def to_val(v: str) -> float:
        return float(v) if v.strip() else 0.0

    total = sum(
        [
            to_val(rent),
            to_val(utilities),
            to_val(groceries),
            to_val(dining),
            to_val(subs),
            to_val(transport),
            to_val(fun),
        ]
    )
    if total <= 0:
        console.print("[yellow]Keeping previous spending amount.[/yellow]")
        return base_spend, None
    console.print(f"[green]Using category total: {total:,.2f}[/green]")
    breakdown = {
        "rent": to_val(rent),
        "utilities": to_val(utilities),
        "groceries": to_val(groceries),
        "dining": to_val(dining),
        "subscriptions": to_val(subs),
        "transport": to_val(transport),
        "fun": to_val(fun),
    }
    return total, breakdown


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
    ensemble: bool = typer.Option(False, help="Use ensemble forecaster (naive+prophet+arima)"),
    out: Optional[Path] = typer.Option(None, help="Output path for baseline forecast JSON"),
):
    """Generate baseline cashflow forecast."""
    entries = ledger_io.load_ledger(ledger)
    config = ForecastConfig(months=months, annual_inflation=annual_inflation)
    if ensemble:
        result = forecaster_ensemble.forecast_cashflow_ensemble(entries, config)
    elif engine == "prophet":
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
    ensemble: bool = False,
):
    if baseline:
        with baseline.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return _forecast_from_json(data)
    if ledger:
        entries = ledger_io.load_ledger(ledger)
        config = ForecastConfig(months=months, annual_inflation=annual_inflation)
        if ensemble:
            return forecaster_ensemble.forecast_cashflow_ensemble(entries, config)
        if engine == "prophet":
            return forecaster_prophet.forecast_cashflow_prophet(entries, config)
        return forecaster.forecast_cashflow(entries, config)
    raise typer.BadParameter("Provide either --baseline or --ledger")


@app.command()
def train(
    ledger: Path = typer.Option(..., help="CSV ledger with date,amount,category,kind"),
):
    """
    Train per-user lightweight stats/models and store to local memory (~/.ftwin_memory.json).
    """
    entries = ledger_io.load_ledger(ledger)
    memory = training.train_user_models(entries)
    typer.echo(f"Trained and stored memory at {_memory_path()}")
    typer.echo(json.dumps(memory, indent=2))


@app.command()
def memory():
    """
    Show stored per-user memory (if available).
    """
    mem = _read_memory()
    if not mem:
        typer.echo("No memory found. Run: ftwin train --ledger <file>")
        return
    typer.echo(json.dumps(mem, indent=2))


@app.command()
def simulate(
    baseline: Optional[Path] = typer.Option(None, help="Baseline forecast JSON"),
    ledger: Optional[Path] = typer.Option(None, help="Ledger CSV (used if baseline not supplied)"),
    months: int = typer.Option(12, help="Months to forecast if ledger is used"),
    annual_inflation: float = typer.Option(0.02, help="Annual inflation rate if ledger is used"),
    engine: str = typer.Option("naive", help="Forecaster engine when using ledger: naive|prophet"),
    ensemble: bool = typer.Option(False, help="Use ensemble forecaster when using ledger"),
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
    forecast_result = _load_forecast(baseline, ledger, months, annual_inflation, engine, ensemble)
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
    ensemble: bool = typer.Option(False, help="Use ensemble forecaster when using ledger"),
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
    mem = _read_memory()

    # If no memory, ask a few questions to generate it quickly
    if not mem:
        console.print("[yellow]No profile found. Let's capture a quick profile to personalize defaults.[/yellow]")
        months_history = typer.prompt("How many months of history do you roughly have?", default=12, type=int)
        est_income = typer.prompt("Rough average monthly income?", type=float)
        est_expense = typer.prompt("Rough average monthly spending (all-in)?", type=float)
        est_savings_rate = typer.prompt("Rough savings rate (e.g., 0.1 = 10%, blank if unsure)", default="", show_default=False)
        est_creep = typer.prompt("Do you usually spend more when income rises? (0-1, blank if unsure)", default="", show_default=False)

        sr_val = _parse_rate(est_savings_rate) if est_savings_rate.strip() else max((est_income - est_expense) / est_income, 0) if est_income else 0.0
        creep_val = _parse_rate(est_creep) if est_creep.strip() else 0.0

        quick_mem = {
            "savings_rate": sr_val,
            "lifestyle_creep": creep_val,
            "categories": {},
            "months_observed": months_history,
            "est_income": est_income,
            "est_expense": est_expense,
            "meta": {"source": "quick_profile"},
        }
        _memory_path().write_text(json.dumps(quick_mem, indent=2))
        mem = quick_mem
        console.print("[green]Profile stored locally. You can refine it later with ftwin train --ledger <file>.[/green]")
    else:
        console.print(
            f"Loaded your profile: savings_rate ~{mem.get('savings_rate', 0)*100:.0f}%, "
            f"lifestyle_creep ~{mem.get('lifestyle_creep', 0)*100:.0f}%"
        )

    default_income = mem.get("est_income") if mem else None
    default_expense = mem.get("est_expense") if mem else None

    monthly_income = typer.prompt(
        "Monthly take-home income",
        default=default_income if default_income else None,
        type=float,
    )
    monthly_expense = typer.prompt(
        "Monthly spending (all-in)",
        default=default_expense if default_expense else None,
        type=float,
    )

    # Optional category breakdown
    if typer.confirm("Add a quick breakdown (rent/utilities/food/dining/subscriptions/fun)?", default=False):
        monthly_expense, breakdown = _prompt_expense_breakdown(console, base_spend=monthly_expense)
    else:
        breakdown = None

    country = typer.prompt("Country (for inflation)", default=state.get("country", "US"))
    months = typer.prompt("Months to simulate", default=state.get("months", 12), type=int)
    occupation = typer.prompt("Your field/role (optional, e.g., psychologist/programmer)", default="", show_default=False)

    start_wealth_in = typer.prompt("Current savings (type a number, blank = 0)", default="", show_default=False)
    start_wealth = float(start_wealth_in) if start_wealth_in.strip() else 0.0

    console.print("\nGrowth presets (optional):")
    console.print("  [green]1[/green] Cash-like (2% return, 0% vol)")
    console.print("  [green]2[/green] Balanced (6% return, 10% vol)")
    console.print("  [green]3[/green] Aggressive (10% return, 18% vol)")
    console.print("  [green]0[/green] Custom (enter your own)")
    default_preset = str(state.get("preset_choice", "0"))
    if mem and "savings_rate" in mem:
        sr = mem["savings_rate"]
        if sr < 0.05:
            default_preset = "1"
        elif sr < 0.15:
            default_preset = "2"
        else:
            default_preset = "2"
    preset_choice = typer.prompt("Choose preset (0-3)", default=default_preset)

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

    # Extra suggestions
    ai_tips = _generate_suggestions(
        income=monthly_income,
        expense=monthly_expense,
        savings_rate=mem.get("savings_rate") if mem else None,
        lifestyle_creep=mem.get("lifestyle_creep") if mem else None,
        horizon_months=months,
        goal=goal_val,
        occupation=occupation,
        breakdown=breakdown,
        sim_result=result,
        sim_conf=sim_conf,
        inflation=inflation,
    )
    if ai_tips:
        console.print("[bold magenta]AI-personalized ideas:[/bold magenta]")
        for s in ai_tips:
            console.print(f"- {s}")
        console.print()
    else:
        console.print("[yellow]AI suggestions unavailable (set HF_TOKEN to enable).[/yellow]")


if __name__ == "__main__":
    app()

