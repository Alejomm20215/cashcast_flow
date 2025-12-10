import datetime as dt

from ftwin_core.domain.models import ForecastConfig, ForecastResult, MonthlyForecast, SimulationConfig
from ftwin_core.services.simulator import run_monte_carlo


def _forecast_with_net(net: float, months: int = 6) -> ForecastResult:
    monthly = [
        MonthlyForecast(month=dt.date(2024, 1, 1) + dt.timedelta(days=30 * i), income=net, expense=0)
        for i in range(months)
    ]
    return ForecastResult(config=ForecastConfig(months=months), monthly=monthly)


def test_goal_success_probability_hits_target():
    forecast = _forecast_with_net(100.0, months=6)  # deterministic nets
    config = SimulationConfig(
        horizon_months=6,
        paths=500,
        return_mean=0.0,
        return_vol=0.0,
        shock_lambda=0.0,
        shock_mean=0.0,
        start_wealth=0.0,
        goal_target=500.0,
        liquidity_floor=None,
        seed=123,
    )
    result = run_monte_carlo(forecast, config)
    assert result.goal_success_prob is not None
    assert abs(result.goal_success_prob - 1.0) < 1e-9


def test_liquidity_breach_probability_detects_breach():
    # negative net ensures breach when floor is zero
    forecast = _forecast_with_net(-50.0, months=4)
    config = SimulationConfig(
        horizon_months=4,
        paths=500,
        return_mean=0.0,
        return_vol=0.0,
        shock_lambda=0.0,
        shock_mean=0.0,
        start_wealth=100.0,
        goal_target=None,
        liquidity_floor=0.0,
        seed=42,
    )
    result = run_monte_carlo(forecast, config)
    assert result.liquidity_breach_prob is not None
    assert result.liquidity_breach_prob == 1.0

