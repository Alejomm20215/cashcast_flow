from __future__ import annotations

import numpy as np

from ftwin_core.domain.models import ForecastResult, SimulationConfig, SimulationResult


def run_monte_carlo(forecast: ForecastResult, config: SimulationConfig) -> SimulationResult:
    """
    Vectorized Monte Carlo over monthly steps using GBM-like returns plus shocks.
    """
    if config.seed is not None:
        np.random.seed(config.seed)

    horizon = min(config.horizon_months, len(forecast.monthly))
    nets = np.array([m.net for m in forecast.monthly[:horizon]], dtype=float)

    mu_m = config.return_mean / 12
    sigma_m = config.return_vol / np.sqrt(12)

    returns = np.random.normal(mu_m, sigma_m, size=(config.paths, horizon))
    shocks = np.random.poisson(config.shock_lambda, size=(config.paths, horizon)) * config.shock_mean

    wealth = np.zeros((config.paths, horizon))
    wealth[:, 0] = config.start_wealth

    for t in range(horizon):
        prev = wealth[:, t - 1] if t > 0 else wealth[:, 0]
        wealth[:, t] = prev * (1 + returns[:, t]) + nets[t] - shocks[:, t]

    percentiles = {
        "p10": np.percentile(wealth, 10, axis=0).tolist(),
        "p50": np.percentile(wealth, 50, axis=0).tolist(),
        "p90": np.percentile(wealth, 90, axis=0).tolist(),
    }

    goal_success_prob = None
    if config.goal_target is not None:
        goal_success_prob = float(np.mean(wealth[:, -1] >= config.goal_target))

    liquidity_breach_prob = None
    if config.liquidity_floor is not None:
        breaches = (wealth < config.liquidity_floor).any(axis=1)
        liquidity_breach_prob = float(np.mean(breaches))

    return SimulationResult(
        config=config,
        percentiles=percentiles,
        goal_success_prob=goal_success_prob,
        liquidity_breach_prob=liquidity_breach_prob,
    )

