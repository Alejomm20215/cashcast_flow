from __future__ import annotations

from ftwin_core.domain.models import ScenarioComparison, ScenarioDelta, SimulationConfig
from ftwin_core.services import scenario as scenario_service
from ftwin_core.services import simulator


def compare_scenario(base_forecast, base_sim_config: SimulationConfig, delta: ScenarioDelta) -> ScenarioComparison:
    baseline_result = simulator.run_monte_carlo(base_forecast, base_sim_config)
    scenario_forecast = scenario_service.apply_scenario(base_forecast, delta)
    scenario_result = simulator.run_monte_carlo(scenario_forecast, base_sim_config)

    delta_percentiles = {}
    for key in baseline_result.percentiles.keys():
        base = baseline_result.percentiles[key]
        scen = scenario_result.percentiles[key]
        delta_percentiles[key] = [s - b for s, b in zip(scen, base)]

    return ScenarioComparison(
        baseline=baseline_result,
        scenario=scenario_result,
        delta=delta_percentiles,
    )

