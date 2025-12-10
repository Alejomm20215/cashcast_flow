import json
from pathlib import Path

from typer.testing import CliRunner

from ftwin_core.cli import app


runner = CliRunner()


def test_cli_forecast_and_simulate(tmp_path: Path):
    ledger_path = tmp_path / "ledger.csv"
    fixture = Path(__file__).parent / "data" / "ledger.csv"
    ledger_path.write_text(fixture.read_text())

    baseline_path = tmp_path / "baseline.json"
    sim_path = tmp_path / "sim.json"

    result_forecast = runner.invoke(
        app,
        [
            "forecast",
            "--ledger",
            str(ledger_path),
            "--months",
            "3",
            "--out",
            str(baseline_path),
        ],
    )
    assert result_forecast.exit_code == 0, result_forecast.stdout
    assert baseline_path.exists()

    result_sim = runner.invoke(
        app,
        [
            "simulate",
            "--baseline",
            str(baseline_path),
            "--paths",
            "200",
            "--return-mean",
            "0.0",
            "--return-vol",
            "0.0",
            "--shock-lambda",
            "0.0",
            "--shock-mean",
            "0.0",
            "--start-wealth",
            "0.0",
            "--goal-target",
            "1000",
            "--liquidity-floor",
            "0",
            "--out",
            str(sim_path),
        ],
    )
    assert result_sim.exit_code == 0, result_sim.stdout
    assert sim_path.exists()

    payload = json.loads(sim_path.read_text())
    assert "percentiles" in payload
    assert payload.get("goal_success_prob") is not None
    assert "liquidity_breach_prob" in payload

