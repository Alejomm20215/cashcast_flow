# Financial Twin CLI & Core

Python/CLI-first toolkit for forecasting cashflow, Monte Carlo simulations, scenario testing, and AI-personalized tips. Includes Django API + Celery worker. Proprietary, non-free (see `LICENSE.md`).

---

## Table of Contents
- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Install](#install)
- [CLI Quickstart](#cli-quickstart)
- [Interactive Mode](#interactive-mode)
- [AI Suggestions](#ai-suggestions)
- [Memory & Privacy](#memory--privacy)
- [Django API](#django-api)
- [Docker](#docker)
- [Data Formats](#data-formats)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)
- [Licensing](#licensing)

---

## What it does
- Forecasts: naive, Prophet, or ensemble (naive + Prophet + ARIMA).
- Simulations: Monte Carlo with goal success and liquidity breach metrics.
- Scenarios: apply income/expense deltas and compare baseline vs scenario.
- Interactive mode: captures per-user memory, optional AI suggestions (LangChain + HF/local endpoint).
- Backend: Django API endpoints with async Celery worker; Redis broker.

## Architecture
- Layer 1: Forecasting — naive/Prophet/ensemble, inflation-aware.
- Layer 2: Stochastic simulation — returns (GBM-style), shocks (Poisson), inflation; outputs p10/p50/p90, goal/liq metrics.
- Layer 3: Scenario generator — applies deltas, re-runs simulations, compares vs baseline.
- LLM layer: LangChain-powered suggestions using model outputs + user profile.
- Persistence: local per-user memory (`~/.ftwin_memory.json`) for derived stats.

## Install
```bash
pip install -r requirements.txt
# optional dev
pip install -e ".[dev]"
```
Requires Python 3.11+. Prophet/ARIMA need build tools (covered by Dockerfile if using containers).

## CLI Quickstart
- Baseline (naive): `ftwin forecast --ledger data/ledger.csv --months 12 --out baseline.json`
- Baseline (Prophet): `ftwin forecast --ledger data/ledger.csv --engine prophet --out baseline.json`
- Baseline (ensemble): `ftwin forecast --ledger data/ledger.csv --ensemble --out baseline.json`
- Simulate: `ftwin simulate --baseline baseline.json --paths 5000 --start-wealth 0 --goal-target 50000 --liquidity-floor 0 --out sim.json`
- Scenario: `ftwin scenario --delta delta.json --baseline baseline.json --out scen.json`
- Interactive: `ftwin interactive` (builds/uses local memory; optional AI suggestions)
- Train memory: `ftwin train --ledger data/ledger.csv`
- Show memory: `ftwin memory`

Outputs include: p10/p50/p90 wealth trajectories, `goal_success_prob`, `liquidity_breach_prob`.

## Interactive Mode
- If no profile, prompts a quick profile and saves to `~/.ftwin_memory.json`.
- Can prompt for expense breakdown (rent, utilities, groceries, dining/coffee, subscriptions, transport, fun).
- Uses saved memory to prefill defaults; runs simulation; optionally generates AI suggestions.

## AI Suggestions
- Enabled when `HF_TOKEN` is set (optionally `HF_MODEL`, default `mistralai/Mistral-7B-Instruct-v0.2`).
- LangChain calls Hugging Face Inference (or a compatible endpoint) with structured context from the simulation (p10/p50/p90, goal gap, net/month), profile (savings rate, creep, occupation), shocks, return/vol, inflation, and expense breakdown.
- If no token, suggestions are skipped with a notice.
- To use a local or custom endpoint, adjust `ftwin_core/services/suggestions.py`.

## Memory & Privacy
- Per-user memory stored at `~/.ftwin_memory.json` (gitignored). Contains derived stats: savings rate, lifestyle creep, category means/std, estimates.
- No raw PII required. Keep ledgers out of version control.
- Interactive flow reads memory to set defaults; `ftwin memory` shows stored values.

## Django API
- POST `/api/simulations/` (multipart: ledger file, params) → async run id (Celery).
- POST `/api/scenarios/` (ledger + delta) → async scenario run.
- GET `/api/runs/<id>/` → status + result JSON.
- Services: Django + DRF; Celery worker; Redis broker.

## Docker
```bash
docker-compose up --build
```
Services: `web` (Django), `worker` (Celery), `redis`. The compose file omits deprecated version keys.

## Data Formats

### Ledger CSV
Required columns: `date,amount,category,kind` (`kind` = income|expense).

Example:
```csv
date,amount,category,kind
2024-01-15,3500.00,salary,income
2024-01-20,-1200.00,rent,expense
2024-01-22,-400.00,groceries,expense
```

### Scenario Delta JSON
```json
{
  "income_deltas": {"raise": 500},
  "expense_deltas": {"netflix": -19.99, "coffee": -60},
  "savings_rate_override": 0.8
}
```

### Simulation Output JSON
```json
{
  "config": {
    "horizon_months": 12,
    "paths": 5000,
    "return_mean": 0.07,
    "return_vol": 0.15,
    "start_wealth": 0.0,
    "goal_target": 50000.0,
    "liquidity_floor": 0.0
  },
  "percentiles": {
    "p10": [1000, 2000],
    "p50": [1500, 3000],
    "p90": [2000, 4000]
  },
  "goal_success_prob": 0.75,
  "liquidity_breach_prob": 0.05
}
```

## Environment Variables
- `HF_TOKEN`: Hugging Face token for AI suggestions.
- `HF_MODEL`: Model id (default `mistralai/Mistral-7B-Instruct-v0.2`).
- Django/Celery: `DJANGO_SETTINGS_MODULE`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` (see compose).

## Troubleshooting

### Prophet/ARIMA Installation
- **Windows**: Install Visual C++ Build Tools first, then `pip install prophet`
- **Linux/Mac**: Ensure build tools installed (`build-essential` on Debian/Ubuntu, `cmake` on macOS)
- **Docker**: Dockerfile includes all dependencies; use `docker-compose up` for consistent environment

### AI Suggestions Not Working
1. Check `HF_TOKEN` is set: `echo $HF_TOKEN`
2. Verify network access to Hugging Face API
3. Check model ID is valid
4. Review logs for API errors

### Memory File Not Found
- Run `ftwin train --ledger <file>` to create memory
- Or answer quick profile in `ftwin interactive`
- Check `~/.ftwin_memory.json` exists and is readable

### Simulation Results Seem Wrong
1. Verify input data: check ledger CSV format
2. Check simulation config: return/vol assumptions reasonable?
3. Increase path count: 5000+ for stable results
4. Review forecast output: does monthly net make sense?

## Examples

### Basic Workflow
```bash
# 1. Generate baseline forecast
ftwin forecast --ledger transactions.csv --months 12 --out baseline.json

# 2. Run simulation with goal
ftwin simulate --baseline baseline.json --paths 5000 \
  --start-wealth 5000 --goal-target 50000 --out sim.json

# 3. Inspect results
cat sim.json | jq '.percentiles.p50[-1]'  # Final median wealth
cat sim.json | jq '.goal_success_prob'    # Goal success probability
```

### Scenario Testing
Create `delta.json`:
```json
{
  "income_deltas": {"raise": 500},
  "expense_deltas": {"netflix": -19.99, "coffee": -60},
  "savings_rate_override": 0.8
}
```

Run scenario:
```bash
ftwin scenario --delta delta.json --baseline baseline.json --out scen.json
```

### Interactive Mode with AI
```bash
export HF_TOKEN=hf_...
ftwin interactive
# Answer prompts; get AI-generated suggestions at the end
```

## Python Integration

```python
from ftwin_core.io import ledger as ledger_io
from ftwin_core.services import forecaster_ensemble, simulator
from ftwin_core.domain.models import ForecastConfig, SimulationConfig

# Load ledger
entries = ledger_io.load_ledger("transactions.csv")

# Forecast
config = ForecastConfig(months=12, annual_inflation=0.02)
forecast = forecaster_ensemble.forecast_cashflow_ensemble(entries, config)

# Simulate
sim_config = SimulationConfig(
    horizon_months=12,
    paths=5000,
    return_mean=0.07,
    return_vol=0.15,
    start_wealth=5000,
    goal_target=50000,
    liquidity_floor=1000
)
result = simulator.run_monte_carlo(forecast, sim_config)

# Access results
p50_final = result.percentiles["p50"][-1]
goal_prob = result.goal_success_prob
print(f"Median final wealth: ${p50_final:,.0f}")
print(f"Goal success probability: {goal_prob*100:.1f}%")
```

## Performance Notes
- **Forecasting**: Naive <1s, Prophet 5-30s, Ensemble 10-60s
- **Simulation**: 2000 paths ~1-2s, 5000 paths ~3-5s, 10000 paths ~6-10s
- **Memory**: Forecasting <100MB, Simulation 50-200MB, Ensemble 200-500MB

## Development & Tests
```bash
pytest
```

## Kaggle LoRA Helper
Generate a GPU-requesting notebook for small LoRA fine-tunes:
```bash
python scripts/generate_kaggle_lora_notebook.py
```
Upload `kaggle_lora_finetune.ipynb` to Kaggle and select GPU in the sidebar (when available).

## Licensing
Proprietary, non-free. Internal evaluation/development use only. See `LICENSE.md` for full terms. Contact the authors for commercial licensing.

---

*For detailed architecture and advanced usage, see `PACKAGE_OVERVIEW.md`.*

*Last updated: 2024-12-10*
