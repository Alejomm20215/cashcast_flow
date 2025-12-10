# Financial Twin CLI & Core

Python/CLI-first toolkit for forecasting cashflow, Monte Carlo simulations, scenario testing, and AI-personalized tips. Includes Django API + Celery worker. Licensed under a proprietary, non-free license (see `LICENSE.md`).

## What it does
- Forecasts: naive, Prophet, or ensemble (naive+Prophet+ARIMA).
- Simulations: Monte Carlo with goal success and liquidity breach metrics.
- Scenarios: apply income/expense deltas and compare baseline vs scenario.
- Interactive mode: captures per-user memory, optional AI-generated suggestions (LangChain + HF/local endpoint).
- Backend: Django API endpoints with async Celery worker; Redis broker.

## Requirements
- Python 3.11+
- `pip install -r requirements.txt`
- For Prophet/ARIMA: system build tools (already in Dockerfile).

## Quick start (CLI)
- Baseline (naive): `ftwin forecast --ledger data/ledger.csv --months 12 --out baseline.json`
- Baseline (Prophet): `ftwin forecast --ledger data/ledger.csv --engine prophet --out baseline.json`
- Baseline (ensemble): `ftwin forecast --ledger data/ledger.csv --ensemble --out baseline.json`
- Simulate: `ftwin simulate --baseline baseline.json --paths 5000 --start-wealth 0 --goal-target 50000 --liquidity-floor 0 --out sim.json`
- Scenario: `ftwin scenario --delta delta.json --baseline baseline.json --out scen.json`
- Interactive: `ftwin interactive` (builds/uses local memory; optional AI suggestions)
- Train memory: `ftwin train --ledger data/ledger.csv`
- Show memory: `ftwin memory`

Ledger CSV columns: `date,amount,category,kind` (`kind` = income|expense).

Outputs include percentiles (p10/p50/p90), `goal_success_prob`, `liquidity_breach_prob`.

### AI suggestions (optional)
- Set `HF_TOKEN` (and optionally `HF_MODEL`, default `mistralai/Mistral-7B-Instruct-v0.2`) to enable AI-generated suggestions in `ftwin interactive`.
- If unset, suggestions are skipped.
- For local/other endpoints, adjust LangChain config in `ftwin_core/services/suggestions.py`.

## Django API (brief)
- POST `/api/simulations/` (multipart: ledger file, params) → async run id.
- POST `/api/scenarios/` (ledger + delta) → async scenario run.
- GET `/api/runs/<id>/` → status + result JSON.

## Docker
```bash
docker-compose up --build
```
Services: `web` (Django), `worker` (Celery), `redis`.

## Dev & tests
```bash
pytest
```

## Kaggle LoRA helper
Generate a GPU-requesting Kaggle notebook for small LoRA fine-tunes:
```bash
python scripts/generate_kaggle_lora_notebook.py
```
Upload `kaggle_lora_finetune.ipynb` to Kaggle and select GPU in the sidebar (when available).

## Notes on memory & privacy
- Per-user memory stored locally at `~/.ftwin_memory.json` (gitignored). Contains derived stats (savings rate, creep, category means/std).
- No PII is required; keep raw ledgers out of version control.

## Licensing
Proprietary, non-free. Internal evaluation/development use only. See `LICENSE.md` for details. Contact authors for commercial terms.

