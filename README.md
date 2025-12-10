# Financial Twin CLI and Core

CLI-first package for forecasting cashflow, running Monte Carlo simulations, and testing scenarios. Python/Django friendly; exposes Typer CLI (`ftwin`) and reusable services under `ftwin_core`.

## Quick CLI usage
- Baseline (naive): `ftwin forecast --ledger data/ledger.csv --months 12 --out baseline.json`
- Baseline (Prophet): `ftwin forecast --ledger data/ledger.csv --engine prophet --out baseline.json`
- Simulate: `ftwin simulate --baseline baseline.json --paths 5000 --out sim.json`
- Scenario: `ftwin scenario --delta delta.json --baseline baseline.json --out scen.json`

Ledger CSV requires columns: `date,amount,category,kind` where `kind` is `income|expense`.

## Django API
- POST `/api/simulations/` with multipart `{ledger: file, months, ...}` → returns run id (async Celery job).
- POST `/api/scenarios/` with `{ledger: file, delta: file, ...}` → async scenario sim run.
- GET `/api/runs/<id>/` → run status and result payload (when completed).

