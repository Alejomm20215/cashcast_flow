# Package Overview

This project implements a “Financial Twin” with a 3-layer model, an AI suggestions layer, and CLI/API interfaces. It also supports per-user memory and optional model adaptation.

## Three-Layer Model

### Layer 1: Forecasting (Data + ML)
- Forecasters: naive mean, Prophet, or ensemble (naive + Prophet + ARIMA).
- Inputs: categorized ledger (`date, amount, category, kind`).
- Outputs: monthly income/expense forecasts; inflation-aware.
- Per-user memory: derived stats (savings rate, lifestyle creep, category means/std) stored at `~/.ftwin_memory.json`.

### Layer 2: Stochastic Simulation
- Monte Carlo over forecasted cashflows.
- Models returns (GBM-style), shocks (Poisson), inflation, starting wealth.
- Metrics: p10/p50/p90 trajectories, goal success probability, liquidity breach probability.

### Layer 3: Scenario Generator
- Applies deltas (income changes, expense adds/removals, savings split overrides).
- Re-runs simulations; compares baseline vs scenario (percentiles, goal/liq metrics).

## AI Suggestions (LLM Layer)
- LangChain-based; consumes simulation outputs (p10/p50/p90, goal gap), profile (savings rate, creep, occupation), shocks, return/vol, inflation, optional expense breakdown.
- Default: Hugging Face endpoint (config via `HF_TOKEN`, `HF_MODEL`), can point to local/hosted TGI.
- If no token, suggestions are skipped. Prompt enforces concise, actionable bullets; no PII needed.

## CLI Commands
- `ftwin forecast` — baseline (naive/prophet/ensemble).
- `ftwin simulate` — Monte Carlo on a baseline or ledger + forecaster.
- `ftwin scenario` — apply deltas then simulate.
- `ftwin interactive` — guided Q&A, per-user memory, optional AI suggestions.
- `ftwin train` — build per-user memory from a ledger.
- `ftwin memory` — view stored per-user memory.

### Data Formats
- Ledger CSV: `date,amount,category,kind` (`kind` = income|expense).
- Scenario delta JSON: income/expense deltas; optional savings overrides.

## Backend (Django + Celery)
- DRF endpoints for simulations and scenarios.
- Celery workers for async jobs; Redis as broker.

## Learning & Privacy
- Per-user memory stored locally (gitignored); no raw PII required.
- Optional adaptation: collect de-identified prompt/feedback to fine-tune a small LoRA adapter and serve via HF/TGI.

## LLM & Adaptation (optional)
- Inference: HF Inference or self-hosted TGI; configurable model id.
- Adaptation: fine-tune a LoRA on de-identified interaction data (e.g., via Kaggle/Colab/RunPod), then load adapter on the serving endpoint.

---

## Detailed Architecture

### Layer 1: Enhanced Data Foundation & Forecasting

#### Fixed Income Detection
- **Recurring Income (I_t)**: Simple projection based on detected patterns
- **Detection Method**: Clustering by amount and date (within tolerance)
- **Output**: Exact day and amount of recurring income (salary, rent, dividends)
- **Use Case**: Predicts when money arrives, avoiding liquidity issues

#### Variable Expenses Forecasting (E_V,t)
- **Methods Available**:
  - **Naive**: Mean monthly expense per category
  - **Prophet**: Seasonality-aware (handles weekly/monthly/yearly patterns)
  - **ARIMA**: Auto-regressive integrated moving average
  - **Ensemble**: Weighted combination of all three
- **Category-Level**: Each expense category (groceries, utilities, discretionary) forecasted separately
- **Seasonality Handling**: 
  - Higher heating bills in winter
  - Holiday spending spikes
  - Back-to-school expenses
- **Uncertainty Quantification**: Mean and standard deviation per category per month

#### Transaction Timing Prediction
- **Classification Model**: Logistic regression (future: hazard models)
- **Purpose**: Predicts *when* major bills hit (credit card, variable loan payments)
- **Features**: Day of month, category, amount, historical patterns
- **Output**: Probability distribution over days of month
- **Use Case**: Avoids liquidity crunches by predicting bill timing

#### Inflation Adjustment (J_t)
- **External Data Feed**: Can integrate inflation APIs (BLS, country-specific)
- **User Override**: Manual inflation rate input
- **Application**: Applied to all future variable expenses
- **Formula**: `E_V,t × (1 + J_t)` where J_t is monthly inflation rate

#### The Refined Monthly Cash Flow Equation
For any given month (t), the predicted liquid cash flow (CF_t) is:

```
CF_t = (I_t + Other Inflows) - (E_F,t + E_V,t × (1 + J_t))
```

Where:
- `I_t`: Fixed income (from recurring detection)
- `E_F,t`: Fixed expenses (rent, subscriptions - constant)
- `E_V,t`: Variable expenses (from ML forecasting)
- `J_t`: Inflation adjustment

The system uses ML to calculate precise values of `I_t` and `E_V,t` for each month.

### Layer 2: Core Prediction Engine (Stochastic Modeling)

#### Monte Carlo Simulation Overview
The financial twin runs thousands (default 10,000) of complete simulations over the desired planning horizon (default 12 months, configurable up to 20+ years). In each simulation run, uncertain external variables are randomly selected from defined probability distributions.

#### Uncertain Variables and Their Distributions

| Variable | Modeling Approach | Distribution Parameters | Example |
|----------|------------------|------------------------|---------|
| **Investment Returns (R_t)** | Geometric Brownian Motion | Mean: 7.0%, Volatility: 15.0% (Annual) | Stock market returns |
| **Inflation Rate (J_t)** | Normal or Lognormal | Mean: 3.0%, Std Dev: 1.5% (Annual) | CPI inflation |
| **Unexpected Expenses (U_t)** | Poisson Distribution | Lambda: 0.1, Mean: $200 | Car repair, medical bill |

#### The Dynamic Wealth Equation
The system projects the value of your entire investment portfolio (A_t) forward:

```
A_t = A_{t-1} × (1 + R_t) + S_t
```

Where:
- `A_t`: Total wealth at month t
- `A_{t-1}`: Previous month's wealth
- `R_t`: Random return sampled from distribution
- `S_t`: Savings from cash flow (CF_t from Layer 1, invested)

#### Output: Percentiles of Success

After running thousands of simulations, the system generates key metrics:

- **P50 (Median)**: The outcome that 50% of scenarios surpassed. This is the **most likely** result.
- **P10 (Worst-Case)**: The outcome that only 10% of scenarios were worse than. This is the **stress-test result** you need to plan around.
- **P90 (Best-Case)**: The outcome that 90% of scenarios were worse than. Shows optimistic potential.
- **Probability of Goal Success**: Percentage of runs that successfully meet a long-term goal (e.g., having $500,000 saved by age 50).
- **Liquidity Breach Probability**: Percentage of runs where wealth drops below a safety floor.

### Layer 3: Scenario Generator (Testing the Change)

#### How Scenarios Work
To test a user's scenario, the system modifies variables in Layer 1, then re-runs the Monte Carlo simulations in Layer 2.

#### Scenario 1: Stopping the Netflix Payment
- **Input**: Netflix cost ($19.99) removed from `E_F,t` (Fixed Expenses) for all future t
- **Resulting Action**: Cash flow `CF_t` increases by $19.99. This amount is automatically re-allocated to Savings (`S_t`)
- **Twin's Output**: Compares P50 and P10 outcomes of "New Scenario" to "Default Model"
- **Example Output**: "Stopping Netflix increases your P50 wealth projection in 10 years by **$3,200** (a 4.5% improvement in goal success probability)."

#### Scenario 2: Increasing Income by $700 USD/month
- **Input**: Income `I_t` increases by $700 for all future t
- **Crucial Step: Behavioral Modeling**: The system assumes a split based on historical behavior or user definition (e.g., save 80% and spend 20% of extra income)
  - **Savings Allocation**: $560 goes to `S_t`
  - **Lifestyle Creep**: $140 goes to `E_D,t` (Discretionary Expenses, increasing ML-predicted spending limit)
- **Twin's Output**: 
  - **Example Output**: "If you save 80% of the raise, your P10 (worst-case) liquidity buffer never drops below **$25,000** over the next 5 years, compared to dipping to **$18,000** in the Default Model."

#### Scenario Comparison Metrics
- **ΔP50**: Change in median final wealth
- **ΔP10**: Change in worst-case final wealth
- **ΔGoal Success**: Change in probability of reaching goal
- **ΔLiquidity Breach**: Change in probability of dropping below floor
- **Min Liquidity Improvement**: Worst-case buffer increase

---

## AI Suggestions Layer (LLM Integration)

### Architecture
The AI suggestions layer sits on top of the 3-layer model, consuming its outputs to generate personalized, actionable advice.

### Input Context
The LLM receives structured context including:

1. **Simulation Results**:
   - Net monthly cash flow
   - Final wealth percentiles (P10, P50, P90)
   - Goal gap (difference between goal and median final wealth)
   - Goal success probability
   - Liquidity breach probability
   - Horizon (months simulated)

2. **Simulation Assumptions**:
   - Return mean and volatility
   - Shock frequency (lambda) and mean cost
   - Inflation rate
   - Starting wealth

3. **User Profile** (from memory):
   - Savings rate (historical)
   - Lifestyle creep percentage
   - Occupation/field
   - Estimated income and expense
   - Category-level expense breakdown (if provided)

4. **Current Context**:
   - Current savings
   - Goal target
   - Liquidity floor

### Prompt Engineering
The prompt is carefully structured to:
- Emphasize math-informed suggestions (references goal gap, net monthly)
- Include occupation-specific advice (programmer → tech skills, psychologist → telehealth)
- Consider expense breakdown (suggests cutting specific categories)
- Provide actionable items (not vague advice)
- Limit output to 3-5 concise bullet points

### Model Options

#### Hugging Face Inference (Default)
- **Endpoint**: Hugging Face Inference API
- **Default Model**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Configuration**: Via `HF_TOKEN` and `HF_MODEL` environment variables
- **Pros**: No local compute needed, easy to switch models
- **Cons**: Requires internet, API costs, potential latency

#### Self-Hosted TGI
- **Endpoint**: Text Generation Inference server (local or hosted)
- **Configuration**: Via `LOCAL_LLM_ENDPOINT` environment variable
- **Pros**: Full control, no API costs, privacy
- **Cons**: Requires GPU/server, maintenance overhead

#### Custom Adapters (LoRA)
- **Fine-tuning**: Train LoRA adapters on de-identified interaction data
- **Platforms**: Kaggle (free GPU), Colab, RunPod, Modal
- **Process**:
  1. Collect prompt/response pairs from user interactions
  2. Fine-tune small LoRA adapter (8-bit quantization)
  3. Push adapter to Hugging Face Hub
  4. Load adapter on serving endpoint
- **Benefits**: Model learns from actual user feedback, improves over time

### Response Format
The LLM is instructed to return:
- 3-5 bullet points
- Each bullet is actionable (specific action, not vague advice)
- Context-aware (uses occupation, expense breakdown)
- Math-informed (references goal gap, net monthly, percentiles)

### Fallback Behavior
- **No Token**: Skip suggestions, show notice
- **API Failure**: Skip suggestions, log error
- **Invalid Response**: Skip suggestions, log warning
- **Timeout**: Skip suggestions after 30 seconds

---

## CLI Command Reference

### `ftwin forecast`
Generate baseline cashflow forecast from ledger.

**Options**:
- `--ledger`: Path to CSV ledger (required)
- `--months`: Forecast horizon (default: 12)
- `--annual-inflation`: Inflation rate (default: 0.02)
- `--engine`: Forecaster engine: `naive` | `prophet` (default: `naive`)
- `--ensemble`: Use ensemble forecaster (default: false)
- `--out`: Output JSON path (optional, prints to stdout if omitted)

**Example**:
```bash
ftwin forecast --ledger data/transactions.csv --months 24 --ensemble --out baseline.json
```

### `ftwin simulate`
Run Monte Carlo simulation on baseline forecast.

**Options**:
- `--baseline`: Baseline forecast JSON (or use `--ledger` to generate on-the-fly)
- `--ledger`: CSV ledger (used if baseline not provided)
- `--paths`: Number of Monte Carlo paths (default: 5000)
- `--return-mean`: Annual return mean (default: 0.07)
- `--return-vol`: Annual return volatility (default: 0.15)
- `--start-wealth`: Starting wealth (default: 0.0)
- `--goal-target`: Goal target wealth (optional)
- `--liquidity-floor`: Liquidity floor (optional)
- `--out`: Output JSON path

**Example**:
```bash
ftwin simulate --baseline baseline.json --paths 10000 --goal-target 50000 --out sim.json
```

### `ftwin scenario`
Apply scenario deltas and run simulation.

**Options**:
- `--delta`: Scenario delta JSON (required)
- `--baseline`: Baseline forecast JSON (or use `--ledger`)
- `--ledger`: CSV ledger (used if baseline not provided)
- All simulation options from `ftwin simulate`
- `--out`: Output JSON path

**Example**:
```bash
ftwin scenario --delta netflix_cut.json --baseline baseline.json --out scen.json
```

### `ftwin interactive`
Interactive guided mode with memory and AI suggestions.

**Flow**:
1. Check for existing memory (`~/.ftwin_memory.json`)
2. If missing, prompt for quick profile (history length, income/spend estimates, savings rate, lifestyle creep)
3. Save profile to memory
4. Prompt for current financial inputs (income, spending, country, months, savings, goal, etc.)
5. Optionally prompt for expense breakdown (rent, utilities, groceries, etc.)
6. Run simulation with spinner
7. Display results with colored output
8. If `HF_TOKEN` set, generate AI suggestions
9. Save interactive state (country, months, preset, shock level)

**Example**:
```bash
export HF_TOKEN=hf_...
ftwin interactive
```

### `ftwin train`
Train per-user memory from ledger.

**Options**:
- `--ledger`: CSV ledger (required)

**Output**:
- Saves to `~/.ftwin_memory.json`
- Computes: savings rate, lifestyle creep, category stats, estimates

**Example**:
```bash
ftwin train --ledger historical_transactions.csv
```

### `ftwin memory`
Display stored per-user memory.

**Output**:
- Pretty-prints `~/.ftwin_memory.json` contents
- Shows savings rate, lifestyle creep, category stats, estimates, last trained timestamp

**Example**:
```bash
ftwin memory
```

---

## Data Formats & Schemas

### Ledger CSV Format
Required columns: `date`, `amount`, `category`, `kind`

**Example**:
```csv
date,amount,category,kind
2024-01-15,3500.00,salary,income
2024-01-20,-1200.00,rent,expense
2024-01-22,-400.00,groceries,expense
2024-01-25,-19.99,netflix,expense
```

**Constraints**:
- `date`: ISO format (YYYY-MM-DD) or parseable date string
- `amount`: Numeric (positive for income, negative for expense, or use `kind` field)
- `category`: String (consistent naming recommended)
- `kind`: Exactly `"income"` or `"expense"`

### Scenario Delta JSON Format
```json
{
  "income_deltas": {
    "salary": 500,
    "bonus": 1000
  },
  "expense_deltas": {
    "netflix": -19.99,
    "coffee": -60,
    "rent": 100
  },
  "savings_rate_override": 0.8
}
```

**Fields**:
- `income_deltas`: Object mapping category names to monthly income changes (additive)
- `expense_deltas`: Object mapping category names to monthly expense changes (additive, negative = reduction)
- `savings_rate_override`: Optional float (0-1) overriding default savings split

### Forecast Output JSON Format
```json
{
  "config": {
    "months": 12,
    "annual_inflation": 0.02,
    "seed": null
  },
  "monthly": [
    {
      "month": "2024-01-01",
      "income": 3500.0,
      "expense": 2800.0
    }
  ]
}
```

### Simulation Output JSON Format
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
    "p10": [1000, 2000, 3000],
    "p50": [1500, 3000, 4500],
    "p90": [2000, 4000, 6000]
  },
  "goal_success_prob": 0.75,
  "liquidity_breach_prob": 0.05
}
```

### Memory File JSON Format (`~/.ftwin_memory.json`)
```json
{
  "savings_rate": 0.15,
  "lifestyle_creep": 0.2,
  "category_stats": {
    "groceries": {"mean": 400, "std": 50},
    "rent": {"mean": 1200, "std": 0}
  },
  "estimated_income": 3500,
  "estimated_expense": 2800,
  "last_trained": "2024-01-15T10:30:00Z"
}
```

---

## Backend Architecture (Django + Celery)

### Django Application Structure
```
server/
  manage.py
  server/
    settings.py
    urls.py
    celery.py
    wsgi.py
  financial_twin/
    models.py          # SimulationRun model
    views.py           # DRF viewsets
    serializers.py     # DRF serializers
    tasks.py           # Celery tasks
    urls.py            # API routes
```

### API Endpoints

#### POST `/api/simulations/`
Creates async simulation job.

**Request** (multipart/form-data):
- `ledger`: CSV file
- `months`: int (default: 12)
- `engine`: str (default: "naive")
- `ensemble`: bool (default: false)
- `paths`: int (default: 5000)
- `return_mean`: float (default: 0.07)
- `return_vol`: float (default: 0.15)
- `start_wealth`: float (default: 0.0)
- `goal_target`: float (optional)
- `liquidity_floor`: float (optional)

**Response**:
```json
{
  "run_id": "uuid-here",
  "status": "pending"
}
```

#### POST `/api/scenarios/`
Creates async scenario simulation job.

**Request** (multipart/form-data):
- `ledger`: CSV file
- `delta`: JSON file (scenario deltas)
- Same simulation params as above

**Response**:
```json
{
  "run_id": "uuid-here",
  "status": "pending"
}
```

#### GET `/api/runs/<id>/`
Retrieves simulation run status and results.

**Response** (pending):
```json
{
  "status": "pending",
  "result": null
}
```

**Response** (completed):
```json
{
  "status": "completed",
  "result": {
    "percentiles": {...},
    "goal_success_prob": 0.75,
    "liquidity_breach_prob": 0.05
  },
  "result_path": "/media/runs/uuid.json"
}
```

**Response** (failed):
```json
{
  "status": "failed",
  "error": "Error message here"
}
```

### Celery Task Flow

1. **User Request**: POST to `/api/simulations/` or `/api/scenarios/`
2. **Django View**: Creates `SimulationRun` record with `status="pending"`
3. **Task Enqueue**: Calls `run_simulation.delay(run_id, ...)`
4. **Celery Worker**: Picks up task from Redis queue
5. **Processing**:
   - Loads ledger from uploaded file
   - Runs forecast (naive/prophet/ensemble)
   - If scenario, applies deltas
   - Runs Monte Carlo simulation
   - Saves result JSON to `MEDIA_ROOT/runs/<run_id>.json`
6. **Update Status**: Sets `SimulationRun.status = "completed"` and `result_path`
7. **User Polling**: User polls `/api/runs/<id>/` until status changes

### Redis Configuration
- **Broker**: Redis for Celery message queue
- **Result Backend**: Redis for task results
- **URL**: `redis://localhost:6379/0` (default)
- **Docker**: Provided via `redis` service in docker-compose

---

## Learning & Adaptation System

### Per-User Memory
- **Storage**: Local file `~/.ftwin_memory.json` (gitignored)
- **Contents**: Derived statistics, no raw PII
- **Privacy**: No transaction data, only aggregates
- **Usage**: Pre-fills defaults, informs AI suggestions, guides scenario splits

### Model Adaptation (Optional)

#### Data Collection
- **Interaction Logs**: Store prompt/response pairs locally
- **Feedback**: Collect user feedback (thumbs up/down, "applied" flags)
- **De-identification**: Remove PII before training

#### Fine-Tuning Process
1. **Prepare Dataset**: Convert interaction logs to JSONL format
2. **Choose Platform**: Kaggle (free GPU), Colab, RunPod, Modal
3. **Fine-Tune LoRA**: Train small adapter (8-bit quantization, rank 8-16)
4. **Evaluate**: Test on held-out data
5. **Deploy**: Push adapter to Hugging Face Hub, load on serving endpoint

#### Kaggle Notebook Helper
The `scripts/generate_kaggle_lora_notebook.py` script creates a ready-to-use notebook:
- Requests GPU accelerator
- Includes all dependencies
- Implements LoRA fine-tuning
- Handles model loading/saving
- Can push to Hugging Face Hub

**Usage**:
```bash
python scripts/generate_kaggle_lora_notebook.py
# Upload kaggle_lora_finetune.ipynb to Kaggle
# Select GPU in sidebar
# Run cells
```

---

## Privacy & Security

### Data Handling
- **Local Storage**: All user data stored locally (memory file, ledgers)
- **No Cloud Sync**: Memory file not synced to cloud by default
- **Gitignored**: Memory file, ledgers, feedback logs excluded from git
- **Encryption**: Optional encryption of memory file (future feature)

### PII Minimization
- **No Raw Data**: Memory stores only derived statistics
- **No Identifiers**: No names, account numbers, addresses
- **Category Names**: User-defined, can be anonymized
- **AI Prompts**: Only include aggregated stats, no transaction details

### API Security
- **Authentication**: Django REST Framework authentication (JWT/session)
- **File Uploads**: Validated file types, size limits
- **Rate Limiting**: Optional rate limiting on endpoints
- **CORS**: Configured for allowed origins

---

## Performance & Scalability

### Forecasting Performance
- **Naive**: <1 second for typical ledger (1000 transactions)
- **Prophet**: 5-30 seconds (depends on data size, seasonality complexity)
- **Ensemble**: 10-60 seconds (runs all three models)

### Simulation Performance
- **2000 paths**: ~1-2 seconds
- **5000 paths**: ~3-5 seconds
- **10000 paths**: ~6-10 seconds
- **Scales linearly**: 2x paths ≈ 2x time

### Memory Usage
- **Forecasting**: <100MB for typical ledger
- **Simulation**: ~50-200MB (depends on path count)
- **Ensemble**: ~200-500MB (multiple models loaded simultaneously)

### Optimization Strategies
1. **Caching**: Cache forecasts if running multiple scenarios
2. **Async Processing**: Use Celery for long-running jobs
3. **Path Reduction**: Use 2000 paths for interactive, 5000+ for final results
4. **Model Selection**: Use naive for quick iterations, ensemble for accuracy

---

## Future Enhancements

### Planned Features
1. **Per-Category Timing Classifier**: Predict bill timing using logistic regression
2. **Behavior Model**: Learn savings split and lifestyle creep from historical deltas
3. **RL-Based Goal Optimization**: Find optimal savings rate/risk level to hit goals
4. **Local LLM Adapters**: Fine-tune LoRA adapters on user feedback
5. **Enhanced Scenario Comparisons**: ΔP10/P50/P90, min liquidity improvements
6. **Web UI Dashboard**: Visual interface for forecasts, simulations, scenarios
7. **Multi-Goal Support**: Coordinate multiple financial goals simultaneously
8. **Portfolio Allocation**: Map user's asset allocation to return/vol assumptions

### Research Directions
- **Federated Learning**: Aggregate learning across users without centralizing data
- **Explainable AI**: Show why suggestions were made (feature importance)
- **Real-Time Updates**: Auto-retrain when new transactions arrive
- **Life Event Detection**: Detect salary changes, big purchases, trigger retraining

---

*Last updated: 2024-12-10*

