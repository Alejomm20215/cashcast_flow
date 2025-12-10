from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Dict, List, Optional, Sequence, Tuple


@dataclasses.dataclass(frozen=True)
class Money:
    amount: float
    currency: str = "USD"

    def add(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError("Currency mismatch")
        return Money(self.amount + other.amount, self.currency)


@dataclasses.dataclass(frozen=True)
class LedgerEntry:
    date: dt.date
    amount: float
    category: str
    kind: str  # "income" or "expense"


@dataclasses.dataclass(frozen=True)
class ForecastConfig:
    months: int = 12
    annual_inflation: float = 0.02
    seed: Optional[int] = None


@dataclasses.dataclass
class MonthlyForecast:
    month: dt.date
    income: float
    expense: float

    @property
    def net(self) -> float:
        return self.income - self.expense


@dataclasses.dataclass
class ForecastResult:
    config: ForecastConfig
    monthly: List[MonthlyForecast]

    def to_timeseries(self) -> List[Tuple[str, float]]:
        return [(m.month.isoformat(), m.net) for m in self.monthly]


@dataclasses.dataclass(frozen=True)
class SimulationConfig:
    horizon_months: int = 12
    paths: int = 5000
    return_mean: float = 0.07
    return_vol: float = 0.15
    inflation_mean: float = 0.03
    inflation_std: float = 0.01
    shock_lambda: float = 0.1
    shock_mean: float = 200.0
    seed: Optional[int] = None


@dataclasses.dataclass
class ScenarioDelta:
    income_deltas: Dict[str, float]
    expense_deltas: Dict[str, float]
    savings_rate_override: Optional[float] = None  # 0..1


@dataclasses.dataclass
class SimulationResult:
    config: SimulationConfig
    percentiles: Dict[str, List[float]]  # keys like "p10","p50","p90"
    goal_success_prob: Optional[float] = None

    def snapshot(self) -> Dict[str, List[float]]:
        return self.percentiles


@dataclasses.dataclass
class ScenarioComparison:
    baseline: SimulationResult
    scenario: SimulationResult
    delta: Dict[str, List[float]]

