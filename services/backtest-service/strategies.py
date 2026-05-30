from __future__ import annotations

from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from backtesting import Strategy  # type: ignore[import-untyped]
from backtesting.lib import crossover  # type: ignore[import-untyped]


def sma(values: Any, period: int) -> Any:
    return pd.Series(values).rolling(period).mean().to_numpy()


class MaCrossStrategy(Strategy):  # type: ignore[misc]
    fast = 20
    slow = 50
    trend = 200

    def init(self) -> None:
        self.fast_ma = self.I(sma, self.data.Close, self.fast)
        self.slow_ma = self.I(sma, self.data.Close, self.slow)
        self.trend_ma = self.I(sma, self.data.Close, self.trend)

    def next(self) -> None:
        price = self.data.Close[-1]
        if (
            not self.position
            and price > self.trend_ma[-1]
            and crossover(self.fast_ma, self.slow_ma)
        ):
            self.buy()
        elif self.position and (price < self.trend_ma[-1] or crossover(self.slow_ma, self.fast_ma)):
            self.position.close()


DEFAULT_PARAMS: dict[str, dict[str, int]] = {
    "ma_cross": {"fast": 20, "slow": 50, "trend": 200},
}

STRATEGIES: dict[str, type[Strategy]] = {
    "ma_cross": MaCrossStrategy,
}

# TODO: add a strategy adapter for TradingAgents scorecard signal streams.
