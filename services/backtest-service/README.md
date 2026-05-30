# Backtest Service

Small internal FastAPI service for historical strategy simulation. It reads
public OHLCV data, runs local backtesting.py simulations, and returns summary
statistics, sampled equity, and trade rows.

Default endpoints:

```text
GET /healthz
GET /strategies
POST /backtest
```

Example:

```text
POST /backtest
{
  "symbol": "BTCUSDT",
  "source": "binance",
  "timeframe": "1d",
  "start": "2023-01-01",
  "end": "2024-01-01",
  "strategy": "ma_cross",
  "params": {"fast": 20, "slow": 50, "trend": 200},
  "cash": 10000,
  "commission": 0.001
}
```

The service is isolated from the execution path and only returns simulation
results.
