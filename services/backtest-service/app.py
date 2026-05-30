from __future__ import annotations

from datetime import date, datetime
from typing import Any, cast

from backtesting import Backtest, Strategy  # type: ignore[import-untyped]
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

import data as data_module
from data import DataLoadError, Source
from strategies import DEFAULT_PARAMS, STRATEGIES

app = FastAPI(title="backtest-service", version="0.1.0")


class BacktestRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=40)
    source: Source
    timeframe: str = "1d"
    start: date
    end: date
    strategy: str = "ma_cross"
    params: dict[str, int] = Field(default_factory=dict)
    cash: float = Field(default=10_000, gt=0)
    commission: float = Field(default=0.001, ge=0, le=0.2)

    @field_validator("params")
    @classmethod
    def validate_params(cls, value: dict[str, int]) -> dict[str, int]:
        allowed = {"fast", "slow", "trend"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unsupported params: {', '.join(sorted(unknown))}")
        for key, raw in value.items():
            if raw <= 0:
                raise ValueError(f"{key} must be positive")
        return value


class BacktestStats(BaseModel):
    return_pct: float
    buy_hold_return_pct: float
    max_drawdown_pct: float
    win_rate: float
    sharpe: float
    num_trades: int
    exposure_pct: float


class EquityPoint(BaseModel):
    date: str
    equity: float


class TradeItem(BaseModel):
    entry_time: str
    exit_time: str
    pnl_pct: float
    size: float


class BacktestResponse(BaseModel):
    stats: BacktestStats
    equity_curve: list[EquityPoint]
    trades: list[TradeItem]


class StrategyInfo(BaseModel):
    name: str
    default_params: dict[str, int]


def _float_stat(stats: Any, key: str) -> float:
    value = stats.get(key, 0) if hasattr(stats, "get") else 0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_stat(stats: Any, key: str) -> int:
    value = stats.get(key, 0) if hasattr(stats, "get") else 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _configured_strategy(base_cls: type[Strategy], params: dict[str, int]) -> type[Strategy]:
    attrs: dict[str, Any] = {}
    for key in ("fast", "slow", "trend"):
        if key in params:
            attrs[key] = params[key]
    return cast(type[Strategy], type("ConfiguredStrategy", (base_cls,), attrs))


def _date_str(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _safe_float(value: object) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    return 0.0


def _sample_equity_curve(stats: Any) -> list[EquityPoint]:
    curve = stats.get("_equity_curve") if hasattr(stats, "get") else None
    if curve is None or not hasattr(curve, "iterrows"):
        return []
    length = len(curve)
    if length <= 250:
        step = 1
    else:
        step = max(length // 250, 1)
    points: list[EquityPoint] = []
    for index, (row_index, row) in enumerate(curve.iterrows()):
        if index % step != 0 and index != length - 1:
            continue
        equity = getattr(row, "Equity", None)
        if equity is None and hasattr(row, "get"):
            equity = row.get("Equity")
        points.append(EquityPoint(date=_date_str(row_index), equity=_safe_float(equity)))
    return points


def _trade_value(row: Any, *names: str) -> object:
    for name in names:
        if hasattr(row, "get"):
            value = row.get(name)
            if value is not None:
                return value
        if hasattr(row, name):
            return getattr(row, name)
    return None


def _trades(stats: Any) -> list[TradeItem]:
    frame = stats.get("_trades") if hasattr(stats, "get") else None
    if frame is None or not hasattr(frame, "iterrows"):
        return []
    trades: list[TradeItem] = []
    for _, row in frame.iterrows():
        entry = _trade_value(row, "EntryTime")
        exit_time = _trade_value(row, "ExitTime")
        pnl_pct_raw = _trade_value(row, "ReturnPct")
        size_raw = _trade_value(row, "Size")
        pnl_pct = _safe_float(pnl_pct_raw) * 100
        size = _safe_float(size_raw)
        trades.append(
            TradeItem(
                entry_time=_date_str(entry),
                exit_time=_date_str(exit_time),
                pnl_pct=pnl_pct,
                size=size,
            )
        )
    return trades


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/strategies", response_model=list[StrategyInfo])
def strategies() -> list[StrategyInfo]:
    return [
        StrategyInfo(name=name, default_params=DEFAULT_PARAMS[name])
        for name in sorted(STRATEGIES)
    ]


@app.post("/backtest", response_model=BacktestResponse)
def run_backtest(request: BacktestRequest) -> BacktestResponse:
    strategy_cls = STRATEGIES.get(request.strategy)
    if strategy_cls is None:
        raise HTTPException(status_code=400, detail={"code": "UNKNOWN_STRATEGY"})

    params = {**DEFAULT_PARAMS.get(request.strategy, {}), **request.params}
    if params["fast"] >= params["slow"]:
        raise HTTPException(status_code=400, detail={"code": "FAST_MUST_BE_LT_SLOW"})
    if params["slow"] >= params["trend"]:
        raise HTTPException(status_code=400, detail={"code": "SLOW_MUST_BE_LT_TREND"})

    try:
        frame = data_module.load_ohlcv(
            request.symbol,
            request.source,
            request.timeframe,
            request.start,
            request.end,
        )
    except DataLoadError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "DATA_UNAVAILABLE", "message": str(exc)},
        ) from exc

    if len(frame) <= params["trend"]:
        raise HTTPException(status_code=422, detail={"code": "INSUFFICIENT_BARS"})

    configured = _configured_strategy(strategy_cls, params)
    try:
        backtest = Backtest(
            frame,
            configured,
            cash=request.cash,
            commission=request.commission,
            exclusive_orders=True,
            finalize_trades=True,
        )
        stats = backtest.run()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422,
            detail={"code": "BACKTEST_FAILED", "message": str(exc)},
        ) from exc

    response_stats = BacktestStats(
        return_pct=_float_stat(stats, "Return [%]"),
        buy_hold_return_pct=_float_stat(stats, "Buy & Hold Return [%]"),
        max_drawdown_pct=_float_stat(stats, "Max. Drawdown [%]"),
        win_rate=_float_stat(stats, "Win Rate [%]"),
        sharpe=_float_stat(stats, "Sharpe Ratio"),
        num_trades=_int_stat(stats, "# Trades"),
        exposure_pct=_float_stat(stats, "Exposure Time [%]"),
    )
    return BacktestResponse(
        stats=response_stats,
        equity_curve=_sample_equity_curve(stats),
        trades=_trades(stats),
    )
