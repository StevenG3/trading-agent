from __future__ import annotations

import importlib
import os
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal, cast

import pandas as pd  # type: ignore[import-untyped]

Source = Literal["binance", "okx", "bybit", "yfinance"]

MAX_BARS = int(os.getenv("BACKTEST_MAX_BARS", "5000"))
REQUEST_LIMIT = 1000
RETRY_COUNT = 2
TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
}


class DataLoadError(RuntimeError):
    pass


def _parse_date(value: date | datetime | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time())
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_crypto_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper().replace("-", "/")
    if "/" in normalized:
        return normalized
    for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
        if normalized.endswith(quote) and len(normalized) > len(quote):
            return f"{normalized[:-len(quote)]}/{quote}"
    return normalized


def _to_backtesting_frame(rows: list[list[object]]) -> pd.DataFrame:
    if not rows:
        raise DataLoadError("no historical bars returned")
    frame = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    frame["ts"] = pd.to_datetime(frame["ts"], unit="ms", utc=True)
    frame = frame.set_index("ts").sort_index()
    frame.index = frame.index.tz_convert(None)
    frame = frame[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna()
    if len(frame) < 20:
        raise DataLoadError("not enough bars for backtest")
    return cast(pd.DataFrame, frame)


def _int_value(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        return int(value)
    raise DataLoadError("invalid timestamp in historical data")


def _request_with_retry(func: Any, *args: object, **kwargs: object) -> Any:
    last_error: Exception | None = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < RETRY_COUNT:
                time.sleep(0.25 * (attempt + 1))
    raise DataLoadError(str(last_error) if last_error else "historical data request failed")


def _load_crypto_ohlcv(
    symbol: str,
    source: Literal["binance", "okx", "bybit"],
    timeframe: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    if timeframe not in TIMEFRAME_MS:
        raise DataLoadError(f"unsupported timeframe: {timeframe}")
    ccxt = importlib.import_module("ccxt")
    factory = getattr(ccxt, source)
    exchange = factory({"enableRateLimit": True, "timeout": 10_000})
    market_symbol = _format_crypto_symbol(symbol)
    since = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    step_ms = TIMEFRAME_MS[timeframe]
    rows: list[list[object]] = []
    seen: set[int] = set()

    while since < end_ms and len(rows) < MAX_BARS:
        batch = _request_with_retry(
            exchange.fetch_ohlcv,
            market_symbol,
            timeframe,
            since=since,
            limit=min(REQUEST_LIMIT, MAX_BARS - len(rows)),
        )
        if not isinstance(batch, list) or not batch:
            break
        advanced = False
        for raw in batch:
            if not isinstance(raw, list) or len(raw) < 6:
                continue
            ts = int(raw[0])
            if ts >= end_ms:
                continue
            if ts in seen:
                continue
            seen.add(ts)
            rows.append(raw[:6])
            advanced = True
        last_ts = int(batch[-1][0]) if isinstance(batch[-1], list) and batch[-1] else since
        next_since = last_ts + step_ms
        if not advanced or next_since <= since:
            break
        since = next_since

    rows = [row for row in rows if _int_value(row[0]) >= int(start_dt.timestamp() * 1000)]
    return _to_backtesting_frame(rows[:MAX_BARS])


def _load_yfinance_ohlcv(
    symbol: str,
    timeframe: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    yf = importlib.import_module("yfinance")
    interval = timeframe
    end_for_download = end_dt + timedelta(days=1) if timeframe == "1d" else end_dt
    data = _request_with_retry(
        yf.download,
        symbol,
        start=start_dt.date().isoformat(),
        end=end_for_download.date().isoformat(),
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise DataLoadError("no historical bars returned")
    frame = data.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(col[0]) for col in frame.columns]
    rename = {
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
    }
    frame = frame.rename(columns=rename)
    missing = [column for column in rename.values() if column not in frame.columns]
    if missing:
        raise DataLoadError(f"missing yfinance columns: {', '.join(missing)}")
    frame = frame[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(None)
    frame = frame.sort_index().dropna()
    frame = frame[
        (frame.index >= start_dt.replace(tzinfo=None))
        & (frame.index <= end_dt.replace(tzinfo=None))
    ]
    if len(frame) > MAX_BARS:
        frame = frame.iloc[-MAX_BARS:]
    if len(frame) < 20:
        raise DataLoadError("not enough bars for backtest")
    return cast(pd.DataFrame, frame)


def load_ohlcv(
    symbol: str,
    source: Source,
    timeframe: str,
    start: date | datetime | str,
    end: date | datetime | str,
) -> pd.DataFrame:
    start_dt = _parse_date(start)
    end_dt = _parse_date(end)
    if end_dt <= start_dt:
        raise DataLoadError("end must be after start")
    if source == "binance":
        return _load_crypto_ohlcv(symbol, source, timeframe, start_dt, end_dt)
    if source == "okx":
        return _load_crypto_ohlcv(symbol, source, timeframe, start_dt, end_dt)
    if source == "bybit":
        return _load_crypto_ohlcv(symbol, source, timeframe, start_dt, end_dt)
    if source == "yfinance":
        return _load_yfinance_ohlcv(symbol, timeframe, start_dt, end_dt)
    raise DataLoadError(f"unsupported source: {source}")
