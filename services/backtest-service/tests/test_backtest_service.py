from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
from fastapi.testclient import TestClient


def load_service_app():
    service_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(service_dir))
    for name in ("app", "data", "strategies"):
        sys.modules.pop(name, None)
    path = service_dir / "app.py"
    spec = importlib.util.spec_from_file_location("backtest_service_app", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["backtest_service_app"] = module
    spec.loader.exec_module(module)
    return module


def sample_frame(rows: int = 120) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=rows, freq="D")
    close = []
    value = 100.0
    for item in range(rows):
        if item < 30:
            value -= 0.5
        elif item < 70:
            value += 1.2
        elif item < 90:
            value -= 1.5
        else:
            value += 0.8
        close.append(value)
    frame = pd.DataFrame(
        {
            "Open": [value - 0.5 for value in close],
            "High": [value + 1.0 for value in close],
            "Low": [value - 1.0 for value in close],
            "Close": close,
            "Volume": [1000.0 + item for item in range(rows)],
        },
        index=index,
    )
    return frame


def base_payload() -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "source": "binance",
        "timeframe": "1d",
        "start": "2023-01-01",
        "end": "2023-05-01",
        "strategy": "ma_cross",
        "params": {"fast": 5, "slow": 10, "trend": 20},
        "cash": 10000,
        "commission": 0.001,
    }


def test_strategies_lists_defaults() -> None:
    module = load_service_app()
    response = TestClient(module.app).get("/strategies")
    assert response.status_code == 200
    assert response.json() == [
        {"name": "ma_cross", "default_params": {"fast": 20, "slow": 50, "trend": 200}}
    ]


def test_backtest_returns_stats_equity_and_trades(monkeypatch) -> None:
    module = load_service_app()
    monkeypatch.setattr(module.data_module, "load_ohlcv", lambda *args: sample_frame())

    response = TestClient(module.app).post("/backtest", json=base_payload())

    assert response.status_code == 200
    body = response.json()
    assert set(body["stats"]) == {
        "return_pct",
        "buy_hold_return_pct",
        "max_drawdown_pct",
        "win_rate",
        "sharpe",
        "num_trades",
        "exposure_pct",
    }
    assert body["equity_curve"]
    assert set(body["equity_curve"][0]) == {"date", "equity"}
    assert isinstance(body["trades"], list)
    if body["trades"]:
        assert set(body["trades"][0]) == {"entry_time", "exit_time", "pnl_pct", "size"}


def test_unknown_strategy_returns_400(monkeypatch) -> None:
    module = load_service_app()
    monkeypatch.setattr(module.data_module, "load_ohlcv", lambda *args: sample_frame())
    payload = dict(base_payload(), strategy="missing")

    response = TestClient(module.app).post("/backtest", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "UNKNOWN_STRATEGY"


def test_insufficient_data_returns_friendly_error(monkeypatch) -> None:
    module = load_service_app()
    monkeypatch.setattr(module.data_module, "load_ohlcv", lambda *args: sample_frame(15))

    response = TestClient(module.app).post("/backtest", json=base_payload())

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INSUFFICIENT_BARS"


def test_source_contains_no_mutating_exchange_calls() -> None:
    service_dir = Path(__file__).resolve().parents[1]
    source = "\n".join(path.read_text() for path in service_dir.rglob("*.py"))
    blocked = [
        "_".join(("create", "order")),
        "_".join(("place", "order")),
        "can" + "cel",
        "with" + "draw",
    ]
    assert all(term not in source for term in blocked)
