from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient


class FakeExchange:
    def __init__(self, config: dict[str, object], balance: dict[str, object] | None = None) -> None:
        self.config = config
        self.balance = balance or {
            "free": {"BTC": "1.5", "ETH": "0", "USDT": "25"},
            "used": {"BTC": "0.5", "ETH": "0", "USDT": "0"},
            "total": {"BTC": "2", "ETH": "0", "USDT": "25"},
        }

    def fetch_balance(self) -> dict[str, object]:
        return self.balance


class FakeCcxt(types.SimpleNamespace):
    def __init__(self) -> None:
        super().__init__()
        self.created: dict[str, list[dict[str, object]]] = {
            "binance": [],
            "okx": [],
            "bybit": [],
        }

    def binance(self, config: dict[str, object]) -> FakeExchange:
        self.created["binance"].append(config)
        return FakeExchange(config)

    def okx(self, config: dict[str, object]) -> FakeExchange:
        self.created["okx"].append(config)
        return FakeExchange(
            config,
            {
                "free": {"USDT": "3", "ZERO": "0"},
                "used": {"USDT": "2", "ZERO": "0"},
                "total": {"USDT": "5", "ZERO": "0"},
            },
        )

    def bybit(self, config: dict[str, object]) -> FakeExchange:
        self.created["bybit"].append(config)
        return FakeExchange(
            config,
            {
                "free": {"SOL": "7"},
                "used": {"SOL": "1"},
                "total": {"SOL": "8"},
            },
        )


def load_service_app(monkeypatch):
    service_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(service_dir))
    sys.modules.pop("app", None)
    sys.modules.pop("exchange_client", None)
    fake_ccxt = FakeCcxt()
    monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt)
    path = service_dir / "app.py"
    spec = importlib.util.spec_from_file_location("exchange_bridge_app", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exchange_bridge_app"] = module
    spec.loader.exec_module(module)
    return module, fake_ccxt


def configure_all(monkeypatch) -> None:
    monkeypatch.setenv("EXCHANGE_API_KEY", "binance-key")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "binance-secret")
    monkeypatch.setenv("OKX_API_KEY", "okx-key")
    monkeypatch.setenv("OKX_API_SECRET", "okx-secret")
    monkeypatch.setenv("OKX_API_PASSPHRASE", "okx-pass")
    monkeypatch.setenv("BYBIT_API_KEY", "bybit-key")
    monkeypatch.setenv("BYBIT_API_SECRET", "bybit-secret")


def test_balances_aggregates_exchanges_and_filters_zero_totals(monkeypatch) -> None:
    configure_all(monkeypatch)
    bridge_app, _fake_ccxt = load_service_app(monkeypatch)

    response = TestClient(bridge_app.app).get("/balances")

    assert response.status_code == 200
    body = response.json()
    assert body["exchanges"] == {"binance": True, "okx": True, "bybit": True}
    assert body["errors"] == {}
    assert body["balances"] == [
        {"exchange": "binance", "asset": "BTC", "free": "1.5", "used": "0.5", "total": "2"},
        {"exchange": "binance", "asset": "USDT", "free": "25", "used": "0", "total": "25"},
        {"exchange": "okx", "asset": "USDT", "free": "3", "used": "2", "total": "5"},
        {"exchange": "bybit", "asset": "SOL", "free": "7", "used": "1", "total": "8"},
    ]


def test_balances_can_filter_to_single_exchange(monkeypatch) -> None:
    configure_all(monkeypatch)
    bridge_app, _fake_ccxt = load_service_app(monkeypatch)

    response = TestClient(bridge_app.app).get("/balances", params={"exchange": "bybit"})

    assert response.status_code == 200
    body = response.json()
    assert body["exchanges"] == {"binance": False, "okx": False, "bybit": True}
    assert body["balances"] == [
        {"exchange": "bybit", "asset": "SOL", "free": "7", "used": "1", "total": "8"}
    ]


def test_unconfigured_exchange_is_skipped_and_readiness_reflects_it(monkeypatch) -> None:
    monkeypatch.setenv("EXCHANGE_API_KEY", "binance-key")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "binance-secret")
    monkeypatch.delenv("OKX_API_KEY", raising=False)
    monkeypatch.delenv("OKX_API_SECRET", raising=False)
    monkeypatch.delenv("OKX_API_PASSPHRASE", raising=False)
    monkeypatch.delenv("BYBIT_API_KEY", raising=False)
    monkeypatch.delenv("BYBIT_API_SECRET", raising=False)
    bridge_app, fake_ccxt = load_service_app(monkeypatch)

    balances_response = TestClient(bridge_app.app).get("/balances")
    ready_response = TestClient(bridge_app.app).get("/readyz")

    assert balances_response.status_code == 200
    assert balances_response.json()["exchanges"] == {"binance": True, "okx": False, "bybit": False}
    assert ready_response.status_code == 200
    assert ready_response.json() == {
        "status": "ready",
        "exchanges": {"binance": True, "okx": False, "bybit": False},
    }
    assert len(fake_ccxt.created["binance"]) == 1
    assert fake_ccxt.created["okx"] == []
    assert fake_ccxt.created["bybit"] == []


def test_readyz_returns_503_when_no_exchange_is_configured(monkeypatch) -> None:
    for name in (
        "EXCHANGE_API_KEY",
        "EXCHANGE_API_SECRET",
        "OKX_API_KEY",
        "OKX_API_SECRET",
        "OKX_API_PASSPHRASE",
        "BYBIT_API_KEY",
        "BYBIT_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    bridge_app, _fake_ccxt = load_service_app(monkeypatch)

    response = TestClient(bridge_app.app).get("/readyz")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "exchanges": {"binance": False, "okx": False, "bybit": False},
    }


def test_source_contains_no_ccxt_mutation_calls() -> None:
    service_dir = Path(__file__).resolve().parents[1]
    source = "\n".join(path.read_text() for path in service_dir.rglob("*.py"))
    blocked = [
        "_".join(("create", "order")),
        "_".join(("create", "market")),
        "_".join(("create", "limit")),
        "can" + "cel",
        "with" + "draw",
        "trans" + "fer",
    ]
    assert all(term not in source for term in blocked)
