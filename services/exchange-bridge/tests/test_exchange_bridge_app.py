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


class FakeHttpResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class FakeHttp:
    def request(
        self,
        method: str,
        url: str,
        **kwargs: object,
    ) -> FakeHttpResponse:
        _ = method, kwargs
        if "/api/v3/ticker/price" in url:
            return FakeHttpResponse(
                [
                    {"symbol": "BTCUSDT", "price": "100000"},
                    {"symbol": "ETHUSDT", "price": "2000"},
                    {"symbol": "DOGEUSDT", "price": "0.1"},
                    {"symbol": "SOLUSDT", "price": "100"},
                ]
            )
        if "/api/v3/account" in url:
            return FakeHttpResponse(
                {
                    "balances": [
                        {"asset": "BTC", "free": "1.5", "locked": "0.5"},
                        {"asset": "ETH", "free": "0", "locked": "0"},
                        {"asset": "USDT", "free": "25", "locked": "0"},
                        {"asset": "LDUSDT", "free": "20", "locked": "0"},
                        {"asset": "DOGE", "free": "1", "locked": "0"},
                    ]
                }
            )
        if "/sapi/v1/asset/get-funding-asset" in url:
            return FakeHttpResponse([{"asset": "USDT", "free": "10", "locked": "2"}])
        if "/fapi/v2/balance" in url:
            return FakeHttpResponse([{"asset": "USDT", "balance": "7", "crossUnPnl": "1"}])
        if "/simple-earn/flexible/position" in url:
            return FakeHttpResponse(
                {
                    "rows": [
                        {"asset": "ETH", "totalAmount": "0.02"},
                        {"asset": "USDT", "totalAmount": "20"},
                    ]
                }
            )
        if "/simple-earn/locked/position" in url:
            return FakeHttpResponse({"rows": []})
        if "/dapi/v1/balance" in url:
            return FakeHttpResponse([])
        if "/dualInvestment/positions" in url:
            return FakeHttpResponse({"list": []})
        if "/asset/wallet/balance" in url:
            return FakeHttpResponse([])
        if "/api/v5/account/balance" in url:
            return FakeHttpResponse(
                {
                    "code": "0",
                    "data": [
                        {
                            "details": [
                                {
                                    "ccy": "USDT",
                                    "eq": "3",
                                    "frozenBal": "1",
                                    "eqUsd": "3",
                                },
                                {"ccy": "BTC", "eq": "0.001", "eqUsd": "100"},
                            ]
                        }
                    ],
                }
            )
        if "/api/v5/asset/balances" in url:
            return FakeHttpResponse(
                {"code": "0", "data": [{"ccy": "USDT", "availBal": "2", "frozenBal": "1"}]}
            )
        if "/api/v5/finance/savings/balance" in url:
            return FakeHttpResponse(
                {"code": "0", "data": [{"ccy": "ETH", "amt": "0.02", "eqUsd": "40"}]}
            )
        if "/api/v5/account/positions" in url:
            return FakeHttpResponse(
                {
                    "code": "0",
                    "data": [
                        {
                            "instType": "SWAP",
                            "ccy": "USDT",
                            "margin": "5",
                            "instId": "BTC-USDT-SWAP",
                        }
                    ],
                }
            )
        if "/api/v5/market/ticker" in url:
            return FakeHttpResponse({"code": "0", "data": [{"last": "2000"}]})
        raise AssertionError(f"unexpected request {url}")


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
    exchange_client_module = sys.modules["exchange_client"]
    exchange_client_module.httpx = FakeHttp()
    fake_ccxt.created = {"binance": [], "okx": [], "bybit": []}
    module.client = module.ExchangeClient.from_env()
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
    assert body["summaries"] == {
        "binance": {
            "total_usd": "200105.1",
            "visible_usd": "200105",
            "hidden_usd": "0.1",
            "hidden_count": "1",
            "total_assets": "4",
            "visible_assets": "3",
            "min_usd_detail": "10",
        },
        "okx": {
            "total_usd": "151",
            "visible_usd": "151",
            "hidden_usd": "0",
            "hidden_count": "0",
            "total_assets": "3",
            "visible_assets": "3",
            "min_usd_detail": "10",
        },
        "bybit": {
            "total_usd": "0",
            "visible_usd": "0",
            "hidden_usd": "0",
            "hidden_count": "0",
            "total_assets": "1",
            "visible_assets": "1",
            "min_usd_detail": "10",
        },
    }
    assert body["balances"] == [
        {
            "exchange": "binance",
            "asset": "BTC",
            "free": "1.5",
            "used": "0.5",
            "total": "2",
            "usd_value": "200000",
            "sources": "spot:2",
        },
        {
            "exchange": "binance",
            "asset": "USDT",
            "free": "63",
            "used": "2",
            "total": "65",
            "usd_value": "65",
            "sources": "earn-flex:20, funding:12, futures-usds:8, spot:25",
        },
        {
            "exchange": "binance",
            "asset": "ETH",
            "free": "0.02",
            "used": "0",
            "total": "0.02",
            "usd_value": "40",
            "sources": "earn-flex:0.02",
        },
        {
            "exchange": "okx",
            "asset": "BTC",
            "free": "0.001",
            "used": "0",
            "total": "0.001",
            "usd_value": "100",
            "sources": "trading:0.001",
        },
        {
            "exchange": "okx",
            "asset": "ETH",
            "free": "0.02",
            "used": "0",
            "total": "0.02",
            "usd_value": "40",
            "sources": "savings:0.02",
        },
        {
            "exchange": "okx",
            "asset": "USDT",
            "free": "9",
            "used": "2",
            "total": "11",
            "usd_value": "11",
            "sources": "funding:3, futures-margin:5, trading:3",
        },
        {
            "exchange": "bybit",
            "asset": "SOL",
            "free": "7",
            "used": "1",
            "total": "8",
            "usd_value": "0",
            "sources": "default:8",
        },
    ]


def test_balances_can_filter_to_single_exchange(monkeypatch) -> None:
    configure_all(monkeypatch)
    bridge_app, _fake_ccxt = load_service_app(monkeypatch)

    response = TestClient(bridge_app.app).get("/balances", params={"exchange": "bybit"})

    assert response.status_code == 200
    body = response.json()
    assert body["exchanges"] == {"binance": False, "okx": False, "bybit": True}
    assert body["summaries"] == {
        "bybit": {
            "total_usd": "0",
            "visible_usd": "0",
            "hidden_usd": "0",
            "hidden_count": "0",
            "total_assets": "1",
            "visible_assets": "1",
            "min_usd_detail": "10",
        }
    }
    assert body["balances"] == [
        {
            "exchange": "bybit",
            "asset": "SOL",
            "free": "7",
            "used": "1",
            "total": "8",
            "usd_value": "0",
            "sources": "default:8",
        }
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
