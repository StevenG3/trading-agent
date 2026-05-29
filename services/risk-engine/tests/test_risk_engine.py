import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient


def load_service_app(name: str):
    service_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(service_dir))
    path = service_dir / "app.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


risk_app = load_service_app("risk_engine_app")
app = risk_app.app
VALID = {
    "intent_id": "11111111-1111-4111-8111-111111111111",
    "request_id": "22222222-2222-4222-8222-222222222222",
    "idempotency_key": "demo-paper-1",
    "actor": "user_1",
    "created_at": "2026-05-25T00:00:00Z",
    "mode": "paper",
    "venue": "binance_spot",
    "symbol": "BTCUSDT",
    "side": "buy",
    "order_type": "market",
    "quantity": {"kind": "quote", "value": "100"},
    "limit_price": None,
    "time_in_force": "GTC",
    "reduce_only": False,
    "leverage": None,
    "stop_loss": None,
    "take_profit": None,
    "source": {"origin": "manual_api", "scorecard_id": None, "hermes_message_id": None},
    "client_confirmation_required": False,
}


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"price": "100000.00"}


class FakePnlResponse:
    def __init__(self, total_pnl: str = "0") -> None:
        self.total_pnl = total_pnl

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"total_pnl": self.total_pnl}


def make_httpx_get(total_pnl: str = "0", price: str = "100000.00"):
    def fake_get(url: str, **kwargs: object) -> FakeResponse | FakePnlResponse:
        if url.endswith("/pnl/today"):
            return FakePnlResponse(total_pnl)
        return FakeResponse()

    return fake_get


@pytest.fixture(autouse=True)
def default_pnl_lookup(monkeypatch) -> None:
    monkeypatch.setattr(risk_app.httpx, "get", make_httpx_get())


def test_small_order_is_approved_without_confirmation() -> None:
    response = TestClient(app).post("/validate", json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert body["approved"] is True
    assert body["requires_confirmation"] is False
    assert body["confirmation_token"] is None


def test_medium_order_requires_confirmation() -> None:
    response = TestClient(app).post(
        "/validate", json=dict(VALID, quantity={"kind": "quote", "value": "1000"})
    )
    assert response.status_code == 200
    body = response.json()
    assert body["approved"] is True
    assert body["requires_confirmation"] is True
    UUID(body["confirmation_token"])
    evaluated = datetime.fromisoformat(body["evaluated_at"].replace("Z", "+00:00"))
    expires = datetime.fromisoformat(body["confirmation_expires_at"].replace("Z", "+00:00"))
    assert 299 <= (expires - evaluated).total_seconds() <= 301


def test_large_order_is_rejected() -> None:
    response = TestClient(app).post(
        "/validate", json=dict(VALID, quantity={"kind": "quote", "value": "50000"})
    )
    assert response.status_code == 200
    body = response.json()
    assert body["approved"] is False
    assert body["reasons"][0]["code"] == "NOTIONAL_EXCEEDS_HARD_CAP"


def test_market_data_failure_rejects_base_order(monkeypatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise httpx.HTTPError("offline")

    monkeypatch.setattr(risk_app.httpx, "get", fail)
    payload = dict(VALID, quantity={"kind": "base", "value": "0.01"})
    response = TestClient(app).post("/validate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["approved"] is False
    assert body["reasons"][0]["code"] == "MARKET_DATA_UNAVAILABLE"


def test_base_order_uses_market_data(monkeypatch) -> None:
    called = {"count": 0}

    def fake_get(url: str, **kwargs: object) -> FakeResponse | FakePnlResponse:
        called["count"] += 1
        if url.endswith("/pnl/today"):
            return FakePnlResponse("0")
        return FakeResponse()

    monkeypatch.setattr(risk_app.httpx, "get", fake_get)
    payload = dict(VALID, quantity={"kind": "base", "value": "0.001"})
    response = TestClient(app).post("/validate", json=payload)
    assert response.status_code == 200
    assert response.json()["approved"] is True
    assert called["count"] == 2


def test_live_disabled_by_default(monkeypatch) -> None:
    monkeypatch.setattr(risk_app, "LIVE_TRADING_ENABLED", False)
    response = TestClient(app).post("/validate", json=dict(VALID, mode="live"))
    assert response.status_code == 200
    body = response.json()
    assert body["approved"] is False
    assert body["reasons"][0]["code"] == "LIVE_TRADING_DISABLED"


def test_live_enabled_market_order_proceeds_to_notional_check(monkeypatch) -> None:
    monkeypatch.setattr(risk_app, "LIVE_TRADING_ENABLED", True)

    def fail_if_called(url: str, **kwargs: object) -> FakePnlResponse:
        if url.endswith("/pnl/today"):
            return FakePnlResponse("0")
        raise AssertionError("quote-kind live order should not call market-data")

    monkeypatch.setattr(risk_app.httpx, "get", fail_if_called)
    response = TestClient(app).post("/validate", json=dict(VALID, mode="live"))
    assert response.status_code == 200
    body = response.json()
    assert body["approved"] is True
    assert not any(reason["code"] == "LIVE_TRADING_DISABLED" for reason in body["reasons"])


def test_limit_order_without_price_rejected_paper() -> None:
    payload = dict(VALID, order_type="limit", limit_price=None)
    response = TestClient(app).post("/validate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["approved"] is False
    assert body["reasons"][0]["code"] == "LIMIT_PRICE_REQUIRED"


def test_limit_order_without_price_rejected_live(monkeypatch) -> None:
    monkeypatch.setattr(risk_app, "LIVE_TRADING_ENABLED", True)
    payload = dict(VALID, mode="live", order_type="limit", limit_price=None)
    response = TestClient(app).post("/validate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["approved"] is False
    assert body["reasons"][0]["code"] == "LIMIT_PRICE_REQUIRED"


def test_limit_order_with_price_approved_paper() -> None:
    payload = dict(VALID, order_type="limit", limit_price="95000.00")
    response = TestClient(app).post("/validate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["approved"] is True


def test_live_enabled_limit_order_with_price_approved(monkeypatch) -> None:
    monkeypatch.setattr(risk_app, "LIVE_TRADING_ENABLED", True)

    def fail_if_called(url: str, **kwargs: object) -> FakePnlResponse:
        if url.endswith("/pnl/today"):
            return FakePnlResponse("0")
        raise AssertionError("quote-kind limit order should not call market-data")

    monkeypatch.setattr(risk_app.httpx, "get", fail_if_called)
    payload = dict(VALID, mode="live", order_type="limit", limit_price="95000.00")
    response = TestClient(app).post("/validate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["approved"] is True
    assert not any(r["code"] == "LIMIT_ORDER_LIVE_UNSUPPORTED" for r in body["reasons"])


def test_live_enabled_base_order_calls_market_data(monkeypatch) -> None:
    monkeypatch.setattr(risk_app, "LIVE_TRADING_ENABLED", True)
    called = {"count": 0}

    def fake_get(url: str, **kwargs: object) -> FakeResponse | FakePnlResponse:
        called["count"] += 1
        if url.endswith("/pnl/today"):
            return FakePnlResponse("0")
        return FakeResponse()

    monkeypatch.setattr(risk_app.httpx, "get", fake_get)
    payload = dict(VALID, mode="live", quantity={"kind": "base", "value": "0.001"})
    response = TestClient(app).post("/validate", json=payload)
    assert response.status_code == 200
    assert response.json()["approved"] is True
    assert called["count"] == 2


def test_unsupported_venue_rejected_without_market_data(monkeypatch) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("market-data should not be called")

    monkeypatch.setattr(risk_app.httpx, "get", fail_if_called)
    response = TestClient(app).post("/validate", json=dict(VALID, venue="binance_futures"))
    assert response.status_code == 200
    body = response.json()
    assert body["approved"] is False
    assert body["reasons"][0]["code"] == "UNSUPPORTED_VENUE"


def test_risk_accepts_ibkr_paper(monkeypatch) -> None:
    monkeypatch.setattr(risk_app, "LIVE_TRADING_ENABLED", False)
    payload = dict(VALID, venue="ibkr_us_equity", symbol="NVDA")
    response = TestClient(app).post("/validate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["approved"] is True
    assert body["reasons"] == []


def test_risk_rejects_ibkr_live_when_live_globally_enabled(monkeypatch) -> None:
    monkeypatch.setattr(risk_app, "LIVE_TRADING_ENABLED", True)
    monkeypatch.setattr(risk_app, "IBKR_LIVE_TRADING_ENABLED", False)
    payload = dict(VALID, mode="live", venue="ibkr_us_equity", symbol="NVDA")
    response = TestClient(app).post("/validate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["approved"] is False
    assert body["reasons"][0]["code"] == "IBKR_LIVE_TRADING_DISABLED"
    assert "IBKR_LIVE_TRADING_ENABLED" in body["reasons"][0]["detail"]


def test_risk_accepts_ibkr_live_when_global_and_venue_enabled(monkeypatch) -> None:
    monkeypatch.setattr(risk_app, "LIVE_TRADING_ENABLED", True)
    monkeypatch.setattr(risk_app, "IBKR_LIVE_TRADING_ENABLED", True)
    payload = dict(VALID, mode="live", venue="ibkr_us_equity", symbol="NVDA")
    response = TestClient(app).post("/validate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["approved"] is True
    assert not any(reason["code"] == "LIVE_NOT_AVAILABLE_PHASE_21" for reason in body["reasons"])


def test_risk_live_disabled_global_takes_precedence_over_stock_venue(monkeypatch) -> None:
    monkeypatch.setattr(risk_app, "LIVE_TRADING_ENABLED", False)
    payload = dict(VALID, mode="live", venue="ibkr_us_equity", symbol="NVDA")
    response = TestClient(app).post("/validate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["approved"] is False
    assert body["reasons"][0]["code"] == "LIVE_TRADING_DISABLED"


def test_schema_rejects_unknown_fields_and_zero_quantity() -> None:
    payload = dict(VALID, extra="nope")
    assert TestClient(app).post("/validate", json=payload).status_code == 400
    payload = dict(VALID)
    payload["quantity"] = {"kind": "quote", "value": "0"}
    assert TestClient(app).post("/validate", json=payload).status_code == 400


def test_hard_caps_applied_contains_max_notional_and_symbol_exposure() -> None:
    response = TestClient(app).post("/validate", json=VALID)
    caps = response.json()["hard_caps_applied"]
    assert caps["max_notional"] == "10000"
    assert caps["max_drawdown_today"] == "1000"
    assert caps["per_symbol_exposure"] == "50000"


def test_drawdown_breached_rejects_live(monkeypatch) -> None:
    monkeypatch.setattr(risk_app, "LIVE_TRADING_ENABLED", True)
    monkeypatch.setattr(risk_app.httpx, "get", make_httpx_get(total_pnl="-1500"))
    response = TestClient(app).post("/validate", json=dict(VALID, mode="live"))
    body = response.json()
    assert body["approved"] is False
    assert body["reasons"][0]["code"] == "DAILY_DRAWDOWN_BREACHED"


def test_drawdown_breached_rejects_paper_too(monkeypatch) -> None:
    monkeypatch.setattr(risk_app.httpx, "get", make_httpx_get(total_pnl="-1500"))
    response = TestClient(app).post("/validate", json=VALID)
    body = response.json()
    assert body["approved"] is False
    assert body["reasons"][0]["code"] == "DAILY_DRAWDOWN_BREACHED"


def test_drawdown_within_limit_approved(monkeypatch) -> None:
    monkeypatch.setattr(risk_app.httpx, "get", make_httpx_get(total_pnl="-500"))
    response = TestClient(app).post("/validate", json=VALID)
    assert response.json()["approved"] is True


def test_drawdown_lookup_failure_blocks_live(monkeypatch) -> None:
    monkeypatch.setattr(risk_app, "LIVE_TRADING_ENABLED", True)

    def fail(*args: object, **kwargs: object) -> None:
        raise httpx.HTTPError("offline")

    monkeypatch.setattr(risk_app.httpx, "get", fail)
    response = TestClient(app).post("/validate", json=dict(VALID, mode="live"))
    body = response.json()
    assert body["approved"] is False
    assert body["reasons"][0]["code"] == "DRAWDOWN_CHECK_UNAVAILABLE"


def test_drawdown_lookup_failure_allows_paper(monkeypatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise httpx.HTTPError("offline")

    monkeypatch.setattr(risk_app.httpx, "get", fail)
    response = TestClient(app).post("/validate", json=VALID)
    assert response.json()["approved"] is True


def test_hard_caps_max_drawdown_today_populated() -> None:
    response = TestClient(app).post("/validate", json=VALID)
    caps = response.json()["hard_caps_applied"]
    assert caps["max_drawdown_today"] == "1000"
