import hashlib
import hmac
import importlib.util
import json
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def load_service_app(name: str):
    service_dir = Path(__file__).resolve().parents[1]
    sys.modules.pop("db", None)
    sys.path.insert(0, str(service_dir))
    path = service_dir / "app.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


orchestrator_app = load_service_app("orchestrator_service_app")
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


def claude_response(
    symbol: str = "BTCUSDT",
    side: str = "buy",
    order_type: str = "market",
    quantity_kind: str = "quote",
    quantity_value: str = "100",
    limit_price: object = None,
) -> dict[str, object]:
    """Simulate a successful Claude API response."""
    extracted = {
        "symbol": symbol,
        "side": side,
        "order_type": order_type,
        "quantity_kind": quantity_kind,
        "quantity_value": quantity_value,
        "limit_price": limit_price,
    }
    return {
        "content": [{"type": "text", "text": json.dumps(extracted)}],
        "model": "claude-haiku-4-5-20251001",
        "role": "assistant",
    }


class FakeResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class FakePriceResponse:
    def __init__(self, price: str = "100000.00", source: str = "binance") -> None:
        self.price = price
        self.source = source

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"price": self.price, "source": self.source}


@pytest.fixture(autouse=True)
def default_market_data(monkeypatch) -> None:
    monkeypatch.setattr(orchestrator_app.httpx, "get", lambda *args, **kwargs: FakePriceResponse())


def decision(
    intent_id: object = VALID["intent_id"],
    approved: bool = True,
    requires_confirmation: bool = False,
    reasons: list[dict[str, str]] | None = None,
    token: str | None = None,
    expires_at: str | None = None,
) -> dict[str, object]:
    return {
        "decision_id": "33333333-3333-4333-8333-333333333333",
        "intent_id": intent_id,
        "evaluated_at": "2026-05-25T00:00:01Z",
        "approved": approved,
        "reasons": reasons or [],
        "requires_confirmation": requires_confirmation,
        "confirmation_token": token,
        "confirmation_expires_at": expires_at,
        "hard_caps_applied": {
            "max_notional": "10000",
            "max_leverage": None,
            "max_drawdown_today": None,
            "per_symbol_exposure": None,
        },
        "evaluator_version": "risk-engine@0.1.0",
    }


def execution(
    intent_id: object = VALID["intent_id"],
    filled_qty: str = "0.001",
    avg_price: str = "100000.00",
    execution_id: str = "44444444-4444-4444-8444-444444444444",
) -> dict[str, object]:
    return {
        "execution_id": execution_id,
        "intent_id": intent_id,
        "decision_id": "33333333-3333-4333-8333-333333333333",
        "idempotency_key": VALID["idempotency_key"],
        "status": "simulated",
        "venue_order_id": None,
        "fills": [
            {
                "price": avg_price,
                "qty": filled_qty,
                "fee": "0",
                "fee_asset": "USDT",
                "ts": "2026-05-25T00:00:02Z",
            }
        ],
        "avg_price": avg_price,
        "filled_qty": filled_qty,
        "remaining_qty": "0",
        "error": None,
        "raw_venue_response_ref": None,
        "finalized_at": "2026-05-25T00:00:03Z",
    }


def open_execution(
    intent_id: object = VALID["intent_id"],
    venue_order_id: str = "999",
) -> dict[str, object]:
    return {
        "execution_id": "44444444-4444-4444-8444-444444444444",
        "intent_id": intent_id,
        "decision_id": "33333333-3333-4333-8333-333333333333",
        "idempotency_key": VALID["idempotency_key"],
        "status": "open",
        "venue_order_id": venue_order_id,
        "fills": [],
        "avg_price": None,
        "filled_qty": "0",
        "remaining_qty": "0.001",
        "error": None,
        "raw_venue_response_ref": venue_order_id,
        "finalized_at": "2026-05-25T00:00:03Z",
    }


def make_scorecard_payload(
    actor: str = "user_1",
    symbol: str = "BTCUSDT",
    action: str = "buy",
    conviction: str = "0.8",
    source: str = "manual",
    time_horizon: str = "swing",
    ttl_minutes: int | None = 60,
    entry_low: str | None = "95000.00",
    entry_high: str | None = "100000.00",
    stop_loss: str | None = "90000.00",
    take_profit: str | None = "110000.00",
    thesis: str = "Breakout above 100k support retest",
) -> dict[str, object]:
    return {
        "actor": actor,
        "symbol": symbol,
        "action": action,
        "source": source,
        "conviction": conviction,
        "thesis": thesis,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "time_horizon": time_horizon,
        "ttl_minutes": ttl_minutes,
    }


def issue_live_unlock_for_test(monkeypatch, client: TestClient, actor: str = "user_1") -> str:
    monkeypatch.setattr(orchestrator_app, "OPS_TOKEN", "secret")
    return client.post(
        "/admin/live-unlock",
        headers={"x-ops-token": "secret"},
        json={"actor": actor},
    ).json()["token"]


def test_post_intents_small_notional_executes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        return FakeResponse(decision() if url.endswith("/validate") else execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    created = TestClient(orchestrator_app.app).post("/intents", json=VALID)
    assert created.status_code == 200
    assert created.json()["execution"]["status"] == "simulated"


def test_post_intents_medium_notional_returns_202(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        assert url.endswith("/validate")
        return FakeResponse(
            decision(
                requires_confirmation=True,
                token="55555555-5555-4555-8555-555555555555",
                expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
            )
        )

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    response = TestClient(orchestrator_app.app).post(
        "/intents", json=dict(VALID, quantity={"kind": "quote", "value": "1000"})
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending_confirmation"
    assert body["confirmation_token"] == "55555555-5555-4555-8555-555555555555"


def test_post_intents_large_notional_returns_422(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(
            decision(
                approved=False,
                reasons=[{"code": "NOTIONAL_EXCEEDS_HARD_CAP", "detail": "too large"}],
            )
        ),
    )
    response = TestClient(orchestrator_app.app).post(
        "/intents", json=dict(VALID, quantity={"kind": "quote", "value": "50000"})
    )
    assert response.status_code == 422
    assert response.json()["code"] == "RISK_REJECTED"
    assert response.json()["reasons"][0]["code"] == "NOTIONAL_EXCEEDS_HARD_CAP"


def test_live_intent_rejected_by_risk_returns_422(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(
            decision(
                approved=False,
                reasons=[{"code": "LIVE_TRADING_DISABLED", "detail": "disabled"}],
            )
        ),
    )
    client = TestClient(orchestrator_app.app)
    unlock = issue_live_unlock_for_test(monkeypatch, client)
    response = client.post(
        "/intents", headers={"x-live-unlock": unlock}, json=dict(VALID, mode="live")
    )
    assert response.status_code == 422
    assert response.json()["code"] == "RISK_REJECTED"
    assert response.json()["reasons"][0]["code"] == "LIVE_TRADING_DISABLED"


def test_confirm_happy_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    token = "55555555-5555-4555-8555-555555555555"

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(
                decision(
                    requires_confirmation=True,
                    token=token,
                    expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
                )
            )
        return FakeResponse(execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    pending = client.post("/intents", json=dict(VALID, quantity={"kind": "quote", "value": "1000"}))
    assert pending.status_code == 202
    response = client.post(
        f"/intents/{VALID['intent_id']}/confirm",
        json={"intent_id": VALID["intent_id"], "confirmation_token": token},
    )
    assert response.status_code == 200
    assert response.json()["execution"]["status"] == "simulated"


def test_confirm_wrong_token(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    token = "55555555-5555-4555-8555-555555555555"
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(
            decision(
                requires_confirmation=True,
                token=token,
                expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
            )
        ),
    )
    client = TestClient(orchestrator_app.app)
    client.post("/intents", json=dict(VALID, quantity={"kind": "quote", "value": "1000"}))
    response = client.post(
        f"/intents/{VALID['intent_id']}/confirm",
        json={"intent_id": VALID["intent_id"], "confirmation_token": "wrong"},
    )
    assert response.status_code == 403
    assert response.json()["code"] == "INVALID_CONFIRMATION_TOKEN"


def test_confirm_expired_token(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    token = "55555555-5555-4555-8555-555555555555"
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(
            decision(
                requires_confirmation=True,
                token=token,
                expires_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
            )
        ),
    )
    client = TestClient(orchestrator_app.app)
    client.post("/intents", json=dict(VALID, quantity={"kind": "quote", "value": "1000"}))
    response = client.post(
        f"/intents/{VALID['intent_id']}/confirm",
        json={"intent_id": VALID["intent_id"], "confirmation_token": token},
    )
    assert response.status_code == 410
    assert response.json()["code"] == "CONFIRMATION_EXPIRED"


def test_confirm_already_executed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    token = "55555555-5555-4555-8555-555555555555"

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(
                decision(
                    requires_confirmation=True,
                    token=token,
                    expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
                )
            )
        return FakeResponse(execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    client.post("/intents", json=dict(VALID, quantity={"kind": "quote", "value": "1000"}))
    payload = {"intent_id": VALID["intent_id"], "confirmation_token": token}
    assert client.post(f"/intents/{VALID['intent_id']}/confirm", json=payload).status_code == 200
    response = client.post(f"/intents/{VALID['intent_id']}/confirm", json=payload)
    assert response.status_code == 409
    assert response.json()["code"] == "ALREADY_EXECUTED"


def test_get_intents_lists_items_and_total(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        content = kwargs.get("content")
        intent_id = "11111111-1111-4111-8111-111111111111"
        if isinstance(content, str) and "99999999" in content:
            intent_id = "99999999-9999-4999-8999-999999999999"
        if url.endswith("/validate"):
            return FakeResponse(decision(intent_id=intent_id))
        return FakeResponse(execution(intent_id=intent_id))

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    client.post("/intents", json=VALID)
    second = dict(
        VALID,
        intent_id="99999999-9999-4999-8999-999999999999",
        idempotency_key="demo-paper-2",
    )
    client.post("/intents", json=second)
    response = client.get("/intents")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert len(body["items"]) == 2
    assert body["items"][0]["intent"]["intent_id"] == "99999999-9999-4999-8999-999999999999"


def test_get_intents_filters_mode(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        return FakeResponse(decision() if url.endswith("/validate") else execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    response = TestClient(orchestrator_app.app).post("/intents", json=VALID)
    assert response.status_code == 200
    listing = TestClient(orchestrator_app.app).get("/intents", params={"mode": "paper"})
    assert listing.status_code == 200
    assert listing.json()["total"] == 1


def test_get_intent_still_works(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda url, **kwargs: FakeResponse(
            decision() if url.endswith("/validate") else execution()
        ),
    )
    client = TestClient(orchestrator_app.app)
    created = client.post("/intents", json=VALID)
    assert created.status_code == 200
    fetched = client.get(f"/intents/{VALID['intent_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["execution"]["status"] == "simulated"


def test_zero_quantity_schema_error_is_400() -> None:
    payload = dict(VALID)
    payload["quantity"] = {"kind": "quote", "value": "0"}
    response = TestClient(orchestrator_app.app).post("/intents", json=payload)
    assert response.status_code == 400


def test_post_intents_quote_kind_computes_correct_qty(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app.httpx, "get", lambda *args, **kwargs: FakePriceResponse("50000.00")
    )

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(decision())
        headers = kwargs.get("headers")
        assert isinstance(headers, dict)
        return FakeResponse(execution(filled_qty=str(headers["x-quantity"]), avg_price="50000.00"))

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    response = TestClient(orchestrator_app.app).post("/intents", json=VALID)
    assert response.status_code == 200
    assert response.json()["execution"]["filled_qty"] == "0.00200000"


def test_post_intents_base_kind_unchanged(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app.httpx, "get", lambda *args, **kwargs: FakePriceResponse("50000.00")
    )

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(decision())
        headers = kwargs.get("headers")
        assert isinstance(headers, dict)
        return FakeResponse(execution(filled_qty=str(headers["x-quantity"]), avg_price="50000.00"))

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    payload = dict(VALID, quantity={"kind": "base", "value": "0.01"})
    response = TestClient(orchestrator_app.app).post("/intents", json=payload)
    assert response.status_code == 200
    assert response.json()["execution"]["filled_qty"] == "0.01"


def test_market_data_failure_returns_502(monkeypatch, tmp_path: Path) -> None:
    import httpx

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app.httpx, "post", lambda *args, **kwargs: FakeResponse(decision())
    )

    def fail(*args: object, **kwargs: object) -> None:
        raise httpx.HTTPError("offline")

    monkeypatch.setattr(orchestrator_app.httpx, "get", fail)
    response = TestClient(orchestrator_app.app).post("/intents", json=VALID)
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "MARKET_DATA_UNAVAILABLE"


def test_idempotency_returns_cached_executed_result(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    calls = {"risk": 0}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            calls["risk"] += 1
            return FakeResponse(decision())
        return FakeResponse(execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    first = client.post("/intents", json=VALID)
    second = client.post("/intents", json=VALID)
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == first.json()
    assert calls["risk"] == 1


def test_idempotency_returns_cached_pending(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    token = "55555555-5555-4555-8555-555555555555"
    calls = {"risk": 0}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        assert url.endswith("/validate")
        calls["risk"] += 1
        return FakeResponse(
            decision(
                requires_confirmation=True,
                token=token,
                expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
            )
        )

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    first = client.post("/intents", json=dict(VALID, quantity={"kind": "quote", "value": "1000"}))
    second = client.post("/intents", json=dict(VALID, quantity={"kind": "quote", "value": "1000"}))
    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json()["confirmation_token"] == token
    assert calls["risk"] == 1


def test_idempotency_different_intent_id_same_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda url, **kwargs: FakeResponse(
            decision() if url.endswith("/validate") else execution()
        ),
    )
    client = TestClient(orchestrator_app.app)
    first = client.post("/intents", json=VALID)
    payload = dict(VALID, intent_id="99999999-9999-4999-8999-999999999999")
    second = client.post("/intents", json=payload)
    assert second.status_code == 200
    assert second.json() == first.json()


def test_exposure_limit_is_per_symbol(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PER_SYMBOL_DAILY_LIMIT_USDT", "750")
    monkeypatch.setattr(
        orchestrator_app.httpx, "get", lambda *args, **kwargs: FakePriceResponse("50000.00")
    )
    counter = {"n": 0}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(decision())
        counter["n"] += 1
        headers = kwargs.get("headers")
        assert isinstance(headers, dict)
        return FakeResponse(
            execution(
                filled_qty=str(headers["x-quantity"]),
                avg_price="50000.00",
                execution_id=f"44444444-4444-4444-8444-44444444444{counter['n']}",
            )
        )

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    first = client.post("/intents", json=dict(VALID, quantity={"kind": "quote", "value": "400"}))
    second_payload = dict(
        VALID,
        intent_id="99999999-9999-4999-8999-999999999999",
        idempotency_key="demo-paper-2",
        quantity={"kind": "quote", "value": "400"},
    )
    second = client.post("/intents", json=second_payload)
    third_payload = dict(
        second_payload,
        intent_id="88888888-8888-4888-8888-888888888888",
        idempotency_key="demo-paper-3",
        symbol="ETHUSDT",
    )
    third = client.post("/intents", json=third_payload)
    assert first.status_code == 200
    assert second.status_code == 422
    assert second.json()["code"] == "PER_SYMBOL_DAILY_LIMIT_EXCEEDED"
    assert third.status_code == 200


def test_exposure_live_mode_enforces_limit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PER_SYMBOL_DAILY_LIMIT_USDT", "0")
    client = TestClient(orchestrator_app.app)
    unlock = issue_live_unlock_for_test(monkeypatch, client)
    response = client.post(
        "/intents", headers={"x-live-unlock": unlock}, json=dict(VALID, mode="live")
    )
    assert response.status_code == 422
    assert response.json()["code"] == "PER_SYMBOL_DAILY_LIMIT_EXCEEDED"


def test_confirm_records_fill(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    token = "55555555-5555-4555-8555-555555555555"

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(
                decision(
                    requires_confirmation=True,
                    token=token,
                    expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
                )
            )
        return FakeResponse(execution(filled_qty="0.004", avg_price="100000.00"))

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    client.post("/intents", json=dict(VALID, quantity={"kind": "quote", "value": "1000"}))
    response = client.post(
        f"/intents/{VALID['intent_id']}/confirm",
        json={"intent_id": VALID["intent_id"], "confirmation_token": token},
    )
    assert response.status_code == 200
    with orchestrator_app.connect() as conn:
        assert conn.execute("select count(*) from daily_fills").fetchone()[0] == 1


def test_get_exposure_returns_correct_totals(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PER_SYMBOL_DAILY_LIMIT_USDT", "1000")
    monkeypatch.setattr(
        orchestrator_app.httpx, "get", lambda *args, **kwargs: FakePriceResponse("50000.00")
    )
    counter = {"n": 0}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            content = kwargs.get("content")
            intent_id = "11111111-1111-4111-8111-111111111111"
            if isinstance(content, str) and "99999999" in content:
                intent_id = "99999999-9999-4999-8999-999999999999"
            return FakeResponse(decision(intent_id=intent_id))
        counter["n"] += 1
        headers = kwargs.get("headers")
        assert isinstance(headers, dict)
        return FakeResponse(
            execution(
                filled_qty=str(headers["x-quantity"]),
                avg_price="50000.00",
                execution_id=f"44444444-4444-4444-8444-44444444444{counter['n']}",
            )
        )

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    btc = client.post("/intents", json=dict(VALID, quantity={"kind": "quote", "value": "400"}))
    assert btc.status_code == 200
    eth = dict(
        VALID,
        intent_id="99999999-9999-4999-8999-999999999999",
        idempotency_key="demo-paper-2",
        symbol="ETHUSDT",
        quantity={"kind": "quote", "value": "300"},
    )
    assert client.post("/intents", json=eth).status_code == 200
    response = client.get("/exposure")
    assert response.status_code == 200
    symbols = response.json()["symbols"]
    assert Decimal(symbols["BTCUSDT"]["side_buy"]) == Decimal("400")
    assert Decimal(symbols["ETHUSDT"]["side_buy"]) == Decimal("300")


def test_get_exposure_empty_date(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    response = TestClient(orchestrator_app.app).get("/exposure", params={"date": "1970-01-01"})
    assert response.status_code == 200
    assert response.json() == {"date": "1970-01-01", "limit_usdt": "50000", "symbols": {}}


def test_sqlite_path_migrates_phase1_to_trading(monkeypatch, tmp_path: Path) -> None:
    old = tmp_path / "phase1.sqlite"
    sqlite3.connect(old).close()
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    path = orchestrator_app.connect().execute("pragma database_list").fetchone()[2]
    assert Path(path).name == "trading.sqlite"
    assert not old.exists()
    assert (tmp_path / "trading.sqlite").exists()


def test_cancel_pending_returns_204_and_get_shows_canceled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    token = "55555555-5555-4555-8555-555555555555"

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        assert url.endswith("/validate")
        return FakeResponse(
            decision(
                requires_confirmation=True,
                token=token,
                expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
            )
        )

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    pending = client.post("/intents", json=dict(VALID, quantity={"kind": "quote", "value": "1000"}))
    assert pending.status_code == 202
    canceled = client.delete(f"/intents/{VALID['intent_id']}")
    assert canceled.status_code == 204
    fetched = client.get(f"/intents/{VALID['intent_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "canceled"


def test_cancel_executed_returns_409(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda url, **kwargs: FakeResponse(
            decision() if url.endswith("/validate") else execution()
        ),
    )
    client = TestClient(orchestrator_app.app)
    assert client.post("/intents", json=VALID).status_code == 200
    response = client.delete(f"/intents/{VALID['intent_id']}")
    assert response.status_code == 409
    assert response.json() == {"code": "CANNOT_CANCEL", "current_status": "executed"}


def test_cancel_unknown_returns_404(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    response = TestClient(orchestrator_app.app).delete(
        "/intents/99999999-9999-4999-8999-999999999999"
    )
    assert response.status_code == 404
    assert response.json()["code"] == "INTENT_NOT_FOUND"


def test_resubmit_same_idempotency_after_cancel_returns_410(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    token = "55555555-5555-4555-8555-555555555555"
    calls = {"risk": 0}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        assert url.endswith("/validate")
        calls["risk"] += 1
        return FakeResponse(
            decision(
                requires_confirmation=True,
                token=token,
                expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
            )
        )

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    payload = dict(VALID, quantity={"kind": "quote", "value": "1000"})
    assert client.post("/intents", json=payload).status_code == 202
    assert client.delete(f"/intents/{VALID['intent_id']}").status_code == 204
    response = client.post("/intents", json=payload)
    assert response.status_code == 410
    assert response.json()["code"] == "INTENT_CANCELED"
    assert calls["risk"] == 1


def test_paper_positions_weighted_average_after_two_buys(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    prices = ["100000.00", "102000.00"]
    last_price = {"value": prices[0]}

    def fake_get(*args: object, **kwargs: object) -> FakePriceResponse:
        price = prices.pop(0)
        last_price["value"] = price
        return FakePriceResponse(price)

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        content = kwargs.get("content")
        intent_id = "11111111-1111-4111-8111-111111111111"
        if isinstance(content, str) and "99999999" in content:
            intent_id = "99999999-9999-4999-8999-999999999999"
        if url.endswith("/validate"):
            return FakeResponse(decision(intent_id=intent_id))
        headers = kwargs.get("headers")
        assert isinstance(headers, dict)
        return FakeResponse(
            execution(
                intent_id=intent_id,
                filled_qty=str(headers["x-quantity"]),
                avg_price=last_price["value"],
                execution_id="44444444-4444-4444-8444-444444444445"
                if "99999999" in str(intent_id)
                else "44444444-4444-4444-8444-444444444444",
            )
        )

    monkeypatch.setattr(orchestrator_app.httpx, "get", fake_get)
    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    first = dict(VALID, quantity={"kind": "base", "value": "0.01"})
    second = dict(
        first,
        intent_id="99999999-9999-4999-8999-999999999999",
        idempotency_key="demo-paper-2",
    )
    assert client.post("/intents", json=first).status_code == 200
    assert client.post("/intents", json=second).status_code == 200
    with orchestrator_app.connect() as conn:
        row = conn.execute("select * from paper_positions where actor=?", ("user_1",)).fetchone()
    assert row["qty"] == "0.02000000"
    assert row["avg_cost"] == "101000.00000000"
    assert row["total_cost"] == "2020.00000000"


def test_paper_positions_partial_sell_realized_pnl(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    prices = ["100000.00", "110000.00"]
    last_price = {"value": prices[0]}

    def fake_get(*args: object, **kwargs: object) -> FakePriceResponse:
        price = prices.pop(0)
        last_price["value"] = price
        return FakePriceResponse(price)

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        content = kwargs.get("content")
        intent_id = "11111111-1111-4111-8111-111111111111"
        if isinstance(content, str) and "99999999" in content:
            intent_id = "99999999-9999-4999-8999-999999999999"
        if url.endswith("/validate"):
            return FakeResponse(decision(intent_id=intent_id))
        headers = kwargs.get("headers")
        assert isinstance(headers, dict)
        return FakeResponse(
            execution(
                intent_id=intent_id,
                filled_qty=str(headers["x-quantity"]),
                avg_price=last_price["value"],
                execution_id="44444444-4444-4444-8444-444444444445"
                if "99999999" in str(intent_id)
                else "44444444-4444-4444-8444-444444444444",
            )
        )

    monkeypatch.setattr(orchestrator_app.httpx, "get", fake_get)
    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    buy = client.post("/intents", json=dict(VALID, quantity={"kind": "base", "value": "0.02"}))
    assert buy.status_code == 200
    sell = dict(
        VALID,
        intent_id="99999999-9999-4999-8999-999999999999",
        idempotency_key="demo-paper-2",
        side="sell",
        quantity={"kind": "base", "value": "0.01"},
    )
    assert client.post("/intents", json=sell).status_code == 200
    with orchestrator_app.connect() as conn:
        row = conn.execute("select * from paper_positions where actor=?", ("user_1",)).fetchone()
    assert row["qty"] == "0.01000000"
    assert row["avg_cost"] == "100000.00000000"
    assert row["total_cost"] == "1000.00000000"
    assert row["realized_pnl"] == "100.00000000"


def test_get_paper_positions_requires_actor(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    response = TestClient(orchestrator_app.app).get("/paper/positions")
    assert response.status_code == 400
    assert response.json()["code"] == "ACTOR_REQUIRED"


def test_get_paper_positions_marks_open_position(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda url, **kwargs: FakeResponse(
            decision() if url.endswith("/validate") else execution(filled_qty="0.01")
        ),
    )
    client = TestClient(orchestrator_app.app)
    buy = client.post("/intents", json=dict(VALID, quantity={"kind": "base", "value": "0.01"}))
    assert buy.status_code == 200
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "get",
        lambda *args, **kwargs: FakePriceResponse("105000.00", source="binance"),
    )
    response = client.get("/paper/positions", params={"actor": "user_1"})
    assert response.status_code == 200
    position = response.json()["positions"][0]
    assert position["mark_price"] == "105000.00000000"
    assert position["mark_value"] == "1050.00000000"
    assert position["unrealized_pnl"] == "50.00000000"
    assert position["mark_source"] == "binance"


def test_get_paper_positions_mark_failure_returns_nulls(monkeypatch, tmp_path: Path) -> None:
    import httpx

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda url, **kwargs: FakeResponse(
            decision() if url.endswith("/validate") else execution(filled_qty="0.01")
        ),
    )
    client = TestClient(orchestrator_app.app)
    buy = client.post("/intents", json=dict(VALID, quantity={"kind": "base", "value": "0.01"}))
    assert buy.status_code == 200

    def fail(*args: object, **kwargs: object) -> None:
        raise httpx.HTTPError("offline")

    monkeypatch.setattr(orchestrator_app.httpx, "get", fail)
    response = client.get("/paper/positions", params={"actor": "user_1"})
    assert response.status_code == 200
    position = response.json()["positions"][0]
    assert position["mark_price"] is None
    assert position["mark_value"] is None
    assert position["unrealized_pnl"] is None
    assert position["mark_source"] is None


def test_get_paper_positions_zero_qty_skips_mark_call(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    prices = ["100000.00", "100000.00"]
    last_price = {"value": prices[0]}

    def fake_get(*args: object, **kwargs: object) -> FakePriceResponse:
        price = prices.pop(0)
        last_price["value"] = price
        return FakePriceResponse(price)

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        content = kwargs.get("content")
        intent_id = "11111111-1111-4111-8111-111111111111"
        if isinstance(content, str) and "99999999" in content:
            intent_id = "99999999-9999-4999-8999-999999999999"
        if url.endswith("/validate"):
            return FakeResponse(decision(intent_id=intent_id))
        headers = kwargs.get("headers")
        assert isinstance(headers, dict)
        return FakeResponse(
            execution(
                intent_id=intent_id,
                filled_qty=str(headers["x-quantity"]),
                avg_price=last_price["value"],
                execution_id="44444444-4444-4444-8444-444444444445"
                if "99999999" in str(intent_id)
                else "44444444-4444-4444-8444-444444444444",
            )
        )

    monkeypatch.setattr(orchestrator_app.httpx, "get", fake_get)
    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    buy = client.post("/intents", json=dict(VALID, quantity={"kind": "base", "value": "0.01"}))
    assert buy.status_code == 200
    sell = dict(
        VALID,
        intent_id="99999999-9999-4999-8999-999999999999",
        idempotency_key="demo-paper-2",
        side="sell",
        quantity={"kind": "base", "value": "0.01"},
    )
    assert client.post("/intents", json=sell).status_code == 200
    calls = {"n": 0}

    def fail_if_called(*args: object, **kwargs: object) -> None:
        calls["n"] += 1
        raise AssertionError("closed positions should not be marked")

    monkeypatch.setattr(orchestrator_app.httpx, "get", fail_if_called)
    response = client.get("/paper/positions", params={"actor": "user_1"})
    assert response.status_code == 200
    position = response.json()["positions"][0]
    assert position["qty"] == "0.00000000"
    assert position["mark_price"] is None
    assert calls["n"] == 0


def test_live_intent_now_calls_execution_when_risk_approves(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    calls = {"execution": 0}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(decision())
        calls["execution"] += 1
        result = execution(avg_price="100000.00", filled_qty="0.001")
        result["status"] = "filled"
        return FakeResponse(result)

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    unlock = issue_live_unlock_for_test(monkeypatch, client)
    response = client.post(
        "/intents", headers={"x-live-unlock": unlock}, json=dict(VALID, mode="live")
    )
    assert response.status_code == 200
    assert response.json()["execution"]["status"] == "filled"
    assert calls["execution"] == 1


def test_new_headers_passed_to_execution_service(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    captured: dict[str, str] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(decision())
        headers = kwargs.get("headers")
        assert isinstance(headers, dict)
        captured.update({str(key): str(value) for key, value in headers.items()})
        return FakeResponse(execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    response = TestClient(orchestrator_app.app).post("/intents", json=VALID)
    assert response.status_code == 200
    assert captured["x-side"] == "buy"
    assert captured["x-quantity-kind"] == "quote"
    assert captured["x-quote-qty"] == "100"


def test_limit_order_headers_passed_to_execution_service(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    captured: dict[str, str] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(decision())
        headers = kwargs.get("headers")
        assert isinstance(headers, dict)
        captured.update({str(k): str(v) for k, v in headers.items()})
        return FakeResponse(execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    payload = dict(
        VALID,
        order_type="limit",
        limit_price="95000.00",
        time_in_force="IOC",
    )
    response = TestClient(orchestrator_app.app).post("/intents", json=payload)
    assert response.status_code == 200
    assert captured["x-order-type"] == "limit"
    assert captured["x-limit-price"] == "95000.00"
    assert captured["x-time-in-force"] == "IOC"


def test_market_order_limit_price_header_is_empty(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    captured: dict[str, str] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(decision())
        headers = kwargs.get("headers")
        assert isinstance(headers, dict)
        captured.update({str(k): str(v) for k, v in headers.items()})
        return FakeResponse(execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    response = TestClient(orchestrator_app.app).post("/intents", json=VALID)
    assert response.status_code == 200
    assert captured["x-order-type"] == "market"
    assert captured["x-limit-price"] == ""


def test_live_limit_order_returns_open_status(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(decision())
        open_exec = execution(filled_qty="0", avg_price="95000.00")
        open_exec["status"] = "open"
        open_exec["avg_price"] = None
        open_exec["filled_qty"] = "0"
        return FakeResponse(open_exec)

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    payload = dict(
        VALID,
        mode="live",
        order_type="limit",
        limit_price="95000.00",
        quantity={"kind": "base", "value": "0.001"},
    )
    client = TestClient(orchestrator_app.app)
    unlock = issue_live_unlock_for_test(monkeypatch, client)
    response = client.post("/intents", headers={"x-live-unlock": unlock}, json=payload)
    assert response.status_code == 200
    assert response.json()["execution"]["status"] == "open"


def test_cancel_live_open_order_calls_execution_service_cancel(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    cancel_calls = {"n": 0}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(decision())
        if url.endswith("/cancel"):
            cancel_calls["n"] += 1
            canceled = open_execution()
            canceled["status"] = "canceled"
            return FakeResponse(canceled)
        return FakeResponse(open_execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    unlock = issue_live_unlock_for_test(monkeypatch, client)
    intent = dict(
        VALID,
        mode="live",
        order_type="limit",
        limit_price="95000.00",
        quantity={"kind": "base", "value": "0.001"},
    )
    assert (
        client.post("/intents", headers={"x-live-unlock": unlock}, json=intent).status_code == 200
    )
    response = client.delete(f"/intents/{VALID['intent_id']}")
    assert response.status_code == 200
    assert response.json()["execution"]["status"] == "canceled"
    assert cancel_calls["n"] == 1


def test_cancel_live_open_order_updates_db(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(decision())
        if url.endswith("/cancel"):
            canceled = open_execution()
            canceled["status"] = "canceled"
            return FakeResponse(canceled)
        return FakeResponse(open_execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    unlock = issue_live_unlock_for_test(monkeypatch, client)
    intent = dict(
        VALID,
        mode="live",
        order_type="limit",
        limit_price="95000.00",
        quantity={"kind": "base", "value": "0.001"},
    )
    client.post("/intents", headers={"x-live-unlock": unlock}, json=intent)
    client.delete(f"/intents/{VALID['intent_id']}")
    fetched = client.get(f"/intents/{VALID['intent_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["execution"]["status"] == "canceled"


def test_cancel_executed_market_order_returns_409(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda url, **kwargs: FakeResponse(
            decision() if url.endswith("/validate") else execution()
        ),
    )
    client = TestClient(orchestrator_app.app)
    assert client.post("/intents", json=VALID).status_code == 200
    response = client.delete(f"/intents/{VALID['intent_id']}")
    assert response.status_code == 409
    assert response.json()["code"] == "CANNOT_CANCEL"


def test_refresh_live_open_order_updates_db_when_filled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(decision())
        if url.endswith("/refresh"):
            filled = open_execution()
            filled["status"] = "filled"
            filled["avg_price"] = "95000.00"
            filled["filled_qty"] = "0.001"
            filled["remaining_qty"] = "0"
            return FakeResponse(filled)
        return FakeResponse(open_execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    unlock = issue_live_unlock_for_test(monkeypatch, client)
    intent = dict(
        VALID,
        mode="live",
        order_type="limit",
        limit_price="95000.00",
        quantity={"kind": "base", "value": "0.001"},
    )
    assert (
        client.post("/intents", headers={"x-live-unlock": unlock}, json=intent).status_code == 200
    )
    response = client.post(f"/intents/{VALID['intent_id']}/refresh")
    assert response.status_code == 200
    assert response.json()["execution"]["status"] == "filled"
    fetched = client.get(f"/intents/{VALID['intent_id']}")
    assert fetched.json()["execution"]["status"] == "filled"


def test_refresh_still_open_does_not_update_db(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    refresh_calls = {"n": 0}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(decision())
        if url.endswith("/refresh"):
            refresh_calls["n"] += 1
            return FakeResponse(open_execution())
        return FakeResponse(open_execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    unlock = issue_live_unlock_for_test(monkeypatch, client)
    intent = dict(
        VALID,
        mode="live",
        order_type="limit",
        limit_price="95000.00",
        quantity={"kind": "base", "value": "0.001"},
    )
    client.post("/intents", headers={"x-live-unlock": unlock}, json=intent)
    response = client.post(f"/intents/{VALID['intent_id']}/refresh")
    assert response.status_code == 200
    assert response.json()["execution"]["status"] == "open"
    assert refresh_calls["n"] == 1


def test_refresh_non_live_order_returns_without_binance_call(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(decision())
        if url.endswith("/refresh"):
            raise AssertionError("refresh should not be called for non-live orders")
        return FakeResponse(execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    assert client.post("/intents", json=VALID).status_code == 200
    response = client.post(f"/intents/{VALID['intent_id']}/refresh")
    assert response.status_code == 200
    assert response.json()["execution"]["status"] == "simulated"


def test_refresh_unknown_intent_returns_404(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    response = TestClient(orchestrator_app.app).post(
        "/intents/99999999-9999-4999-8999-999999999999/refresh"
    )
    assert response.status_code == 404
    assert response.json()["code"] == "INTENT_NOT_FOUND"


def test_refresh_records_fill_when_order_fills(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(decision())
        if url.endswith("/refresh"):
            filled = open_execution()
            filled["status"] = "filled"
            filled["avg_price"] = "95000.00"
            filled["filled_qty"] = "0.001"
            filled["remaining_qty"] = "0"
            return FakeResponse(filled)
        return FakeResponse(open_execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    unlock = issue_live_unlock_for_test(monkeypatch, client)
    intent = dict(
        VALID,
        mode="live",
        order_type="limit",
        limit_price="95000.00",
        quantity={"kind": "base", "value": "0.001"},
    )
    client.post("/intents", headers={"x-live-unlock": unlock}, json=intent)
    client.post(f"/intents/{VALID['intent_id']}/refresh")
    with orchestrator_app.connect() as conn:
        count = conn.execute("select count(*) from daily_fills").fetchone()[0]
    assert count == 1


# -- POST /intents/from_nl ----------------------------------------------------


def test_from_nl_missing_api_key_returns_503(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "CLAUDE_API_KEY", "")
    payload = {
        "actor": "user_1",
        "message": "buy 100 USDT of BTC at market",
        "idempotency_key": "nl-001",
    }
    response = TestClient(orchestrator_app.app).post("/intents/from_nl", json=payload)
    assert response.status_code == 503
    assert response.json()["code"] == "HERMES_UNAVAILABLE"


def test_from_nl_claude_api_failure_returns_503(monkeypatch, tmp_path: Path) -> None:
    import httpx as _httpx

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "CLAUDE_API_KEY", "test-key")

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if "anthropic" in url:
            raise _httpx.HTTPError("connection refused")
        return FakeResponse(decision())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    payload = {
        "actor": "user_1",
        "message": "buy 100 USDT of BTC at market",
        "idempotency_key": "nl-002",
    }
    response = TestClient(orchestrator_app.app).post("/intents/from_nl", json=payload)
    assert response.status_code == 503
    assert response.json()["code"] == "HERMES_UNAVAILABLE"


def test_from_nl_ambiguous_message_returns_400(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "CLAUDE_API_KEY", "test-key")

    error_response = {
        "content": [{"type": "text", "text": '{"error": "instruction is ambiguous"}'}],
        "model": "claude-haiku-4-5-20251001",
        "role": "assistant",
    }

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if "anthropic" in url:
            return FakeResponse(error_response)
        return FakeResponse(decision())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    payload = {
        "actor": "user_1",
        "message": "do something with crypto",
        "idempotency_key": "nl-003",
    }
    response = TestClient(orchestrator_app.app).post("/intents/from_nl", json=payload)
    assert response.status_code == 400
    assert response.json()["code"] == "HERMES_PARSE_ERROR"
    assert "ambiguous" in response.json()["detail"]


def test_from_nl_parses_market_buy_and_executes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "CLAUDE_API_KEY", "test-key")

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if "anthropic" in url:
            return FakeResponse(claude_response())
        if url.endswith("/validate"):
            return FakeResponse(decision())
        return FakeResponse(execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    payload = {
        "actor": "user_1",
        "message": "buy 100 USDT of BTC at market",
        "idempotency_key": "nl-004",
        "hermes_message_id": "msg-xyz",
    }
    response = TestClient(orchestrator_app.app).post("/intents/from_nl", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["execution"]["status"] == "simulated"
    assert body["intent"]["symbol"] == "BTCUSDT"
    assert body["intent"]["side"] == "buy"
    assert body["intent"]["source"]["origin"] == "user_nl"
    assert body["intent"]["source"]["hermes_message_id"] == "msg-xyz"


def test_from_nl_sets_correct_quantity_from_extracted_fields(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "CLAUDE_API_KEY", "test-key")

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if "anthropic" in url:
            return FakeResponse(
                claude_response(
                    symbol="ETHUSDT",
                    side="sell",
                    order_type="limit",
                    quantity_kind="base",
                    quantity_value="0.5",
                    limit_price="3500.00",
                )
            )
        if url.endswith("/validate"):
            return FakeResponse(decision())
        return FakeResponse(execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    payload = {
        "actor": "user_1",
        "message": "sell 0.5 ETH at 3500",
        "idempotency_key": "nl-005",
    }
    response = TestClient(orchestrator_app.app).post("/intents/from_nl", json=payload)
    assert response.status_code == 200
    intent = response.json()["intent"]
    assert intent["symbol"] == "ETHUSDT"
    assert intent["side"] == "sell"
    assert intent["order_type"] == "limit"
    assert intent["quantity"]["kind"] == "base"
    assert intent["quantity"]["value"] == "0.5"
    assert intent["limit_price"] == "3500.00"


def test_from_nl_idempotency_deduplicates(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "CLAUDE_API_KEY", "test-key")
    claude_calls = {"n": 0}
    execution_calls = {"n": 0}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if "anthropic" in url:
            claude_calls["n"] += 1
            return FakeResponse(claude_response())
        if url.endswith("/validate"):
            return FakeResponse(decision())
        execution_calls["n"] += 1
        return FakeResponse(execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    payload = {
        "actor": "user_1",
        "message": "buy 100 USDT of BTC at market",
        "idempotency_key": "nl-006",
    }
    first = client.post("/intents/from_nl", json=payload)
    second = client.post("/intents/from_nl", json=payload)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert claude_calls["n"] == 2
    assert execution_calls["n"] == 1


def test_from_nl_rejects_unknown_fields(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "CLAUDE_API_KEY", "test-key")
    payload = {
        "actor": "user_1",
        "message": "buy 100 USDT of BTC",
        "idempotency_key": "nl-007",
        "unknown_field": "oops",
    }
    response = TestClient(orchestrator_app.app).post("/intents/from_nl", json=payload)
    assert response.status_code == 400


def test_from_nl_live_mode_passes_through_risk_rejection(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "CLAUDE_API_KEY", "test-key")

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if "anthropic" in url:
            return FakeResponse(claude_response())
        return FakeResponse(
            decision(
                approved=False,
                reasons=[{"code": "LIVE_TRADING_DISABLED", "detail": "disabled"}],
            )
        )

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    payload = {
        "actor": "user_1",
        "message": "buy 100 USDT of BTC at market",
        "idempotency_key": "nl-008",
        "mode": "live",
    }
    response = TestClient(orchestrator_app.app).post("/intents/from_nl", json=payload)
    assert response.status_code == 403
    assert response.json()["code"] == "LIVE_UNLOCK_REQUIRED"


# -- Phase 9 scorecards / live unlock / PnL ----------------------------------


def test_create_scorecard_returns_persisted_record(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    client = TestClient(orchestrator_app.app)
    created = client.post("/scorecards", json=make_scorecard_payload())
    assert created.status_code == 200
    scorecard_id = created.json()["scorecard_id"]
    fetched = client.get(f"/scorecards/{scorecard_id}")
    assert fetched.status_code == 200
    assert fetched.json()["scorecard_id"] == scorecard_id


def test_get_scorecard_returns_404_for_unknown_id(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    response = TestClient(orchestrator_app.app).get(
        "/scorecards/99999999-9999-4999-8999-999999999999"
    )
    assert response.status_code == 404
    assert response.json()["code"] == "SCORECARD_NOT_FOUND"


def test_list_scorecards_filters_by_actor_and_symbol(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    client = TestClient(orchestrator_app.app)
    client.post("/scorecards", json=make_scorecard_payload(actor="user_1", symbol="BTCUSDT"))
    client.post("/scorecards", json=make_scorecard_payload(actor="user_2", symbol="ETHUSDT"))
    response = client.get("/scorecards?actor=user_1&symbol=BTCUSDT")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["scorecard"]["actor"] == "user_1"


def test_list_scorecards_active_only_excludes_expired(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    client = TestClient(orchestrator_app.app)
    created = client.post("/scorecards", json=make_scorecard_payload(ttl_minutes=1))
    scorecard_id = created.json()["scorecard_id"]
    with orchestrator_app.connect() as conn:
        conn.execute(
            "update scorecards set expires_at = ? where scorecard_id = ?",
            ("2000-01-01T00:00:00+00:00", scorecard_id),
        )
        conn.commit()
    response = client.get("/scorecards?active_only=true")
    assert response.json()["items"] == []


def test_create_scorecard_rejects_conviction_above_one(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    response = TestClient(orchestrator_app.app).post(
        "/scorecards", json=make_scorecard_payload(conviction="1.5")
    )
    assert response.status_code == 400


def test_create_scorecard_rejects_zero_ttl(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    response = TestClient(orchestrator_app.app).post(
        "/scorecards", json=make_scorecard_payload(ttl_minutes=0)
    )
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_TTL"


def test_from_scorecard_market_buy_paper_executes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        return FakeResponse(decision() if url.endswith("/validate") else execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    created = client.post("/scorecards", json=make_scorecard_payload(conviction="0.5"))
    scorecard_id = created.json()["scorecard_id"]
    response = client.post(
        "/intents/from_scorecard",
        json={
            "scorecard_id": scorecard_id,
            "actor": "user_1",
            "idempotency_key": "scorecard-001",
            "usdt_budget": "200",
            "position_fraction": "1.0",
        },
    )
    assert response.status_code == 200
    intent = response.json()["intent"]
    assert intent["quantity"]["value"] == "100.00"
    assert intent["source"]["origin"] == "scorecard"
    assert intent["source"]["scorecard_id"] == scorecard_id


def test_from_scorecard_limit_buy_uses_entry_low(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda url, **kw: FakeResponse(decision() if url.endswith("/validate") else execution()),
    )
    client = TestClient(orchestrator_app.app)
    scorecard_id = client.post("/scorecards", json=make_scorecard_payload(action="buy")).json()[
        "scorecard_id"
    ]
    response = client.post(
        "/intents/from_scorecard",
        json={
            "scorecard_id": scorecard_id,
            "actor": "user_1",
            "idempotency_key": "scorecard-002",
            "usdt_budget": "200",
            "order_type": "limit",
        },
    )
    assert response.status_code == 200
    assert response.json()["intent"]["limit_price"] == "95000.00"


def test_from_scorecard_limit_sell_uses_entry_high(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda url, **kw: FakeResponse(decision() if url.endswith("/validate") else execution()),
    )
    client = TestClient(orchestrator_app.app)
    scorecard_id = client.post("/scorecards", json=make_scorecard_payload(action="sell")).json()[
        "scorecard_id"
    ]
    response = client.post(
        "/intents/from_scorecard",
        json={
            "scorecard_id": scorecard_id,
            "actor": "user_1",
            "idempotency_key": "scorecard-003",
            "usdt_budget": "200",
            "order_type": "limit",
        },
    )
    assert response.status_code == 200
    assert response.json()["intent"]["limit_price"] == "100000.00"


def test_from_scorecard_marks_consumed_after_executed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda url, **kw: FakeResponse(decision() if url.endswith("/validate") else execution()),
    )
    client = TestClient(orchestrator_app.app)
    scorecard_id = client.post("/scorecards", json=make_scorecard_payload()).json()["scorecard_id"]
    response = client.post(
        "/intents/from_scorecard",
        json={
            "scorecard_id": scorecard_id,
            "actor": "user_1",
            "idempotency_key": "scorecard-004",
            "usdt_budget": "200",
        },
    )
    assert response.status_code == 200
    listed = client.get("/scorecards").json()
    assert listed["items"][0]["consumed_by_intent_id"] is not None


def test_from_scorecard_expired_returns_410(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    client = TestClient(orchestrator_app.app)
    scorecard_id = client.post("/scorecards", json=make_scorecard_payload(ttl_minutes=1)).json()[
        "scorecard_id"
    ]
    with orchestrator_app.connect() as conn:
        conn.execute(
            "update scorecards set expires_at = ? where scorecard_id = ?",
            ("2000-01-01T00:00:00+00:00", scorecard_id),
        )
        conn.commit()
    response = client.post(
        "/intents/from_scorecard",
        json={
            "scorecard_id": scorecard_id,
            "actor": "user_1",
            "idempotency_key": "scorecard-005",
            "usdt_budget": "200",
        },
    )
    assert response.status_code == 410
    assert response.json()["code"] == "SCORECARD_EXPIRED"


def test_from_scorecard_hold_returns_400(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    client = TestClient(orchestrator_app.app)
    scorecard_id = client.post("/scorecards", json=make_scorecard_payload(action="hold")).json()[
        "scorecard_id"
    ]
    response = client.post(
        "/intents/from_scorecard",
        json={
            "scorecard_id": scorecard_id,
            "actor": "user_1",
            "idempotency_key": "scorecard-006",
            "usdt_budget": "200",
        },
    )
    assert response.status_code == 400
    assert response.json()["code"] == "SCORECARD_ACTION_HOLD"


def test_from_scorecard_action_actor_mismatch_returns_403(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    client = TestClient(orchestrator_app.app)
    scorecard_id = client.post("/scorecards", json=make_scorecard_payload(actor="user_1")).json()[
        "scorecard_id"
    ]
    response = client.post(
        "/intents/from_scorecard",
        json={
            "scorecard_id": scorecard_id,
            "actor": "user_2",
            "idempotency_key": "scorecard-007",
            "usdt_budget": "200",
        },
    )
    assert response.status_code == 403
    assert response.json()["code"] == "SCORECARD_ACTOR_MISMATCH"


def test_from_scorecard_already_consumed_returns_409(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda url, **kw: FakeResponse(decision() if url.endswith("/validate") else execution()),
    )
    client = TestClient(orchestrator_app.app)
    scorecard_id = client.post("/scorecards", json=make_scorecard_payload()).json()["scorecard_id"]
    payload = {
        "scorecard_id": scorecard_id,
        "actor": "user_1",
        "idempotency_key": "scorecard-008",
        "usdt_budget": "200",
    }
    first = client.post("/intents/from_scorecard", json=payload)
    second = client.post(
        "/intents/from_scorecard", json=dict(payload, idempotency_key="scorecard-009")
    )
    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["intent_id"] == first.json()["intent"]["intent_id"]


def test_from_scorecard_zero_sized_returns_400(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    client = TestClient(orchestrator_app.app)
    scorecard_id = client.post("/scorecards", json=make_scorecard_payload(conviction="0")).json()[
        "scorecard_id"
    ]
    response = client.post(
        "/intents/from_scorecard",
        json={
            "scorecard_id": scorecard_id,
            "actor": "user_1",
            "idempotency_key": "scorecard-010",
            "usdt_budget": "100",
        },
    )
    assert response.status_code == 400
    assert response.json()["code"] == "ZERO_SIZED_INTENT"


def test_from_scorecard_limit_missing_entry_price_returns_400(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    client = TestClient(orchestrator_app.app)
    scorecard_id = client.post(
        "/scorecards", json=make_scorecard_payload(entry_low=None, entry_high=None)
    ).json()["scorecard_id"]
    response = client.post(
        "/intents/from_scorecard",
        json={
            "scorecard_id": scorecard_id,
            "actor": "user_1",
            "idempotency_key": "scorecard-011",
            "usdt_budget": "100",
            "order_type": "limit",
        },
    )
    assert response.status_code == 400
    assert response.json()["code"] == "SCORECARD_MISSING_ENTRY_PRICE"


def test_live_unlock_disabled_when_ops_token_empty(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "OPS_TOKEN", "")
    response = TestClient(orchestrator_app.app).post("/admin/live-unlock", json={"actor": "user_1"})
    assert response.status_code == 503


def test_live_unlock_wrong_ops_token_returns_403(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "OPS_TOKEN", "secret")
    response = TestClient(orchestrator_app.app).post(
        "/admin/live-unlock", headers={"x-ops-token": "wrong"}, json={"actor": "user_1"}
    )
    assert response.status_code == 403


def test_live_unlock_issues_token_with_expiry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "OPS_TOKEN", "secret")
    response = TestClient(orchestrator_app.app).post(
        "/admin/live-unlock", headers={"x-ops-token": "secret"}, json={"actor": "user_1"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token"]
    assert body["actor"] == "user_1"
    assert body["expires_at"]


def test_post_intents_live_without_unlock_token_returns_403(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    response = TestClient(orchestrator_app.app).post("/intents", json=dict(VALID, mode="live"))
    assert response.status_code == 403
    assert response.json()["code"] == "LIVE_UNLOCK_REQUIRED"


def test_post_intents_live_with_invalid_unlock_returns_403(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    response = TestClient(orchestrator_app.app).post(
        "/intents", headers={"x-live-unlock": "bad"}, json=dict(VALID, mode="live")
    )
    assert response.status_code == 403
    assert response.json()["code"] == "INVALID_LIVE_UNLOCK"


def test_post_intents_live_with_expired_unlock_returns_410(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "OPS_TOKEN", "secret")
    client = TestClient(orchestrator_app.app)
    token = client.post(
        "/admin/live-unlock", headers={"x-ops-token": "secret"}, json={"actor": "user_1"}
    ).json()["token"]
    with orchestrator_app.connect() as conn:
        conn.execute(
            "update live_unlock_tokens set expires_at = ? where token = ?",
            ("2000-01-01T00:00:00+00:00", token),
        )
        conn.commit()
    response = client.post(
        "/intents", headers={"x-live-unlock": token}, json=dict(VALID, mode="live")
    )
    assert response.status_code == 410
    assert response.json()["code"] == "LIVE_UNLOCK_EXPIRED"


def test_post_intents_live_consumes_unlock_on_execute(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "OPS_TOKEN", "secret")
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda url, **kw: FakeResponse(decision() if url.endswith("/validate") else execution()),
    )
    client = TestClient(orchestrator_app.app)
    token = client.post(
        "/admin/live-unlock", headers={"x-ops-token": "secret"}, json={"actor": "user_1"}
    ).json()["token"]
    first = client.post(
        "/intents",
        headers={"x-live-unlock": token},
        json=dict(VALID, mode="live", idempotency_key="live-1"),
    )
    second = client.post(
        "/intents",
        headers={"x-live-unlock": token},
        json=dict(
            VALID,
            intent_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            mode="live",
            idempotency_key="live-2",
        ),
    )
    assert first.status_code == 200
    assert second.status_code == 403
    assert second.json()["code"] == "LIVE_UNLOCK_ALREADY_USED"


def test_post_intents_live_pending_confirmation_does_not_consume_token(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "OPS_TOKEN", "secret")
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda url, **kw: FakeResponse(
            decision(
                requires_confirmation=True,
                token="55555555-5555-4555-8555-555555555555",
                expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
            )
        ),
    )
    client = TestClient(orchestrator_app.app)
    token = client.post(
        "/admin/live-unlock", headers={"x-ops-token": "secret"}, json={"actor": "user_1"}
    ).json()["token"]
    response = client.post(
        "/intents",
        headers={"x-live-unlock": token},
        json=dict(
            VALID,
            mode="live",
            quantity={"kind": "quote", "value": "1000"},
            idempotency_key="live-pending",
        ),
    )
    assert response.status_code == 202
    with orchestrator_app.connect() as conn:
        row = conn.execute(
            "select consumed_at from live_unlock_tokens where token = ?", (token,)
        ).fetchone()
    assert row["consumed_at"] is None


def test_confirm_intent_live_consumes_unlock(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "OPS_TOKEN", "secret")
    confirm_token = "55555555-5555-4555-8555-555555555555"

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(
                decision(
                    requires_confirmation=True,
                    token=confirm_token,
                    expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
                )
            )
        return FakeResponse(execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    token = client.post(
        "/admin/live-unlock", headers={"x-ops-token": "secret"}, json={"actor": "user_1"}
    ).json()["token"]
    pending = client.post(
        "/intents",
        headers={"x-live-unlock": token},
        json=dict(
            VALID,
            mode="live",
            quantity={"kind": "quote", "value": "1000"},
            idempotency_key="live-confirm",
        ),
    )
    assert pending.status_code == 202
    response = client.post(
        f"/intents/{VALID['intent_id']}/confirm",
        headers={"x-live-unlock": token},
        json={"intent_id": VALID["intent_id"], "confirmation_token": confirm_token},
    )
    assert response.status_code == 200
    with orchestrator_app.connect() as conn:
        row = conn.execute(
            "select consumed_at from live_unlock_tokens where token = ?", (token,)
        ).fetchone()
    assert row["consumed_at"] is not None


def test_paper_intent_does_not_require_unlock(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda url, **kw: FakeResponse(decision() if url.endswith("/validate") else execution()),
    )
    response = TestClient(orchestrator_app.app).post("/intents", json=VALID)
    assert response.status_code == 200


def test_get_pnl_today_empty_for_unknown_actor(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    response = TestClient(orchestrator_app.app).get("/pnl/today?actor=user_1")
    assert response.status_code == 200
    assert response.json()["total_pnl"] == "0.00000000"


def test_get_pnl_today_aggregates_realized_after_sells(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(decision())
        content = str(kwargs.get("content", ""))
        return FakeResponse(
            execution(filled_qty="0.01", avg_price="105000.00")
            if "sell" in content
            else execution(filled_qty="0.01", avg_price="100000.00")
        )

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    client = TestClient(orchestrator_app.app)
    buy = dict(VALID, quantity={"kind": "base", "value": "0.01"}, idempotency_key="pnl-buy")
    sell = dict(
        VALID,
        intent_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        side="sell",
        quantity={"kind": "base", "value": "0.01"},
        idempotency_key="pnl-sell",
    )
    assert client.post("/intents", json=buy).status_code == 200
    assert client.post("/intents", json=sell).status_code == 200
    response = client.get("/pnl/today?actor=user_1")
    assert response.json()["realized_pnl"] == "50.00000000"


def test_get_pnl_today_includes_unrealized_marks(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda url, **kw: FakeResponse(
            decision()
            if url.endswith("/validate")
            else execution(filled_qty="0.01", avg_price="100000.00")
        ),
    )
    monkeypatch.setattr(
        orchestrator_app.httpx, "get", lambda *args, **kwargs: FakePriceResponse(price="105000.00")
    )
    client = TestClient(orchestrator_app.app)
    buy = dict(VALID, quantity={"kind": "base", "value": "0.01"}, idempotency_key="pnl-unrealized")
    assert client.post("/intents", json=buy).status_code == 200
    response = client.get("/pnl/today?actor=user_1")
    body = response.json()
    assert body["unrealized_pnl"] == "50.00000000"
    assert body["total_pnl"] == "50.00000000"


# -- Phase 10 debt fixes -----------------------------------------------------


def test_from_nl_live_requires_unlock_token(monkeypatch, tmp_path: Path) -> None:
    """NL endpoint must forward x-live-unlock to create_intent (Phase 9 BUG-1)."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "CLAUDE_API_KEY", "test-key")
    monkeypatch.setattr(orchestrator_app, "OPS_TOKEN", "ops")

    client = TestClient(orchestrator_app.app)
    issued = client.post(
        "/admin/live-unlock", json={"actor": "user_1"}, headers={"x-ops-token": "ops"}
    ).json()
    token = issued["token"]

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if "anthropic" in url:
            return FakeResponse(claude_response())
        if url.endswith("/validate"):
            return FakeResponse(decision())
        return FakeResponse(execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    payload = {
        "actor": "user_1",
        "message": "buy 100 USDT of BTC at market",
        "idempotency_key": "nl-live-001",
        "mode": "live",
    }

    no_tok = client.post("/intents/from_nl", json=payload)
    assert no_tok.status_code == 403
    assert no_tok.json()["code"] == "LIVE_UNLOCK_REQUIRED"

    with_tok = client.post("/intents/from_nl", json=payload, headers={"x-live-unlock": token})
    assert with_tok.status_code == 200


def test_from_scorecard_consume_is_conditional_update(monkeypatch, tmp_path: Path) -> None:
    """Steady-state consumed scorecards are rejected; final consume uses rowcount."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda url, **k: (
            FakeResponse(decision()) if url.endswith("/validate") else FakeResponse(execution())
        ),
    )
    monkeypatch.setattr(orchestrator_app.httpx, "get", lambda *a, **k: FakePriceResponse())

    client = TestClient(orchestrator_app.app)
    scorecard = client.post("/scorecards", json=make_scorecard_payload()).json()
    scorecard_id = scorecard["scorecard_id"]

    first = client.post(
        "/intents/from_scorecard",
        json={
            "scorecard_id": scorecard_id,
            "actor": "user_1",
            "idempotency_key": "first",
            "usdt_budget": "200",
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/intents/from_scorecard",
        json={
            "scorecard_id": scorecard_id,
            "actor": "user_1",
            "idempotency_key": "second",
            "usdt_budget": "200",
        },
    )
    assert second.status_code == 409
    assert second.json()["code"] == "SCORECARD_ALREADY_CONSUMED"


def test_scorecard_race_rowcount_returns_409(monkeypatch, tmp_path: Path) -> None:
    """BUG-2 deterministic seam: final conditional UPDATE rowcount=0 returns SCORECARD_RACED."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda url, **k: (
            FakeResponse(decision()) if url.endswith("/validate") else FakeResponse(execution())
        ),
    )
    client = TestClient(orchestrator_app.app)
    scorecard_id = client.post("/scorecards", json=make_scorecard_payload()).json()["scorecard_id"]
    original_should_mark = orchestrator_app._scorecard_should_mark_consumed

    def race_after_response(response: object) -> bool:
        result = original_should_mark(response)
        with orchestrator_app.connect() as conn:
            conn.execute(
                "update scorecards set consumed_by_intent_id = ? where scorecard_id = ?",
                ("concurrent-winner", scorecard_id),
            )
            conn.commit()
        return result

    monkeypatch.setattr(orchestrator_app, "_scorecard_should_mark_consumed", race_after_response)
    response = client.post(
        "/intents/from_scorecard",
        json={
            "scorecard_id": scorecard_id,
            "actor": "user_1",
            "idempotency_key": "raced",
            "usdt_budget": "200",
        },
    )
    assert response.status_code == 409
    assert response.json()["code"] == "SCORECARD_RACED"
    assert "your_intent_id" in response.json()


def test_live_unlock_wet_rowcount_zero_returns_used(monkeypatch, tmp_path: Path) -> None:
    """BUG-3: dry=False must treat conditional UPDATE rowcount=0 as already used."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "OPS_TOKEN", "ops")
    client = TestClient(orchestrator_app.app)
    issued = client.post(
        "/admin/live-unlock", json={"actor": "user_1"}, headers={"x-ops-token": "ops"}
    ).json()
    token = issued["token"]

    calls = {"n": 0}
    original_connect = orchestrator_app.connect

    class RaceCursor:
        rowcount = 0

    class RaceConn:
        def __enter__(self) -> "RaceConn":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def execute(self, sql: str, params: object = ()) -> object:
            if sql.startswith("select actor"):
                calls["n"] += 1
                with original_connect() as conn:
                    return conn.execute(sql, params)
            return RaceCursor()

        def commit(self) -> None:
            return None

    monkeypatch.setattr(orchestrator_app, "connect", lambda: RaceConn())
    err = orchestrator_app._consume_live_unlock_or_error(token, "user_1", dry=False)
    assert err is not None
    body = json.loads(bytes(err.body).decode())
    assert body["code"] == "LIVE_UNLOCK_ALREADY_USED"


# -- Phase 12 scorecard outcomes ---------------------------------------------


def _risk_and_execution(
    monkeypatch, *, avg_price: str = "100000.00", filled_qty: str = "0.001"
) -> None:
    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(decision())
        return FakeResponse(execution(avg_price=avg_price, filled_qty=filled_qty))

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)


def _create_scorecard_order(
    client: TestClient,
    *,
    idempotency_key: str,
    source: str = "tradingagents",
    actor: str = "user_1",
    symbol: str = "BTCUSDT",
    budget: str = "200",
    conviction: str = "0.5",
) -> str:
    scorecard_id = client.post(
        "/scorecards",
        json=make_scorecard_payload(
            actor=actor, symbol=symbol, source=source, conviction=conviction
        ),
    ).json()["scorecard_id"]
    response = client.post(
        "/intents/from_scorecard",
        json={
            "scorecard_id": scorecard_id,
            "actor": actor,
            "idempotency_key": idempotency_key,
            "usdt_budget": budget,
        },
    )
    assert response.status_code == 200
    return scorecard_id


def _manual_sell(client: TestClient, *, idempotency_key: str, qty: str, price: str) -> None:
    payload = dict(
        VALID,
        intent_id=str(orchestrator_app.uuid4()),
        side="sell",
        quantity={"kind": "base", "value": qty},
        idempotency_key=idempotency_key,
    )

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/validate"):
            return FakeResponse(decision())
        return FakeResponse(execution(avg_price=price, filled_qty=qty))

    old_post = orchestrator_app.httpx.post
    orchestrator_app.httpx.post = fake_post
    try:
        response = client.post("/intents", json=payload)
    finally:
        orchestrator_app.httpx.post = old_post
    assert response.status_code == 200


def test_outcome_opens_on_scorecard_buy_fill(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _risk_and_execution(monkeypatch)
    client = TestClient(orchestrator_app.app)
    scorecard_id = _create_scorecard_order(client, idempotency_key="outcome-open")

    response = client.get("/scorecard-outcomes", params={"status": "open"})

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["scorecard_id"] == scorecard_id
    assert item["source"] == "tradingagents"
    assert item["status"] == "open"
    assert item["opened_cost_basis"] == "100.00000000"


def test_outcome_does_not_open_for_manual_intent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _risk_and_execution(monkeypatch)
    client = TestClient(orchestrator_app.app)
    assert client.post("/intents", json=VALID).status_code == 200

    response = client.get("/scorecard-outcomes")

    assert response.status_code == 200
    assert response.json()["items"] == []


def test_outcome_closes_on_full_sell(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _risk_and_execution(monkeypatch, avg_price="100000.00", filled_qty="0.001")
    client = TestClient(orchestrator_app.app)
    _create_scorecard_order(client, idempotency_key="outcome-close-buy")
    _manual_sell(client, idempotency_key="outcome-close-sell", qty="0.001", price="110000.00")

    item = client.get("/scorecard-outcomes", params={"status": "closed"}).json()["items"][0]

    assert item["status"] == "closed"
    assert item["closed_realized_pnl"] == "10.00000000"
    assert item["closed_return_pct"] == "0.10000000"


def test_outcome_partial_sell_keeps_open(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _risk_and_execution(monkeypatch, avg_price="100000.00", filled_qty="0.002")
    client = TestClient(orchestrator_app.app)
    _create_scorecard_order(client, idempotency_key="outcome-partial-buy")
    _manual_sell(client, idempotency_key="outcome-partial-sell", qty="0.001", price="110000.00")

    item = client.get("/scorecard-outcomes").json()["items"][0]

    assert item["status"] == "open"
    assert item["closed_realized_pnl"] is None


def test_outcome_split_attribution_two_scorecards(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _risk_and_execution(monkeypatch, avg_price="100000.00", filled_qty="0.001")
    client = TestClient(orchestrator_app.app)
    _create_scorecard_order(client, idempotency_key="split-a")
    _create_scorecard_order(client, idempotency_key="split-b")
    _manual_sell(client, idempotency_key="split-sell", qty="0.002", price="110000.00")

    items = client.get("/scorecard-outcomes", params={"status": "closed"}).json()["items"]
    pnls = sorted(Decimal(item["closed_realized_pnl"]) for item in items)

    assert len(items) == 2
    assert all(item["notes"] == "split-attribution" for item in items)
    assert pnls == [Decimal("10.00000000"), Decimal("10.00000000")]
    assert sum(pnls) == Decimal("20.00000000")


def test_outcome_attribution_unequal_cost_basis(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    client = TestClient(orchestrator_app.app)
    _risk_and_execution(monkeypatch, avg_price="100000.00", filled_qty="0.001")
    _create_scorecard_order(client, idempotency_key="unequal-a")
    _risk_and_execution(monkeypatch, avg_price="90000.00", filled_qty="0.001")
    _create_scorecard_order(client, idempotency_key="unequal-b")
    _manual_sell(client, idempotency_key="unequal-sell", qty="0.002", price="110000.00")

    items = client.get("/scorecard-outcomes", params={"status": "closed"}).json()["items"]
    by_basis = {item["opened_cost_basis"]: item for item in items}

    assert by_basis["100.00000000"]["closed_realized_pnl"] == "15.78947368"
    assert by_basis["90.00000000"]["closed_realized_pnl"] == "14.21052632"


def test_list_outcomes_filters_by_source_and_status(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _risk_and_execution(monkeypatch)
    client = TestClient(orchestrator_app.app)
    _create_scorecard_order(client, idempotency_key="filter-ta", source="tradingagents")
    _create_scorecard_order(client, idempotency_key="filter-manual", source="manual")

    response = client.get(
        "/scorecard-outcomes", params={"source": "tradingagents", "status": "open"}
    )

    assert response.status_code == 200
    assert len(response.json()["items"]) == 1
    assert response.json()["items"][0]["source"] == "tradingagents"
    assert "attribution_rule" in response.json()


def test_get_outcome_unknown_returns_404(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    response = TestClient(orchestrator_app.app).get(
        "/scorecard-outcomes/99999999-9999-4999-8999-999999999999"
    )
    assert response.status_code == 404
    assert response.json()["code"] == "OUTCOME_NOT_FOUND"


def test_outcomes_summary_hit_rate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    with orchestrator_app.connect() as conn:
        for i, pnl in enumerate(["10", "5", "-3"]):
            conn.execute(
                "insert into scorecard_outcomes "
                "(outcome_id, scorecard_id, actor, symbol, source, action, opened_intent_id, "
                "opened_at, opened_qty, opened_avg_cost, opened_cost_basis, status, "
                "closed_at, closed_realized_pnl, closed_return_pct, notes) "
                "values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)",
                (
                    str(orchestrator_app.uuid4()),
                    f"sc-{i}",
                    "user_1",
                    "BTCUSDT",
                    "tradingagents",
                    "buy",
                    f"intent-{i}",
                    orchestrator_app._now().isoformat(),
                    "0.00100000",
                    "100000.00000000",
                    "100.00000000",
                    "closed",
                    orchestrator_app._now().isoformat(),
                    pnl,
                    "0.10000000",
                ),
            )
        conn.execute(
            "insert into scorecard_outcomes "
            "(outcome_id, scorecard_id, actor, symbol, source, action, opened_intent_id, "
            "opened_at, opened_qty, opened_avg_cost, opened_cost_basis, status) "
            "values (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                str(orchestrator_app.uuid4()),
                "sc-open",
                "user_1",
                "BTCUSDT",
                "tradingagents",
                "buy",
                "intent-open",
                orchestrator_app._now().isoformat(),
                "0.00100000",
                "100000.00000000",
                "100.00000000",
                "open",
            ),
        )
        conn.commit()

    stats = (
        TestClient(orchestrator_app.app)
        .get("/scorecard-outcomes/summary", params={"actor": "user_1"})
        .json()["by_source"]["tradingagents"]
    )

    assert stats["closed_count"] == 3
    assert stats["hits"] == 2
    assert stats["losses"] == 1
    assert stats["open_count"] == 1
    assert stats["hit_rate"] == "0.6667"
    assert stats["realized_pnl"] == "12.00000000"
    assert stats["total_pnl"] == "12.00000000"


def test_outcomes_summary_empty_source(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    response = TestClient(orchestrator_app.app).get("/scorecard-outcomes/summary")
    assert response.status_code == 200
    assert response.json()["by_source"] == {}


def test_outcomes_hook_failure_does_not_break_position_update(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _risk_and_execution(monkeypatch)

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(orchestrator_app, "_maybe_open_scorecard_outcome", fail)
    client = TestClient(orchestrator_app.app)
    response = client.post(
        "/intents/from_scorecard",
        json={
            "scorecard_id": client.post("/scorecards", json=make_scorecard_payload()).json()[
                "scorecard_id"
            ],
            "actor": "user_1",
            "idempotency_key": "hook-fail",
            "usdt_budget": "200",
        },
    )

    assert response.status_code == 200
    with orchestrator_app.connect() as conn:
        position = conn.execute(
            "select qty from paper_positions where actor=?", ("user_1",)
        ).fetchone()
    assert position["qty"] == "0.00100000"


def test_outcome_for_scorecard_deleted_from_table_uses_unknown_source(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _risk_and_execution(monkeypatch)
    client = TestClient(orchestrator_app.app)
    scorecard_id = client.post("/scorecards", json=make_scorecard_payload()).json()["scorecard_id"]
    with orchestrator_app.connect() as conn:
        row = conn.execute(
            "select payload_json from scorecards where scorecard_id = ?", (scorecard_id,)
        ).fetchone()
        conn.execute("delete from scorecards where scorecard_id = ?", (scorecard_id,))
        conn.commit()
    scorecard = orchestrator_app.Scorecard.model_validate_json(row["payload_json"])
    intent = orchestrator_app.OrderIntent(
        intent_id=orchestrator_app.uuid4(),
        request_id=orchestrator_app.uuid4(),
        idempotency_key="deleted-scorecard",
        actor="user_1",
        created_at=orchestrator_app._now(),
        mode="paper",
        venue="binance_spot",
        symbol=scorecard.symbol,
        side="buy",
        order_type="market",
        quantity=orchestrator_app.Quantity(kind="quote", value=Decimal("100")),
        limit_price=None,
        time_in_force="GTC",
        reduce_only=False,
        leverage=None,
        stop_loss=None,
        take_profit=None,
        source=orchestrator_app.Source(
            origin="scorecard", scorecard_id=str(scorecard.scorecard_id), hermes_message_id=None
        ),
        client_confirmation_required=False,
    )

    orchestrator_app._update_position(
        orchestrator_app.ExecutionResult.model_validate(execution()), intent
    )

    item = client.get("/scorecard-outcomes").json()["items"][0]
    assert item["source"] == "unknown"


def test_outcome_close_pushes_reflection_and_sets_reflected_at(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _risk_and_execution(monkeypatch, avg_price="100000.00", filled_qty="0.001")
    reflection_calls: list[dict[str, object]] = []

    def fake_push(outcome: dict[str, object]) -> bool:
        reflection_calls.append(outcome)
        return True

    monkeypatch.setattr(orchestrator_app, "_push_outcome_reflection", fake_push)
    client = TestClient(orchestrator_app.app)
    _create_scorecard_order(client, idempotency_key="reflect-close-buy")
    _manual_sell(client, idempotency_key="reflect-close-sell", qty="0.001", price="110000.00")

    item = client.get("/scorecard-outcomes", params={"status": "closed"}).json()["items"][0]
    assert item["reflected_at"] is not None
    assert reflection_calls[0]["symbol"] == "BTCUSDT"
    assert reflection_calls[0]["closed_return_pct"] == "0.10000000"


def test_outcome_reflection_failure_leaves_reflected_at_null(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _risk_and_execution(monkeypatch, avg_price="100000.00", filled_qty="0.001")
    monkeypatch.setattr(orchestrator_app, "_push_outcome_reflection", lambda outcome: False)
    client = TestClient(orchestrator_app.app)
    _create_scorecard_order(client, idempotency_key="reflect-fail-buy")
    _manual_sell(client, idempotency_key="reflect-fail-sell", qty="0.001", price="110000.00")

    item = client.get("/scorecard-outcomes", params={"status": "closed"}).json()["items"][0]
    assert item["reflected_at"] is None


def test_reflect_pending_endpoint_retries_closed_unreflected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    calls: list[dict[str, object]] = []
    now = orchestrator_app._now().isoformat()
    with orchestrator_app.connect() as conn:
        conn.execute(
            "insert into scorecard_outcomes "
            "(outcome_id, scorecard_id, actor, symbol, source, action, opened_intent_id, "
            "opened_at, opened_qty, opened_avg_cost, opened_cost_basis, status, "
            "closed_at, closed_realized_pnl, closed_return_pct, notes, reflected_at) "
            "values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)",
            (
                str(orchestrator_app.uuid4()),
                "sc-rp",
                "user_1",
                "BTCUSDT",
                "tradingagents",
                "buy",
                "intent-rp",
                now,
                "0.00100000",
                "100000.00000000",
                "100.00000000",
                "closed",
                now,
                "10.00000000",
                "0.10000000",
                None,
            ),
        )
        conn.commit()

    def fake_push(outcome: dict[str, object]) -> bool:
        calls.append(outcome)
        return True

    monkeypatch.setattr(orchestrator_app, "_push_outcome_reflection", fake_push)
    response = TestClient(orchestrator_app.app).post("/reflect/pending", params={"limit": 25})

    assert response.status_code == 200
    body = response.json()
    assert body["attempted"] == 1
    assert body["reflected"] == 1
    assert body["failed"] == 0
    assert calls[0]["scorecard_id"] == "sc-rp"
    item = (
        TestClient(orchestrator_app.app)
        .get("/scorecard-outcomes", params={"status": "closed"})
        .json()["items"][0]
    )
    assert item["reflected_at"] is not None


def test_outcomes_summary_includes_pending_reflection_count(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    now = orchestrator_app._now().isoformat()
    with orchestrator_app.connect() as conn:
        for i, reflected in enumerate([None, now]):
            conn.execute(
                "insert into scorecard_outcomes "
                "(outcome_id, scorecard_id, actor, symbol, source, action, opened_intent_id, "
                "opened_at, opened_qty, opened_avg_cost, opened_cost_basis, status, "
                "closed_at, closed_realized_pnl, closed_return_pct, notes, reflected_at) "
                "values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(orchestrator_app.uuid4()),
                    f"sc-pr-{i}",
                    "user_1",
                    "BTCUSDT",
                    "tradingagents",
                    "buy",
                    f"intent-pr-{i}",
                    now,
                    "0.00100000",
                    "100000.00000000",
                    "100.00000000",
                    "closed",
                    now,
                    "5.00000000",
                    "0.05000000",
                    None,
                    reflected,
                ),
            )
        conn.commit()

    stats = (
        TestClient(orchestrator_app.app)
        .get("/scorecard-outcomes/summary", params={"actor": "user_1"})
        .json()["by_source"]["tradingagents"]
    )
    assert stats["pending_reflection_count"] == 1


def _insert_scorecard_for_reflection(tmp_path: Path, metadata: dict[str, str]) -> str:
    import uuid

    _ = tmp_path
    scorecard_id = str(uuid.uuid4())
    with orchestrator_app.connect() as conn:
        conn.execute(
            "insert into scorecards "
            "(scorecard_id, actor, symbol, action, source, payload_json, created_at, expires_at) "
            "values (?,?,?,?,?,?,?,?)",
            (
                scorecard_id,
                "tg_1",
                "ETHUSDT",
                "buy",
                "tradingagents",
                json.dumps({"metadata": metadata}),
                "2026-05-25T00:00:00+00:00",
                "2026-05-25T01:00:00+00:00",
            ),
        )
        conn.commit()
    return scorecard_id


def test_reflection_alpha_subtracts_benchmark_return(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    scorecard_id = _insert_scorecard_for_reflection(
        tmp_path,
        {
            "ta_ticker": "ETH-USD",
            "ta_date": "2026-05-25",
            "benchmark_symbol": "BTCUSDT",
            "benchmark_open_price": "100.00",
        },
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        orchestrator_app, "_mark_for_symbol_str", lambda symbol: (Decimal("110.00"), "test")
    )

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured.update(kwargs.get("json") or {})
        return FakeResponse({"ok": True, "reflected": True})

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    assert (
        orchestrator_app._push_outcome_reflection(
            {
                "scorecard_id": scorecard_id,
                "symbol": "ETHUSDT",
                "opened_at": "2026-05-25T00:00:00+00:00",
                "closed_at": "2026-05-26T00:00:00+00:00",
                "closed_return_pct": "0.25000000",
            }
        )
        is True
    )
    assert captured["raw_return"] == "0.25000000"
    assert captured["alpha_return"] == "0.15000000"
    assert captured["benchmark_name"] == "BTCUSDT"
    assert "alpha_note" not in captured


def test_reflection_alpha_falls_back_for_self_benchmark(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    scorecard_id = _insert_scorecard_for_reflection(
        tmp_path,
        {"benchmark_symbol": "self", "benchmark_open_price": "100.00"},
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        orchestrator_app,
        "_mark_for_symbol_str",
        lambda symbol: (_ for _ in ()).throw(AssertionError("should not fetch self benchmark")),
    )

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured.update(kwargs.get("json") or {})
        return FakeResponse({"ok": True, "reflected": True})

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    assert (
        orchestrator_app._push_outcome_reflection(
            {"scorecard_id": scorecard_id, "symbol": "BTCUSDT", "closed_return_pct": "0.07000000"}
        )
        is True
    )
    assert captured["alpha_return"] == "0.07000000"
    assert "alpha_note" in captured


def test_watchlist_crud(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    client = TestClient(orchestrator_app.app)

    created = client.post(
        "/watchlist",
        json={"actor": "tg_1", "symbol": "ETHUSDT", "asset_type": "crypto", "cadence_minutes": 30},
    )
    assert created.status_code == 200
    assert created.json()["item"]["symbol"] == "ETHUSDT"

    listed = client.get("/watchlist", params={"actor": "tg_1"})
    assert listed.status_code == 200
    assert listed.json()["items"][0]["cadence_minutes"] == 30

    deleted = client.delete("/watchlist/ETHUSDT", params={"actor": "tg_1"})
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True}
    assert client.get("/watchlist", params={"actor": "tg_1"}).json()["items"] == []


def test_scheduler_tick_fires_due_watchlist_and_reschedules(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    now = datetime(2026, 5, 25, 0, 0, tzinfo=UTC)
    with orchestrator_app.connect() as conn:
        conn.execute(
            "insert into watchlist_entries "
            "(actor, symbol, asset_type, cadence_minutes, last_run_at, "
            "next_run_at, enabled, created_at) "
            "values (?,?,?,?,NULL,?,?,?)",
            (
                "tg_1",
                "ETHUSDT",
                "crypto",
                15,
                "2026-05-24T23:59:00+00:00",
                1,
                "2026-05-24T00:00:00+00:00",
            ),
        )
        conn.commit()
    fired: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        orchestrator_app,
        "_fire_scheduled_analysis",
        lambda actor, symbol, asset_type: fired.append((actor, symbol, asset_type)) or True,
    )

    assert orchestrator_app.scheduler_tick(now=now) == {"due": 1, "fired": 1, "failed": 0}
    assert fired == [("tg_1", "ETHUSDT", "crypto")]
    with orchestrator_app.connect() as conn:
        row = conn.execute(
            "select last_run_at, next_run_at from watchlist_entries where actor = ?", ("tg_1",)
        ).fetchone()
    assert row["last_run_at"] == now.isoformat()
    assert row["next_run_at"] == (now + timedelta(minutes=15)).isoformat()


def _seed_calibration_outcome(
    *,
    actor: str = "tg_1",
    source: str = "tradingagents",
    asset_type: str = "crypto",
    heuristic: str = "0.7000",
    alpha: str = "0.01000000",
) -> None:
    import uuid

    scorecard_id = str(uuid.uuid4())
    outcome_id = str(uuid.uuid4())
    with orchestrator_app.connect() as conn:
        conn.execute(
            "insert into scorecards "
            "(scorecard_id, actor, symbol, action, source, payload_json, created_at, expires_at) "
            "values (?,?,?,?,?,?,?,?)",
            (
                scorecard_id,
                actor,
                "ETHUSDT",
                "buy",
                source,
                json.dumps(
                    {
                        "conviction": heuristic,
                        "metadata": {
                            "heuristic_conviction": heuristic,
                            "asset_type": asset_type,
                        },
                    }
                ),
                "2026-05-25T00:00:00+00:00",
                "2026-05-26T00:00:00+00:00",
            ),
        )
        conn.execute(
            "insert into scorecard_outcomes "
            "(outcome_id, scorecard_id, actor, symbol, source, action, opened_intent_id, "
            "opened_at, opened_qty, opened_avg_cost, opened_cost_basis, status, closed_at, "
            "closed_realized_pnl, closed_return_pct, notes, reflected_at) "
            "values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                outcome_id,
                scorecard_id,
                actor,
                "ETHUSDT",
                source,
                "buy",
                str(uuid.uuid4()),
                "2026-05-25T00:00:00+00:00",
                "1.00000000",
                "100.00000000",
                "100.00000000",
                "closed",
                "2026-05-26T00:00:00+00:00",
                "1.00000000",
                alpha,
                None,
                "2026-05-26T00:01:00+00:00",
            ),
        )
        conn.commit()


def _seed_autonomy_scorecard(
    *,
    actor: str = "tg_1",
    source: str = "tradingagents",
    action: str = "buy",
    conviction: str = "0.8000",
    consumed: bool = False,
    expires_at: str = "2026-05-26T00:00:00+00:00",
) -> str:
    import uuid

    scorecard_id = str(uuid.uuid4())
    with orchestrator_app.connect() as conn:
        conn.execute(
            "insert into scorecards "
            "(scorecard_id, actor, symbol, action, source, payload_json, created_at, "
            "expires_at, consumed_by_intent_id) values (?,?,?,?,?,?,?,?,?)",
            (
                scorecard_id,
                actor,
                "ETHUSDT",
                action,
                source,
                json.dumps({"conviction": conviction}),
                "2026-05-25T00:00:00+00:00",
                expires_at,
                "intent-used" if consumed else None,
            ),
        )
        conn.commit()
    return scorecard_id


def test_recompute_with_no_outcomes_returns_zero_buckets(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    client = TestClient(orchestrator_app.app)
    response = client.post("/calibration/recompute")
    assert response.status_code == 200
    assert response.json() == {"buckets_written": 0, "rows_considered": 0}


def test_recompute_bucketing_correct(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for _ in range(12):
        _seed_calibration_outcome(heuristic="0.7200", alpha="0.01000000")
    for _ in range(8):
        _seed_calibration_outcome(heuristic="0.6200", alpha="-0.01000000")
    client = TestClient(orchestrator_app.app)
    assert client.post("/calibration/recompute").json()["buckets_written"] == 2
    items = client.get("/calibration", params={"source": "tradingagents"}).json()["items"]
    by_bucket = {item["heuristic_bucket"]: item for item in items}
    assert by_bucket["0.70-0.80"]["sample_count"] == 12
    assert by_bucket["0.70-0.80"]["hit_count"] == 12
    assert by_bucket["0.60-0.70"]["sample_count"] == 8
    assert by_bucket["0.60-0.70"]["hit_count"] == 0


def test_recompute_shrinkage_pulls_low_sample_toward_half(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for _ in range(2):
        _seed_calibration_outcome(heuristic="0.8200", alpha="0.01000000")
    client = TestClient(orchestrator_app.app)
    client.post("/calibration/recompute")
    item = client.get("/calibration").json()["items"][0]
    assert item["empirical_hit_rate"] == "1.00000000"
    assert Decimal(item["calibrated_conviction"]) < Decimal("1")
    assert Decimal(item["calibrated_conviction"]) > Decimal("0.5")


def test_recompute_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _seed_calibration_outcome(heuristic="0.7200", alpha="0.01000000")
    client = TestClient(orchestrator_app.app)
    client.post("/calibration/recompute")
    first = [
        {k: v for k, v in item.items() if k != "updated_at"}
        for item in client.get("/calibration").json()["items"]
    ]
    client.post("/calibration/recompute")
    second = [
        {k: v for k, v in item.items() if k != "updated_at"}
        for item in client.get("/calibration").json()["items"]
    ]
    assert second == first


def test_autonomy_settings_default_disabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    body = (
        TestClient(orchestrator_app.app).get("/autonomy/settings", params={"actor": "tg_1"}).json()
    )
    assert body["enabled"] is False
    assert body["daily_budget_usdt"] == "0"


def test_autonomy_settings_update_and_validation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    client = TestClient(orchestrator_app.app)
    assert (
        client.post("/autonomy/settings", json={"actor": "tg_1", "daily_budget_usdt": "-1"}).json()[
            "code"
        ]
        == "INVALID_BUDGET"
    )
    assert (
        client.post("/autonomy/settings", json={"actor": "tg_1", "min_conviction": "1.5"}).json()[
            "code"
        ]
        == "INVALID_MIN_CONVICTION"
    )
    updated = client.post(
        "/autonomy/settings",
        json={"actor": "tg_1", "enabled": True, "daily_budget_usdt": "100", "per_trade_usdt": "50"},
    ).json()
    assert updated["enabled"] is True
    assert (
        client.get("/autonomy/settings", params={"actor": "tg_1"}).json()["daily_budget_usdt"]
        == "100"
    )


def test_auto_trade_tick_places_paper_trade_and_records_spend(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    now = datetime(2026, 5, 25, 0, 0, tzinfo=UTC)
    scorecard_id = _seed_autonomy_scorecard()
    TestClient(orchestrator_app.app).post(
        "/autonomy/settings",
        json={"actor": "tg_1", "enabled": True, "daily_budget_usdt": "100", "per_trade_usdt": "50"},
    )
    captured: list[dict[str, object]] = []

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured.append(kwargs.get("json") or {})
        return FakeResponse({"ok": True})

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    assert orchestrator_app.auto_trade_tick(now=now) == {
        "placed": 1,
        "skipped_budget": 0,
        "skipped_other": 0,
    }
    assert captured[0]["scorecard_id"] == scorecard_id
    assert captured[0]["mode"] == "paper"
    with orchestrator_app.connect() as conn:
        spend = conn.execute(
            "select spent_usdt from autonomy_spend where actor = ? and date = ?",
            ("tg_1", "2026-05-25"),
        ).fetchone()
    assert spend["spent_usdt"] == "50.00000000"


def test_auto_trade_tick_filters_and_budget(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    now = datetime(2026, 5, 25, 0, 0, tzinfo=UTC)
    _seed_autonomy_scorecard(conviction="0.6000")
    _seed_autonomy_scorecard(source="manual")
    _seed_autonomy_scorecard(action="hold")
    _seed_autonomy_scorecard(consumed=True)
    _seed_autonomy_scorecard(expires_at="2026-05-24T00:00:00+00:00")
    _seed_autonomy_scorecard(conviction="0.8000")
    TestClient(orchestrator_app.app).post(
        "/autonomy/settings",
        json={
            "actor": "tg_1",
            "enabled": True,
            "daily_budget_usdt": "50",
            "per_trade_usdt": "50",
            "min_conviction": "0.7",
        },
    )
    calls = []
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda *args, **kwargs: (
            calls.append(kwargs.get("json") or {}) or FakeResponse({"ok": True})
        ),
    )
    result = orchestrator_app.auto_trade_tick(now=now)
    assert result["placed"] == 1
    assert len(calls) == 1
    assert calls[0]["mode"] == "paper"


def test_auto_trade_tick_fail_open_on_http_error(monkeypatch, tmp_path: Path) -> None:
    import httpx as _httpx

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _seed_autonomy_scorecard()
    TestClient(orchestrator_app.app).post(
        "/autonomy/settings",
        json={"actor": "tg_1", "enabled": True, "daily_budget_usdt": "100", "per_trade_usdt": "50"},
    )
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(_httpx.HTTPError("down")),
    )
    result = orchestrator_app.auto_trade_tick(now=datetime(2026, 5, 25, 0, 0, tzinfo=UTC))
    assert result["placed"] == 0
    assert (
        TestClient(orchestrator_app.app)
        .get("/autonomy/today", params={"actor": "tg_1"})
        .json()["spent_usdt"]
        == "0"
    )


def test_live_auto_global_disabled_blocks(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "LIVE_AUTONOMY_GLOBAL_ENABLED", False)
    payload = {
        "source": "tradingagents",
        "conviction": "0.7500",
        "metadata": {"asset_type": "crypto", "heuristic_conviction": "0.7200"},
    }
    settings = {
        "enabled": True,
        "daily_live_budget_usdt": "100",
        "daily_live_trade_count_max": 3,
        "min_calibrated_conviction": "0.70",
        "min_closed_outcomes": 20,
        "allowed_sources": "tradingagents",
    }
    assert orchestrator_app._eligible_for_live_auto("tg_1", payload, settings) == (
        False,
        "LIVE_AUTONOMY_GLOBAL_DISABLED",
    )
    assert orchestrator_app.live_auto_trade_tick() == {
        "placed": 0,
        "skipped": 0,
        "reason": "GLOBAL_DISABLED",
    }


def test_live_auto_no_calibration_blocks(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "LIVE_AUTONOMY_GLOBAL_ENABLED", True)
    payload = {
        "source": "tradingagents",
        "conviction": "0.7500",
        "metadata": {"asset_type": "crypto", "heuristic_conviction": "0.7200"},
    }
    settings = {
        "enabled": True,
        "daily_live_budget_usdt": "100",
        "daily_live_trade_count_max": 3,
        "min_calibrated_conviction": "0.70",
        "min_closed_outcomes": 20,
        "allowed_sources": "tradingagents",
    }
    assert orchestrator_app._eligible_for_live_auto("tg_1", payload, settings) == (
        False,
        "NO_CALIBRATION_DATA",
    )


def test_live_autonomy_settings_default_and_validation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    client = TestClient(orchestrator_app.app)
    assert (
        client.get("/live-autonomy/settings", params={"actor": "tg_1"}).json()["enabled"] is False
    )
    assert (
        client.post(
            "/live-autonomy/settings", json={"actor": "tg_1", "min_calibrated_conviction": "0.1"}
        ).json()["code"]
        == "INVALID_MIN_CONVICTION"
    )
    assert (
        client.post(
            "/live-autonomy/settings", json={"actor": "tg_1", "per_live_trade_max_usdt": "501"}
        ).json()["code"]
        == "INVALID_PER_TRADE"
    )
    assert (
        client.post(
            "/live-autonomy/settings",
            json={"actor": "tg_1", "enabled": True, "daily_live_budget_usdt": "100"},
        ).json()["enabled"]
        is True
    )


def test_auto_unlock_token_single_use_actor_and_expiry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    token = orchestrator_app._mint_user_live_unlock_token("tg_1")
    assert (
        orchestrator_app._consume_live_unlock_or_error(token, "tg_2", dry=True).status_code == 403
    )
    assert orchestrator_app._consume_live_unlock_or_error(token, "tg_1", dry=False) is None
    assert (
        orchestrator_app._consume_live_unlock_or_error(token, "tg_1", dry=False).status_code == 403
    )
    with orchestrator_app.connect() as conn:
        expired = orchestrator_app._mint_user_live_unlock_token("tg_1")
        conn.execute(
            "update live_unlock_tokens set expires_at = ? where token = ?",
            ("2000-01-01T00:00:00+00:00", expired),
        )
        conn.commit()
    assert (
        orchestrator_app._consume_live_unlock_or_error(expired, "tg_1", dry=True).status_code == 410
    )


def test_auto_unlock_token_bound_to_intent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    intent_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    other_intent_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"

    token = orchestrator_app._mint_auto_unlock_bound_token("tg_1", intent_id)

    mismatch = orchestrator_app._consume_live_unlock_or_error(
        token, "tg_1", dry=True, intent_id=orchestrator_app.UUID(other_intent_id)
    )
    assert mismatch.status_code == 403
    assert mismatch.body == b'{"code":"LIVE_UNLOCK_INTENT_MISMATCH"}'
    assert (
        orchestrator_app._consume_live_unlock_or_error(
            token, "tg_1", dry=False, intent_id=orchestrator_app.UUID(intent_id)
        )
        is None
    )


def test_create_intent_from_scorecard_respects_requested_intent_id(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    client = TestClient(orchestrator_app.app)
    scorecard_id = client.post("/scorecards", json=make_scorecard_payload(actor="tg_1")).json()[
        "scorecard_id"
    ]
    requested_intent = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"

    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda url, **kw: FakeResponse(
            decision(intent_id=requested_intent)
            if url.endswith("/validate")
            else execution(intent_id=requested_intent)
        ),
    )
    response = client.post(
        "/intents/from_scorecard",
        json={
            "scorecard_id": scorecard_id,
            "actor": "tg_1",
            "idempotency_key": "scorecard-bound-intent",
            "mode": "paper",
            "usdt_budget": "100",
            "intent_id": requested_intent,
        },
    )
    assert response.status_code == 200
    assert response.json()["intent"]["intent_id"] == requested_intent


def test_live_auto_mints_intent_bound_unlock(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["json"] = kwargs["json"]
        captured["headers"] = kwargs["headers"]
        return FakeResponse({"status": "executed"})

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)

    assert orchestrator_app._fire_live_autonomous_trade("tg_1", "scorecard-abc", Decimal("25"))
    payload = captured["json"]
    headers = captured["headers"]
    assert isinstance(payload, dict)
    assert isinstance(headers, dict)
    assert payload["mode"] == "live"
    assert payload["intent_id"]
    token = headers["x-live-unlock"]
    with orchestrator_app.connect() as conn:
        row = conn.execute(
            "select actor, bound_intent_id from live_unlock_tokens where token = ?",
            (token,),
        ).fetchone()
    assert row["actor"] == "tg_1"
    assert row["bound_intent_id"] == payload["intent_id"]


def test_live_exposure_cap_defaults_closed_and_blocks_breach(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "LIVE_AUTONOMY_GLOBAL_ENABLED", True)
    settings = {
        "enabled": True,
        "allowed_sources": "tradingagents",
        "min_calibrated_conviction": "0.70",
        "min_closed_outcomes": 5,
        "daily_live_budget_usdt": "1000",
        "per_live_trade_max_usdt": "25",
        "daily_live_trade_count_max": 3,
    }
    scorecard = {
        "source": "tradingagents",
        "conviction": "0.9",
        "metadata": {"asset_type": "crypto", "heuristic_conviction": "0.9"},
    }
    with orchestrator_app.connect() as conn:
        conn.execute(
            "insert into conviction_calibration values (?,?,?,?,?,?,?,?,?)",
            ("tradingagents", "crypto", "0.90-1.01", 5, 5, "0", "1", "0.9", "now"),
        )
        conn.commit()

    allowed, reason = orchestrator_app._eligible_for_live_auto("tg_1", scorecard, settings)
    assert allowed is False
    assert reason == "MAX_LIVE_EXPOSURE_NOT_SET"

    settings["max_live_exposure_usdt"] = "100"
    with orchestrator_app.connect() as conn:
        conn.execute(
            "insert into paper_positions "
            "(actor, symbol, qty, avg_cost, total_cost, realized_pnl, "
            "paper_qty, paper_avg_cost, live_qty, live_avg_cost, last_updated) "
            "values (?,?,?,?,?,?,?,?,?,?,?)",
            ("tg_1", "BTCUSDT", "0.001", "90000", "90", "0", "0", "0", "0.001", "90000", "now"),
        )
        conn.execute(
            "insert into intents (intent_id, payload_json, created_at, status, idempotency_key) "
            "values (?,?,?,?,?)",
            (
                "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                json.dumps({"actor": "tg_1", "symbol": "BTCUSDT", "mode": "live"}),
                "now",
                "executed",
                "live-exposure-existing",
            ),
        )
        conn.commit()
    allowed, reason = orchestrator_app._eligible_for_live_auto("tg_1", scorecard, settings)
    assert allowed is False
    assert reason == "LIVE_EXPOSURE_CAP_BREACHED:90.00000000/100"


def test_live_autonomy_settings_include_exposure_cap(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    client = TestClient(orchestrator_app.app)
    assert (
        client.post(
            "/live-autonomy/settings",
            json={"actor": "tg_1", "max_live_exposure_usdt": "-1"},
        ).json()["code"]
        == "INVALID_MAX_EXPOSURE"
    )
    saved = client.post(
        "/live-autonomy/settings",
        json={"actor": "tg_1", "max_live_exposure_usdt": "250"},
    ).json()
    assert saved["max_live_exposure_usdt"] == "250"
    loaded = client.get("/live-autonomy/settings", params={"actor": "tg_1"}).json()
    assert loaded["max_live_exposure_usdt"] == "250"
    assert loaded["current_live_exposure_usdt"] == "0"


def test_notification_subscribe_sends_hmac_fill_and_records_delivery(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app, "NOTIFICATION_HOST_ALLOWLIST", frozenset({"hermes-agent"})
    )
    client = TestClient(orchestrator_app.app)
    subscribed = client.post(
        "/notifications/subscribe",
        json={
            "actor": "tg_1",
            "webhook_url": "http://hermes-agent:9090/trading/fill",
            "events": ["fill"],
        },
    )
    assert subscribed.status_code == 200
    secret = subscribed.json()["secret"]
    calls: list[dict[str, object]] = []

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url == "http://hermes-agent:9090/trading/fill":
            calls.append(kwargs)
            return FakeResponse({"ok": True}, status_code=204)
        if url.endswith("/validate"):
            return FakeResponse(decision())
        return FakeResponse(execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    created = client.post("/intents", json={**VALID, "actor": "tg_1", "idempotency_key": "fill-1"})
    assert created.status_code == 200
    assert len(calls) == 1
    sent = calls[0]
    body = sent["content"]
    headers = sent["headers"]
    assert isinstance(body, str)
    assert isinstance(headers, dict)
    payload = json.loads(body)
    assert set(payload) == {
        "event_type",
        "actor",
        "symbol",
        "side",
        "qty_str",
        "avg_price_str",
        "mode",
        "status",
        "intent_id",
    }
    expected = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
    assert headers["x-trading-agent-signature"] == expected
    deliveries = client.get("/notifications/deliveries", params={"actor": "tg_1"}).json()
    assert deliveries["deliveries"][0]["ok"] is True
    assert deliveries["deliveries"][0]["status_code"] == 204


def test_position_updates_keep_paper_and_live_buckets_separate(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    paper_intent = orchestrator_app.OrderIntent.model_validate(VALID)
    live_intent = orchestrator_app.OrderIntent.model_validate(
        {
            **VALID,
            "intent_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "idempotency_key": "live-fill-1",
            "mode": "live",
        }
    )

    orchestrator_app._update_position(
        orchestrator_app.ExecutionResult.model_validate(
            execution(filled_qty="0.002", avg_price="100000.00")
        ),
        paper_intent,
    )
    orchestrator_app._update_position(
        orchestrator_app.ExecutionResult.model_validate(
            execution(
                intent_id=live_intent.intent_id,
                filled_qty="0.001",
                avg_price="90000.00",
                execution_id="55555555-5555-4555-8555-555555555555",
            )
        ),
        live_intent,
    )

    with orchestrator_app.connect() as conn:
        row = conn.execute(
            "select qty, avg_cost, paper_qty, paper_avg_cost, live_qty, live_avg_cost "
            "from paper_positions where actor = ? and symbol = ?",
            ("user_1", "BTCUSDT"),
        ).fetchone()
    assert row["paper_qty"] == "0.00200000"
    assert row["paper_avg_cost"] == "100000.00000000"
    assert row["live_qty"] == "0.00100000"
    assert row["live_avg_cost"] == "90000.00000000"
    assert row["qty"] == "0.00300000"
    assert row["avg_cost"] == "96666.66666667"


def test_legacy_position_qty_backfills_paper_bucket(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db_file = tmp_path / "trading.sqlite"
    conn = sqlite3.connect(db_file)
    conn.execute(
        """
        create table paper_positions (
            actor text not null,
            symbol text not null,
            qty text not null default '0',
            avg_cost text not null default '0',
            total_cost text not null default '0',
            realized_pnl text not null default '0',
            last_updated text not null,
            primary key (actor, symbol)
        )
        """
    )
    conn.execute(
        "insert into paper_positions values (?,?,?,?,?,?,?)",
        ("tg_1", "BTCUSDT", "0.004", "50000", "200", "0", "legacy"),
    )
    conn.commit()
    conn.close()

    with orchestrator_app.connect() as migrated:
        row = migrated.execute(
            "select paper_qty, paper_avg_cost, live_qty, live_avg_cost "
            "from paper_positions where actor = ? and symbol = ?",
            ("tg_1", "BTCUSDT"),
        ).fetchone()
    assert row["paper_qty"] == "0.004"
    assert row["paper_avg_cost"] == "50000"
    assert row["live_qty"] == "0"
    assert row["live_avg_cost"] == "0"


def test_live_exposure_cap_uses_live_bucket_only(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    with orchestrator_app.connect() as conn:
        conn.execute(
            """
            insert into paper_positions
              (actor, symbol, qty, avg_cost, total_cost, realized_pnl,
               paper_qty, paper_avg_cost, live_qty, live_avg_cost, last_updated)
            values (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "tg_1",
                "BTCUSDT",
                "0.003",
                "96666.66666667",
                "290",
                "0",
                "0.002",
                "100000",
                "0.001",
                "90000",
                "now",
            ),
        )
        conn.commit()

    allowed, current = orchestrator_app._check_live_exposure_cap(
        "tg_1", Decimal("5"), Decimal("100")
    )

    assert allowed is True
    assert current == Decimal("90.00000000")


def test_stop_loss_watchdog_fires_paper_protective_sell(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "STOP_LOSS_WATCHDOG_ENABLED", True)
    monkeypatch.setattr(
        orchestrator_app,
        "_mark_for_symbol",
        lambda symbol: (Decimal("89000"), "test"),
    )
    calls: list[dict[str, object]] = []

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse({"status": "executed"})

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    _seed_open_outcome_with_position(
        actor="tg_1",
        mode="paper",
        stop_loss="90000",
        take_profit="120000",
    )

    result = orchestrator_app.stop_loss_watchdog_tick()

    assert result == {"checked": 1, "fired": 1, "skipped": 0, "failed": 0}
    assert len(calls) == 1
    sent = calls[0]
    assert str(sent["url"]).endswith("/intents")
    payload = sent["json"]
    assert isinstance(payload, dict)
    assert payload["actor"] == "tg_1"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["side"] == "sell"
    assert payload["mode"] == "paper"
    assert payload["quantity"] == {"kind": "base", "value": "0.00200000"}
    assert "x-live-unlock" not in sent.get("headers", {})


def test_stop_loss_watchdog_mints_bound_token_for_live_sell(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "STOP_LOSS_WATCHDOG_ENABLED", True)
    monkeypatch.setattr(
        orchestrator_app,
        "_mark_for_symbol",
        lambda symbol: (Decimal("121000"), "test"),
    )
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured.update({"url": url, **kwargs})
        return FakeResponse({"status": "executed"})

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    _seed_open_outcome_with_position(
        actor="tg_1",
        mode="live",
        stop_loss="90000",
        take_profit="120000",
    )

    result = orchestrator_app.stop_loss_watchdog_tick()

    assert result["fired"] == 1
    payload = captured["json"]
    headers = captured["headers"]
    assert isinstance(payload, dict)
    assert isinstance(headers, dict)
    assert payload["mode"] == "live"
    token = headers["x-live-unlock"]
    with orchestrator_app.connect() as conn:
        row = conn.execute(
            "select actor, bound_intent_id from live_unlock_tokens where token = ?",
            (token,),
        ).fetchone()
    assert row["actor"] == "tg_1"
    assert row["bound_intent_id"] == payload["intent_id"]


def test_stop_loss_watchdog_never_raises_per_row(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "STOP_LOSS_WATCHDOG_ENABLED", True)

    def fail_mark(symbol: str) -> tuple[Decimal | None, str | None]:
        raise RuntimeError("market down")

    monkeypatch.setattr(orchestrator_app, "_mark_for_symbol", fail_mark)
    _seed_open_outcome_with_position(
        actor="tg_1",
        mode="paper",
        stop_loss="90000",
        take_profit="120000",
    )

    result = orchestrator_app.stop_loss_watchdog_tick()

    assert result == {"checked": 1, "fired": 0, "skipped": 0, "failed": 1}


def _seed_open_outcome_with_position(
    *,
    actor: str,
    mode: str,
    stop_loss: str,
    take_profit: str,
) -> None:
    now = datetime.now(UTC).isoformat()
    scorecard_id = f"scorecard-{mode}"
    opened_intent_id = f"11111111-1111-4111-8111-00000000000{1 if mode == 'paper' else 2}"
    paper_qty = "0.00200000" if mode == "paper" else "0"
    live_qty = "0.00200000" if mode == "live" else "0"
    with orchestrator_app.connect() as conn:
        conn.execute(
            """
            insert into scorecards
              (scorecard_id, actor, symbol, action, source, payload_json,
               created_at, expires_at, consumed_by_intent_id)
            values (?,?,?,?,?,?,?,?,?)
            """,
            (
                scorecard_id,
                actor,
                "BTCUSDT",
                "buy",
                "tradingagents",
                json.dumps(
                    {
                        "symbol": "BTCUSDT",
                        "action": "buy",
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                    }
                ),
                now,
                (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                opened_intent_id,
            ),
        )
        conn.execute(
            """
            insert into intents (intent_id, payload_json, created_at, status, idempotency_key)
            values (?,?,?,?,?)
            """,
            (
                opened_intent_id,
                json.dumps({"actor": actor, "symbol": "BTCUSDT", "mode": mode}),
                now,
                "executed",
                f"open-{mode}",
            ),
        )
        conn.execute(
            """
            insert into paper_positions
              (actor, symbol, qty, avg_cost, total_cost, realized_pnl,
               paper_qty, paper_avg_cost, live_qty, live_avg_cost, last_updated)
            values (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                actor,
                "BTCUSDT",
                "0.00200000",
                "100000.00000000",
                "200.00000000",
                "0",
                paper_qty,
                "100000.00000000" if mode == "paper" else "0",
                live_qty,
                "100000.00000000" if mode == "live" else "0",
                now,
            ),
        )
        conn.execute(
            """
            insert into scorecard_outcomes
              (outcome_id, scorecard_id, actor, symbol, source, action,
               opened_intent_id, opened_at, opened_qty, opened_avg_cost,
               opened_cost_basis, status, closed_at, closed_realized_pnl,
               closed_return_pct, notes)
            values (?,?,?,?,?,?,?,?,?,?,?,'open',NULL,NULL,NULL,NULL)
            """,
            (
                f"outcome-{mode}",
                scorecard_id,
                actor,
                "BTCUSDT",
                "tradingagents",
                "buy",
                opened_intent_id,
                now,
                "0.00200000",
                "100000.00000000",
                "200.00000000",
            ),
        )
        conn.commit()


def test_notification_rejects_disallowed_host_and_prunes_history(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "NOTIFICATION_HOST_ALLOWLIST", frozenset({"allowed"}))
    monkeypatch.setattr(orchestrator_app, "NOTIFICATION_HISTORY_LIMIT", 2)
    client = TestClient(orchestrator_app.app)
    rejected = client.post(
        "/notifications/subscribe",
        json={"actor": "tg_1", "webhook_url": "http://evil.example/hook", "events": ["fill"]},
    )
    assert rejected.status_code == 400
    assert rejected.json()["code"] == "WEBHOOK_HOST_NOT_ALLOWED"

    assert (
        client.post(
            "/notifications/subscribe",
            json={"actor": "tg_1", "webhook_url": "http://allowed/hook", "events": ["fill"]},
        ).status_code
        == 200
    )

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url == "http://allowed/hook":
            return FakeResponse({"ok": True}, status_code=500)
        if url.endswith("/validate"):
            return FakeResponse(decision())
        return FakeResponse(execution())

    monkeypatch.setattr(orchestrator_app.httpx, "post", fake_post)
    for index in range(3):
        body = {
            **VALID,
            "intent_id": f"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa{index}",
            "actor": "tg_1",
            "idempotency_key": f"fill-prune-{index}",
        }
        response = client.post("/intents", json=body)
        assert response.status_code == 200
    deliveries = client.get("/notifications/deliveries", params={"actor": "tg_1"}).json()
    assert len(deliveries["deliveries"]) == 2
    assert deliveries["deliveries"][0]["ok"] is False


def test_disable_and_enable_live_autonomy_kill_switch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orchestrator_app, "OPS_TOKEN", "ops")
    monkeypatch.setattr(orchestrator_app, "LIVE_AUTONOMY_GLOBAL_ENABLED", True)
    client = TestClient(orchestrator_app.app)
    assert (
        client.post("/admin/live-autonomy/disable", headers={"x-ops-token": "bad"}).status_code
        == 403
    )
    assert (
        client.post("/admin/live-autonomy/disable", headers={"x-ops-token": "ops"}).json()["killed"]
        is True
    )
    assert orchestrator_app.LIVE_AUTONOMY_GLOBAL_ENABLED is False
    assert (
        client.post("/admin/live-autonomy/enable", headers={"x-ops-token": "ops"}).json()["killed"]
        is False
    )
    assert orchestrator_app.LIVE_AUTONOMY_GLOBAL_ENABLED is False
