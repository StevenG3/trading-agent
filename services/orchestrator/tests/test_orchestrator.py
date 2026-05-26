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
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

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

    with_tok = client.post(
        "/intents/from_nl", json=payload, headers={"x-live-unlock": token}
    )
    assert with_tok.status_code == 200


def test_from_scorecard_consume_is_conditional_update(monkeypatch, tmp_path: Path) -> None:
    """Steady-state consumed scorecards are rejected; final consume uses rowcount."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        orchestrator_app.httpx,
        "post",
        lambda url, **k: (
            FakeResponse(decision())
            if url.endswith("/validate")
            else FakeResponse(execution())
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
            FakeResponse(decision())
            if url.endswith("/validate")
            else FakeResponse(execution())
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

