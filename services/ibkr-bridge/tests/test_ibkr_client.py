from __future__ import annotations

import logging
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibkr_client import IBKRClient, IBKRConfig, PlaceOrderRequest, _is_live_gateway_port


class FakeIB:
    def __init__(self) -> None:
        self.connected = False
        self.qualified: list[object] = []
        self.placed: list[tuple[object, object]] = []
        self.disconnected = False

    def connect(
        self, host: str, port: int, clientId: int, timeout: float
    ) -> bool:  # noqa: N803
        _ = host, port, clientId, timeout
        self.connected = True
        return True

    def isConnected(self) -> bool:  # noqa: N802
        return self.connected

    def disconnect(self) -> None:
        self.disconnected = True
        self.connected = False

    def qualifyContracts(self, contract: object) -> list[object]:  # noqa: N802
        self.qualified.append(contract)
        return [contract]

    def placeOrder(self, contract: object, order: object) -> object:  # noqa: N802
        self.placed.append((contract, order))
        return SimpleNamespace(
            order=SimpleNamespace(orderId=123),
            orderStatus=SimpleNamespace(status="Filled", filled=Decimal("2"), remaining=0),
            fills=[
                SimpleNamespace(
                    execution=SimpleNamespace(
                        shares=Decimal("2"),
                        price=Decimal("451.25"),
                        time="2026-05-28T00:00:00Z",
                    ),
                    commissionReport=SimpleNamespace(commission=Decimal("1.00")),
                )
            ],
        )

    def reqMktData(self, contract: object, *args: object, **kwargs: object) -> object:  # noqa: N802
        return SimpleNamespace(marketPrice=lambda: 452.5)

    def sleep(self, seconds: float) -> None:
        assert seconds == 1.0


def test_connect_disconnect_and_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ibkr_client.IB", FakeIB)
    client = IBKRClient(
        IBKRConfig(host="host.docker.internal", port=4002, client_id=7, timeout_sec=3.0)
    )
    assert client.is_ready() is False
    client.connect()
    assert client.is_ready() is True
    client.disconnect()
    assert client.is_ready() is False


def test_place_market_order_normalizes_fills(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ibkr_client.IB", FakeIB)
    monkeypatch.setattr(
        "ibkr_client.Stock",
        lambda symbol, exchange, currency: SimpleNamespace(
            symbol=symbol, exchange=exchange, currency=currency
        ),
    )
    monkeypatch.setattr(
        "ibkr_client.MarketOrder",
        lambda action, quantity: SimpleNamespace(action=action, totalQuantity=quantity),
    )
    client = IBKRClient(
        IBKRConfig(host="host.docker.internal", port=4002, client_id=7, timeout_sec=3.0)
    )
    client.connect()
    result = client.place_order(
        PlaceOrderRequest(
            idempotency_key="abc",
            symbol="NVDA",
            side="buy",
            order_type="market",
            quantity=Decimal("2"),
            limit_price=None,
            time_in_force="GTC",
        )
    )
    assert result["id"] == "ibkr-123"
    assert result["status"] == "filled"
    assert result["avg_price"] == "451.25"
    assert result["filled_qty"] == "2"
    assert result["fills"][0]["fee_asset"] == "USD"


def test_duplicate_idempotency_returns_same_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ibkr_client.IB", FakeIB)
    monkeypatch.setattr(
        "ibkr_client.Stock",
        lambda symbol, exchange, currency: SimpleNamespace(symbol=symbol),
    )
    monkeypatch.setattr(
        "ibkr_client.MarketOrder",
        lambda action, quantity: SimpleNamespace(action=action, totalQuantity=quantity),
    )
    client = IBKRClient(IBKRConfig())
    client.connect()
    request = PlaceOrderRequest(
        idempotency_key="dup",
        symbol="MSFT",
        side="sell",
        order_type="market",
        quantity=Decimal("1"),
        limit_price=None,
        time_in_force="GTC",
    )
    first = client.place_order(request)
    second = client.place_order(request)
    assert second == first


def test_limit_order_requires_limit_price(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ibkr_client.IB", FakeIB)
    client = IBKRClient(IBKRConfig())
    client.connect()
    with pytest.raises(ValueError, match="limit_price"):
        client.place_order(
            PlaceOrderRequest(
                idempotency_key="limit-missing",
                symbol="NVDA",
                side="buy",
                order_type="limit",
                quantity=Decimal("1"),
                limit_price=None,
                time_in_force="GTC",
            )
        )


def test_live_gateway_port_warns(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    assert _is_live_gateway_port(7496) is True
    assert "live" in caplog.text.lower()


def test_live_port_refused_without_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_instantiated() -> NoReturn:
        raise AssertionError("IBKR network client should not be created")

    monkeypatch.setattr("ibkr_client.IB", fail_if_instantiated)
    client = IBKRClient(IBKRConfig(port=7496, allow_live_port=False))
    with pytest.raises(RuntimeError, match="LIVE_PORT_NOT_AUTHORIZED"):
        client.connect()


def test_live_port_4001_refused_without_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_instantiated() -> NoReturn:
        raise AssertionError("IBKR network client should not be created")

    monkeypatch.setattr("ibkr_client.IB", fail_if_instantiated)
    client = IBKRClient(IBKRConfig(port=4001, allow_live_port=False))
    with pytest.raises(RuntimeError, match="LIVE_PORT_NOT_AUTHORIZED"):
        client.connect()


def test_live_port_connects_when_authorized(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("ibkr_client.IB", FakeIB)
    caplog.set_level(logging.WARNING)
    client = IBKRClient(IBKRConfig(port=7496, allow_live_port=True))
    client.connect()
    assert client.is_ready() is True
    assert "IBKR_AUDIT live_port_authorized" in caplog.text


def test_paper_port_logs_audit_info(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("ibkr_client.IB", FakeIB)
    caplog.set_level(logging.INFO)
    client = IBKRClient(IBKRConfig(port=4002, allow_live_port=False))
    client.connect()
    assert client.is_ready() is True
    assert "IBKR_AUDIT paper_port" in caplog.text
