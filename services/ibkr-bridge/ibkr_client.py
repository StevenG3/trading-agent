from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

try:  # pragma: no cover - exercised in the bridge container, monkeypatched in repo tests.
    _ib_async: Any = importlib.import_module("ib_async")
except ImportError:  # pragma: no cover
    _ib_async = None

IB: Any = getattr(_ib_async, "IB", None)
LimitOrder: Any = getattr(_ib_async, "LimitOrder", None)
MarketOrder: Any = getattr(_ib_async, "MarketOrder", None)
Stock: Any = getattr(_ib_async, "Stock", None)

logger = logging.getLogger(__name__)
LIVE_GATEWAY_PORTS = {7496, 4001}

OrderStatus = Literal["pending", "submitted", "filled", "partial", "canceled", "rejected", "error"]


class PlaceOrderRequest(BaseModel):
    idempotency_key: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit"]
    quantity: Decimal
    limit_price: Decimal | None = None
    time_in_force: Literal["GTC", "IOC", "FOK"] = "GTC"

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.upper().strip()

    @field_validator("quantity")
    @classmethod
    def quantity_positive(cls, value: Decimal) -> Decimal:
        if value <= 0:
            raise ValueError("quantity must be greater than 0")
        return value


@dataclass(frozen=True)
class IBKRConfig:
    host: str = "host.docker.internal"
    port: int = 4002
    client_id: int = 1
    timeout_sec: float = 10.0
    allow_live_port: bool = False


def _is_live_gateway_port(port: int) -> bool:
    if port in LIVE_GATEWAY_PORTS:
        logger.warning("IBKR bridge configured for a live gateway port: %s", port)
        return True
    return False


def _decimal(value: object, default: Decimal = Decimal("0")) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return default


def _normalize_status(status: str) -> OrderStatus:
    value = status.lower()
    if value in {"filled"}:
        return "filled"
    if value in {"partiallyfilled", "partial"}:
        return "partial"
    if value in {"cancelled", "canceled"}:
        return "canceled"
    if value in {"inactive", "rejected"}:
        return "rejected"
    if value in {"presubmitted", "submitted", "api pending", "pending"}:
        return "submitted"
    return "pending"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class IBKRClient:
    def __init__(self, config: IBKRConfig) -> None:
        self._config = config
        self._ib: Any | None = None
        self._orders: dict[str, dict[str, object]] = {}
        self._idempotency: dict[str, str] = {}

    def connect(self) -> None:
        if IB is None:
            raise RuntimeError("ib_async is not installed")
        if _is_live_gateway_port(self._config.port):
            if not self._config.allow_live_port:
                raise RuntimeError(
                    "LIVE_PORT_NOT_AUTHORIZED: set IBKR_ALLOW_LIVE_PORT=true to use "
                    f"IBKR live gateway port {self._config.port}"
                )
            logger.warning("IBKR_AUDIT live_port_authorized port=%s", self._config.port)
        else:
            logger.info("IBKR_AUDIT paper_port port=%s", self._config.port)
        self._ib = IB()
        self._ib.connect(
            self._config.host,
            self._config.port,
            clientId=self._config.client_id,
            timeout=self._config.timeout_sec,
        )

    def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()

    def is_ready(self) -> bool:
        return self._ib is not None and bool(self._ib.isConnected())

    def _require_ready(self) -> Any:
        if self._ib is None or not self._ib.isConnected():
            raise RuntimeError("IBKR client is not connected")
        return self._ib

    def place_order(self, request: PlaceOrderRequest) -> dict[str, object]:
        existing_id = self._idempotency.get(request.idempotency_key)
        if existing_id is not None:
            return self._orders[existing_id]
        ib = self._require_ready()
        if request.order_type == "limit" and request.limit_price is None:
            raise ValueError("limit_price is required for limit orders")
        if Stock is None:
            raise RuntimeError("ib_async contract classes are not available")

        contract = Stock(request.symbol, "SMART", "USD")
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            raise ValueError("INVALID_SYMBOL")
        contract = qualified[0]
        action = "BUY" if request.side == "buy" else "SELL"
        if request.order_type == "market":
            if MarketOrder is None:
                raise RuntimeError("ib_async market order class is not available")
            order = MarketOrder(action, float(request.quantity))
        else:
            if LimitOrder is None:
                raise RuntimeError("ib_async limit order class is not available")
            assert request.limit_price is not None
            order = LimitOrder(action, float(request.quantity), float(request.limit_price))
            order.tif = request.time_in_force

        trade = ib.placeOrder(contract, order)
        result = self._result_from_trade(trade)
        order_id = str(result["id"])
        self._orders[order_id] = result
        self._idempotency[request.idempotency_key] = order_id
        return result

    def get_order(self, order_id: str) -> dict[str, object] | None:
        if order_id in self._orders:
            return self._orders[order_id]
        self._require_ready()
        return None

    def cancel_order(self, order_id: str) -> dict[str, object] | None:
        payload = self.get_order(order_id)
        if payload is None:
            return None
        updated = dict(payload)
        updated["status"] = "canceled"
        self._orders[order_id] = updated
        return updated

    def ticker(self, symbol: str) -> dict[str, str]:
        ib = self._require_ready()
        if Stock is None:
            raise RuntimeError("ib_async contract classes are not available")
        normalized = symbol.upper().strip()
        contract = Stock(normalized, "SMART", "USD")
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            raise ValueError("INVALID_SYMBOL")
        ticker = ib.reqMktData(qualified[0], "", False, False)
        ib.sleep(1.0)
        price = _decimal(ticker.marketPrice(), Decimal("0"))
        if price <= 0:
            raise ValueError("MARKET_DATA_UNAVAILABLE")
        return {"symbol": normalized, "price": str(price), "source": "ibkr"}

    def _result_from_trade(self, trade: Any) -> dict[str, object]:
        order_id = str(getattr(getattr(trade, "order", object()), "orderId", ""))
        status_raw = str(getattr(getattr(trade, "orderStatus", object()), "status", "Pending"))
        fills: list[dict[str, str]] = []
        total_qty = Decimal("0")
        total_notional = Decimal("0")
        total_fee = Decimal("0")
        for fill in getattr(trade, "fills", []) or []:
            execution = getattr(fill, "execution", object())
            qty = _decimal(getattr(execution, "shares", "0"))
            price = _decimal(getattr(execution, "price", "0"))
            commission = _decimal(
                getattr(getattr(fill, "commissionReport", object()), "commission", "0")
            )
            ts = str(getattr(execution, "time", _now_iso()))
            total_qty += qty
            total_notional += qty * price
            total_fee += commission
            fills.append(
                {
                    "price": str(price),
                    "qty": str(qty),
                    "fee": str(commission),
                    "fee_asset": "USD",
                    "ts": ts,
                }
            )
        avg_price = total_notional / total_qty if total_qty else None
        remaining = _decimal(getattr(getattr(trade, "orderStatus", object()), "remaining", "0"))
        filled_status = _normalize_status(status_raw)
        if fills and remaining == 0:
            filled_status = "filled"
        return {
            "id": f"ibkr-{order_id}",
            "status": filled_status,
            "fills": fills,
            "avg_price": str(avg_price) if avg_price is not None else None,
            "filled_qty": str(total_qty),
            "remaining_qty": str(remaining),
            "error": None,
            "raw_order_ref": order_id,
            "fees_total": str(total_fee),
        }
