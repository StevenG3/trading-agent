from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from schemas import HardCapsApplied, OrderIntent, RiskDecision, RiskReason

app = FastAPI(title="risk-engine", version="0.1.0")

MARKET_DATA_URL = os.getenv("MARKET_DATA_URL", "http://market-data:8083")
MAX_NOTIONAL_USDT = Decimal(os.getenv("MAX_NOTIONAL_USDT", "10000"))
CONFIRMATION_THRESHOLD_USDT = Decimal(os.getenv("CONFIRMATION_THRESHOLD_USDT", "500"))
PER_SYMBOL_DAILY_LIMIT_USDT = Decimal(os.getenv("PER_SYMBOL_DAILY_LIMIT_USDT", "50000"))
LIVE_TRADING_ENABLED = os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"
IBKR_LIVE_TRADING_ENABLED = (
    os.getenv("IBKR_LIVE_TRADING_ENABLED", "false").lower() == "true"
)
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8080")
DAILY_DRAWDOWN_HARD_STOP_USDT = Decimal(
    os.getenv("DAILY_DRAWDOWN_HARD_STOP_USDT", "1000")
)
SUPPORTED_VENUES = {"binance_spot", "ibkr_us_equity"}
LIVE_AVAILABLE_VENUES = {"binance_spot", "ibkr_us_equity"}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    _ = request
    return JSONResponse(status_code=400, content=jsonable_encoder({"detail": exc.errors()}))


def _hard_caps() -> HardCapsApplied:
    return HardCapsApplied(
        max_notional=MAX_NOTIONAL_USDT,
        max_leverage=None,
        max_drawdown_today=DAILY_DRAWDOWN_HARD_STOP_USDT,
        per_symbol_exposure=PER_SYMBOL_DAILY_LIMIT_USDT,
    )


def _decision(
    intent: OrderIntent,
    evaluated_at: datetime,
    approved: bool,
    reasons: list[RiskReason],
    requires_confirmation: bool = False,
    confirmation_token: str | None = None,
    confirmation_expires_at: datetime | None = None,
) -> RiskDecision:
    return RiskDecision(
        decision_id=uuid4(),
        intent_id=intent.intent_id,
        evaluated_at=evaluated_at,
        approved=approved,
        reasons=reasons,
        requires_confirmation=requires_confirmation,
        confirmation_token=confirmation_token,
        confirmation_expires_at=confirmation_expires_at,
        hard_caps_applied=_hard_caps(),
    )


def _notional(intent: OrderIntent) -> Decimal | RiskReason:
    if intent.quantity.kind == "quote":
        return intent.quantity.value
    try:
        params = {"symbol": intent.symbol}
        if intent.venue == "ibkr_us_equity":
            params["asset_type"] = "stock"
        response = httpx.get(
            f"{MARKET_DATA_URL}/ticker", params=params, timeout=3.0
        )
        response.raise_for_status()
        price = Decimal(str(response.json()["price"]))
    except (httpx.HTTPError, KeyError, ValueError):
        return RiskReason(code="MARKET_DATA_UNAVAILABLE", detail="market data unavailable")
    return intent.quantity.value * price


def _drawdown_check(intent: OrderIntent) -> RiskReason | None:
    """Return a RiskReason when today's actor PnL has breached the hard stop."""
    try:
        response = httpx.get(
            f"{ORCHESTRATOR_URL}/pnl/today",
            params={"actor": intent.actor},
            timeout=2.0,
        )
        response.raise_for_status()
        payload = response.json()
        total_pnl = Decimal(str(payload["total_pnl"]))
    except (httpx.HTTPError, KeyError, ValueError):
        if intent.mode == "live":
            return RiskReason(
                code="DRAWDOWN_CHECK_UNAVAILABLE",
                detail="could not fetch today's PnL; live blocked for safety",
            )
        return None
    if total_pnl <= -DAILY_DRAWDOWN_HARD_STOP_USDT:
        return RiskReason(
            code="DAILY_DRAWDOWN_BREACHED",
            detail=(
                f"actor PnL today {total_pnl} <= -{DAILY_DRAWDOWN_HARD_STOP_USDT}; "
                "trading paused for the day"
            ),
        )
    return None


def evaluate(intent: OrderIntent) -> RiskDecision:
    evaluated_at = datetime.now(UTC)
    reasons: list[RiskReason] = []
    if intent.mode == "live":
        if not LIVE_TRADING_ENABLED:
            reasons.append(
                RiskReason(
                    code="LIVE_TRADING_DISABLED",
                    detail="set LIVE_TRADING_ENABLED=true to enable live trading",
                )
            )
            return _decision(intent, evaluated_at, False, reasons)
        if intent.venue == "ibkr_us_equity" and not IBKR_LIVE_TRADING_ENABLED:
            reasons.append(
                RiskReason(
                    code="IBKR_LIVE_TRADING_DISABLED",
                    detail="set IBKR_LIVE_TRADING_ENABLED=true to enable IBKR live trading",
                )
            )
            return _decision(intent, evaluated_at, False, reasons)
    if intent.order_type == "limit" and intent.limit_price is None:
        reasons.append(
            RiskReason(
                code="LIMIT_PRICE_REQUIRED",
                detail="limit orders require limit_price to be set",
            )
        )
        return _decision(intent, evaluated_at, False, reasons)
    if intent.venue not in SUPPORTED_VENUES:
        reasons.append(
            RiskReason(code="UNSUPPORTED_VENUE", detail=intent.venue)
        )
        return _decision(intent, evaluated_at, False, reasons)

    drawdown = _drawdown_check(intent)
    if drawdown is not None:
        reasons.append(drawdown)
        return _decision(intent, evaluated_at, False, reasons)

    notional = _notional(intent)
    if isinstance(notional, RiskReason):
        return _decision(intent, evaluated_at, False, [notional])
    if notional > MAX_NOTIONAL_USDT:
        reasons.append(
            RiskReason(
                code="NOTIONAL_EXCEEDS_HARD_CAP",
                detail=f"notional {notional} exceeds max {MAX_NOTIONAL_USDT}",
            )
        )
    if reasons:
        return _decision(intent, evaluated_at, False, reasons)

    requires_confirmation = notional >= CONFIRMATION_THRESHOLD_USDT
    token = str(uuid4()) if requires_confirmation else None
    expires_at = evaluated_at + timedelta(minutes=5) if requires_confirmation else None
    return _decision(
        intent,
        evaluated_at,
        True,
        [],
        requires_confirmation=requires_confirmation,
        confirmation_token=token,
        confirmation_expires_at=expires_at,
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/validate", response_model=RiskDecision)
def validate(intent: OrderIntent) -> RiskDecision:
    return evaluate(intent)
