from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac as hmac_lib
import json
import os
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Literal, cast
from urllib.parse import urlparse
from uuid import UUID, uuid4

import httpx
from db import connect
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField

from schemas import (
    ConfirmationRequest,
    ExecutionRequest,
    ExecutionResult,
    OrderIntent,
    Quantity,
    RiskDecision,
    Scorecard,
    Source,
)

app = FastAPI(title="orchestrator", version="0.1.0")
PER_SYMBOL_DAILY_LIMIT_USDT = Decimal(os.getenv("PER_SYMBOL_DAILY_LIMIT_USDT", "50000"))
DECIMAL_8 = Decimal("0.00000001")
CALIBRATION_BUCKETS = [
    (Decimal("0.30"), Decimal("0.40")),
    (Decimal("0.40"), Decimal("0.50")),
    (Decimal("0.50"), Decimal("0.60")),
    (Decimal("0.60"), Decimal("0.70")),
    (Decimal("0.70"), Decimal("0.80")),
    (Decimal("0.80"), Decimal("0.90")),
    (Decimal("0.90"), Decimal("1.01")),
]


class _OutcomeSummaryBucket:
    def __init__(self) -> None:
        self.closed_count = 0
        self.hits = 0
        self.losses = 0
        self.open_count = 0
        self.total_pnl = Decimal("0")


CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
HERMES_MODEL = os.getenv("HERMES_MODEL", "claude-haiku-4-5-20251001")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
SCORECARD_DEFAULT_TTL_MIN = int(os.getenv("SCORECARD_DEFAULT_TTL_MIN", "60"))
OPS_TOKEN = os.getenv("OPS_TOKEN", "")
LIVE_UNLOCK_TTL_MIN = int(os.getenv("LIVE_UNLOCK_TTL_MIN", "15"))
ANALYSIS_ADAPTER_URL = os.getenv("ANALYSIS_ADAPTER_URL", "http://analysis-adapter:8085")
REFLECT_TIMEOUT_SEC = float(os.getenv("REFLECT_TIMEOUT_SEC", "60"))
SCHEDULER_ENABLED = os.getenv("SCHEDULER_ENABLED", "true").lower() == "true"
SCHEDULER_TICK_SEC = float(os.getenv("SCHEDULER_TICK_SEC", "60"))
SCHEDULER_BATCH_LIMIT = int(os.getenv("SCHEDULER_BATCH_LIMIT", "5"))
ORCHESTRATOR_SELF_URL = os.getenv("ORCHESTRATOR_SELF_URL", "http://localhost:8080")
AUTO_TRADE_ENABLED = os.getenv("AUTO_TRADE_ENABLED", "true").lower() == "true"
AUTO_TRADE_BATCH_LIMIT = int(os.getenv("AUTO_TRADE_BATCH_LIMIT", "5"))
LIVE_AUTONOMY_GLOBAL_ENABLED = os.getenv("LIVE_AUTONOMY_GLOBAL_ENABLED", "false").lower() == "true"
LIVE_AUTO_TICK_SEC = float(os.getenv("LIVE_AUTO_TICK_SEC", "300"))
LIVE_AUTO_BATCH_LIMIT = int(os.getenv("LIVE_AUTO_BATCH_LIMIT", "1"))
CALIBRATION_MIN_SAMPLES = int(os.getenv("CALIBRATION_MIN_SAMPLES", "5"))
CALIBRATION_SHRINKAGE_K = Decimal(os.getenv("CALIBRATION_SHRINKAGE_K", "10"))
NOTIFICATION_HOST_ALLOWLIST = frozenset(
    item.strip()
    for item in os.getenv(
        "NOTIFICATION_HOST_ALLOWLIST", "hermes-agent,hermes,localhost,127.0.0.1"
    ).split(",")
    if item.strip()
)
NOTIFICATION_TIMEOUT_SEC = float(os.getenv("NOTIFICATION_TIMEOUT_SEC", "5"))
NOTIFICATION_HISTORY_LIMIT = int(os.getenv("NOTIFICATION_HISTORY_LIMIT", "100"))
_scheduler_task: asyncio.Task[None] | None = None

_HERMES_SYSTEM_PROMPT = """\
You are an order intent parser for a cryptocurrency spot trading platform (Binance Spot only).

Given a natural language trading instruction, extract the following fields and respond with
ONLY a valid JSON object -- no markdown, no explanation, no code fences:

{
  "symbol":        "<COIN>USDT uppercase, e.g. BTCUSDT or ETHUSDT",
  "side":          "buy" | "sell",
  "order_type":    "market" | "limit",
  "quantity_kind": "quote" | "base",
  "quantity_value": "<positive decimal string, e.g. '100' or '0.001'>",
  "limit_price":   "<decimal string>" | null
}

Rules:
- quantity_kind="quote" means the user specified a USDT amount (e.g. "100 USDT of BTC")
- quantity_kind="base"  means the user specified a coin amount (e.g. "0.001 BTC")
- limit_price must be null for market orders and a decimal string for limit orders
- Only support spot pairs quoted in USDT (append USDT if the user omits it)
- If the instruction is not a valid, unambiguous trading order, respond with exactly:
  {"error": "<one-sentence reason>"}
"""


class NLIntentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = PydanticField(min_length=1)
    message: str = PydanticField(min_length=1)
    idempotency_key: str = PydanticField(min_length=1)
    hermes_message_id: str | None = None
    mode: Literal["paper", "live"] = "paper"
    request_id: UUID | None = None


NLIntentRequest.model_rebuild(_types_namespace={"Literal": Literal, "UUID": UUID})


class ScorecardCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = PydanticField(min_length=1)
    symbol: str = PydanticField(min_length=1)
    action: Literal["buy", "sell", "hold"]
    source: Literal["manual", "tradingagents", "hermes_chat"]
    conviction: str = PydanticField(min_length=1)
    thesis: str = PydanticField(min_length=1, max_length=4000)
    entry_low: str | None = None
    entry_high: str | None = None
    stop_loss: str | None = None
    take_profit: str | None = None
    time_horizon: Literal["intraday", "swing", "position"]
    ttl_minutes: int | None = None
    metadata: dict[str, str] | None = None


class ScorecardIntentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scorecard_id: UUID
    actor: str = PydanticField(min_length=1)
    idempotency_key: str = PydanticField(min_length=1)
    mode: Literal["paper", "live"] = "paper"
    usdt_budget: str = PydanticField(min_length=1)
    position_fraction: str = "1.0"
    order_type: Literal["market", "limit"] = "market"
    request_id: UUID | None = None
    intent_id: UUID | None = None


class LiveAutonomyUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = PydanticField(min_length=1)
    enabled: bool | None = None
    daily_live_budget_usdt: str | None = None
    per_live_trade_max_usdt: str | None = None
    max_live_exposure_usdt: str | None = None
    daily_live_trade_count_max: int | None = None
    min_calibrated_conviction: str | None = None
    min_closed_outcomes: int | None = None
    allowed_sources: str | None = None


class NotificationSubscribeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = PydanticField(min_length=1)
    webhook_url: str = PydanticField(min_length=1)
    secret: str | None = None
    events: list[str] = PydanticField(default_factory=lambda: ["fill"])


class AutonomyUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = PydanticField(min_length=1)
    enabled: bool | None = None
    daily_budget_usdt: str | None = None
    min_conviction: str | None = None
    per_trade_usdt: str | None = None
    allowed_sources: str | None = None


class WatchlistAddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = PydanticField(min_length=1)
    symbol: str = PydanticField(min_length=1)
    asset_type: Literal["stock", "crypto"] = "crypto"
    cadence_minutes: int = PydanticField(ge=15, le=1440)


class LiveUnlockRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = PydanticField(min_length=1)


ScorecardCreateRequest.model_rebuild(_types_namespace={"Literal": Literal})
ScorecardIntentRequest.model_rebuild(_types_namespace={"Literal": Literal, "UUID": UUID})
LiveAutonomyUpdateRequest.model_rebuild()
NotificationSubscribeRequest.model_rebuild()
AutonomyUpdateRequest.model_rebuild()
WatchlistAddRequest.model_rebuild(_types_namespace={"Literal": Literal})
LiveUnlockRequest.model_rebuild()


class DuplicateIntentIdError(Exception):
    pass


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    _ = request
    return JSONResponse(status_code=400, content=jsonable_encoder({"detail": exc.errors()}))


def _bucket_label(heuristic: Decimal) -> str | None:
    for lo, hi in CALIBRATION_BUCKETS:
        if lo <= heuristic < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return None


def _calibration_row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    return {
        "source": row["source"],
        "asset_type": row["asset_type"],
        "heuristic_bucket": row["heuristic_bucket"],
        "sample_count": row["sample_count"],
        "hit_count": row["hit_count"],
        "avg_alpha_return": row["avg_alpha_return"],
        "empirical_hit_rate": row["empirical_hit_rate"],
        "calibrated_conviction": row["calibrated_conviction"],
        "updated_at": row["updated_at"],
    }


def _watchlist_row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    return {
        "actor": row["actor"],
        "symbol": row["symbol"],
        "asset_type": row["asset_type"],
        "cadence_minutes": row["cadence_minutes"],
        "last_run_at": row["last_run_at"],
        "next_run_at": row["next_run_at"],
        "enabled": bool(row["enabled"]),
        "created_at": row["created_at"],
    }


def _fire_scheduled_analysis(actor: str, symbol: str, asset_type: str) -> bool:
    try:
        response = httpx.post(
            f"{ANALYSIS_ADAPTER_URL}/analyze",
            json={"actor": actor, "symbol": symbol, "asset_type": asset_type},
            timeout=10.0,
        )
        response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def scheduler_tick(now: datetime | None = None) -> dict[str, int]:
    now = now or _now()
    now_iso = now.isoformat()
    with connect() as conn:
        rows = conn.execute(
            "select actor, symbol, asset_type, cadence_minutes from watchlist_entries "
            "where enabled = 1 and next_run_at <= ? order by next_run_at asc limit ?",
            (now_iso, SCHEDULER_BATCH_LIMIT),
        ).fetchall()
    fired = 0
    failed = 0
    for row in rows:
        cadence = int(row["cadence_minutes"])
        ok = _fire_scheduled_analysis(str(row["actor"]), str(row["symbol"]), str(row["asset_type"]))
        next_run = now + timedelta(minutes=cadence if ok else max(cadence, 60))
        with connect() as conn:
            conn.execute(
                "update watchlist_entries set last_run_at = ?, next_run_at = ? "
                "where actor = ? and symbol = ?",
                (now_iso, next_run.isoformat(), row["actor"], row["symbol"]),
            )
            conn.commit()
        if ok:
            fired += 1
        else:
            failed += 1
    return {"due": len(rows), "fired": fired, "failed": failed}


async def _scheduler_loop() -> None:
    last_live_auto_tick = 0.0
    while True:
        try:
            scheduler_tick()
        except Exception:
            pass
        try:
            auto_trade_tick()
        except Exception:
            pass
        now_mono = time.monotonic()
        if now_mono - last_live_auto_tick >= LIVE_AUTO_TICK_SEC:
            try:
                live_auto_trade_tick()
            except Exception:
                pass
            last_live_auto_tick = now_mono
        await asyncio.sleep(SCHEDULER_TICK_SEC)


@app.on_event("startup")
async def _start_scheduler() -> None:
    global _scheduler_task
    if SCHEDULER_ENABLED and _scheduler_task is None:
        _scheduler_task = asyncio.create_task(_scheduler_loop())


@app.on_event("shutdown")
async def _stop_scheduler() -> None:
    global _scheduler_task
    if _scheduler_task is None:
        return
    _scheduler_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await _scheduler_task
    _scheduler_task = None


def _now() -> datetime:
    return datetime.now(UTC)


def _today() -> str:
    return _now().strftime("%Y-%m-%d")


def _daily_limit() -> Decimal:
    return Decimal(os.getenv("PER_SYMBOL_DAILY_LIMIT_USDT", str(PER_SYMBOL_DAILY_LIMIT_USDT)))


def _execution_url() -> str:
    return os.getenv("EXECUTION_SERVICE_URL", "http://execution-service:8082")


def _market_url() -> str:
    return os.getenv("MARKET_DATA_URL", "http://market-data:8083")


def _resolve_qty(intent: OrderIntent, market_url: str) -> tuple[str, str]:
    try:
        response = httpx.get(f"{market_url}/ticker", params={"symbol": intent.symbol}, timeout=3.0)
        response.raise_for_status()
        price_str = str(response.json()["price"])
        price = Decimal(price_str)
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=502, detail={"code": "MARKET_DATA_UNAVAILABLE"}) from exc

    if intent.quantity.kind == "base":
        return str(intent.quantity.value), price_str
    base_qty = (intent.quantity.value / price).quantize(Decimal("0.00000001"))
    return str(base_qty), price_str


def _build_execution_request(intent: OrderIntent, decision: RiskDecision) -> ExecutionRequest:
    return ExecutionRequest(
        execution_id=uuid4(),
        intent_id=intent.intent_id,
        decision_id=decision.decision_id,
        idempotency_key=intent.idempotency_key,
        confirmation_token=decision.confirmation_token,
        dry_run=False,
        submitted_at=_now(),
    )


def _call_execution(intent: OrderIntent, decision: RiskDecision, base_qty: str) -> ExecutionResult:
    request = _build_execution_request(intent, decision)
    execution_response = httpx.post(
        f"{_execution_url()}/execute",
        content=request.model_dump_json(),
        headers={
            "content-type": "application/json",
            "x-decision-approved": str(decision.approved).lower(),
            "x-mode": intent.mode,
            "x-symbol": intent.symbol,
            "x-quantity": base_qty,
            "x-side": intent.side,
            "x-quantity-kind": intent.quantity.kind,
            "x-quote-qty": str(intent.quantity.value) if intent.quantity.kind == "quote" else "",
            "x-order-type": intent.order_type,
            "x-limit-price": str(intent.limit_price) if intent.limit_price is not None else "",
            "x-time-in-force": intent.time_in_force,
        },
        timeout=5.0,
    )
    execution_response.raise_for_status()
    return ExecutionResult.model_validate(execution_response.json())


def _row_to_item(row: sqlite3.Row) -> dict[str, object]:
    execution_json = row["execution_json"]
    return {
        "status": row["status"],
        "intent": OrderIntent.model_validate_json(row["payload_json"]),
        "decision": RiskDecision.model_validate_json(row["decision_json"]),
        "execution": (
            ExecutionResult.model_validate_json(execution_json) if execution_json else None
        ),
    }


def _pending_response(intent: OrderIntent, decision: RiskDecision) -> JSONResponse:
    return JSONResponse(
        status_code=202,
        content=jsonable_encoder(
            {
                "status": "pending_confirmation",
                "intent_id": str(intent.intent_id),
                "confirmation_token": decision.confirmation_token,
                "confirmation_expires_at": decision.confirmation_expires_at,
            }
        ),
    )


def _rejected_response(decision: RiskDecision) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"code": "RISK_REJECTED", "reasons": decision.reasons}),
    )


def _idempotent_response(row: sqlite3.Row) -> JSONResponse | dict[str, object]:
    item = _row_to_item(row)
    decision = item["decision"]
    intent = item["intent"]
    if not isinstance(decision, RiskDecision) or not isinstance(intent, OrderIntent):
        raise RuntimeError("invalid persisted intent")
    status = row["status"]
    if status == "executed":
        return item
    if status == "pending_confirmation":
        return _pending_response(intent, decision)
    if status == "rejected":
        return _rejected_response(decision)
    if status == "canceled":
        return JSONResponse(status_code=410, content={"code": "INTENT_CANCELED"})
    return item


def _current_exposure(symbol: str, date: str) -> Decimal:
    with connect() as conn:
        row = conn.execute(
            "select coalesce(sum(cast(notional_usdt as real)), 0.0) "
            "from daily_fills where date = ? and symbol = ?",
            (date, symbol),
        ).fetchone()
    return Decimal(str(row[0]))


def _exposure_limit_response(
    intent: OrderIntent, current: Decimal, requested: Decimal
) -> JSONResponse | None:
    limit = _daily_limit()
    if current + requested <= limit:
        return None
    return JSONResponse(
        status_code=422,
        content={
            "code": "PER_SYMBOL_DAILY_LIMIT_EXCEEDED",
            "symbol": intent.symbol,
            "limit": str(limit),
            "current": str(current),
            "requested": str(requested),
        },
    )


def _record_fill(execution: ExecutionResult, symbol: str, side: str) -> None:
    if execution.avg_price is None or execution.filled_qty == Decimal("0"):
        return
    notional = execution.filled_qty * execution.avg_price
    with connect() as conn:
        conn.execute(
            "insert or ignore into daily_fills "
            "(fill_id, date, symbol, side, notional_usdt, created_at) "
            "values (?, ?, ?, ?, ?, ?)",
            (
                str(execution.execution_id),
                _today(),
                symbol,
                side,
                str(notional),
                _now().isoformat(),
            ),
        )
        conn.commit()


def _is_allowed_webhook_host(webhook_url: str) -> bool:
    parsed = urlparse(webhook_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    return parsed.hostname in NOTIFICATION_HOST_ALLOWLIST


def _record_notification_delivery(
    actor: str,
    event_type: str,
    webhook_url: str,
    status_code: int | None,
    ok: bool,
    error_class: str | None,
) -> None:
    with connect() as conn:
        conn.execute(
            """
            insert into notification_deliveries
              (actor, event_type, webhook_url, status_code, ok, error_class, created_at)
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                actor,
                event_type,
                webhook_url,
                status_code,
                1 if ok else 0,
                error_class,
                _now().isoformat(),
            ),
        )
        conn.execute(
            """
            delete from notification_deliveries
            where actor = ?
              and id not in (
                select id from notification_deliveries
                where actor = ?
                order by created_at desc, id desc
                limit ?
              )
            """,
            (actor, actor, NOTIFICATION_HISTORY_LIMIT),
        )
        conn.commit()


def _deliver_webhook(
    actor: str, webhook_url: str, secret: str, event_type: str, payload: dict[str, object]
) -> None:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    signature = hmac_lib.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
    try:
        response = httpx.post(
            webhook_url,
            content=body,
            headers={
                "content-type": "application/json",
                "x-trading-agent-event": event_type,
                "x-trading-agent-signature": signature,
            },
            timeout=NOTIFICATION_TIMEOUT_SEC,
        )
        status_code = int(getattr(response, "status_code", 0) or 0)
        _record_notification_delivery(
            actor, event_type, webhook_url, status_code, 200 <= status_code < 300, None
        )
    except Exception as exc:  # noqa: BLE001
        _record_notification_delivery(
            actor, event_type, webhook_url, None, False, exc.__class__.__name__
        )


def _notify_fill(intent: OrderIntent, execution: ExecutionResult) -> None:
    if execution.avg_price is None or execution.filled_qty == Decimal("0"):
        return
    with connect() as conn:
        row = conn.execute(
            "select webhook_url, secret, events_json from notification_subscriptions "
            "where actor = ? and enabled = 1",
            (intent.actor,),
        ).fetchone()
    if row is None:
        return
    try:
        events = json.loads(str(row["events_json"]))
    except json.JSONDecodeError:
        return
    if "fill" not in events:
        return
    payload: dict[str, object] = {
        "event_type": "fill",
        "actor": intent.actor,
        "symbol": intent.symbol,
        "side": intent.side,
        "qty_str": _q8s(execution.filled_qty),
        "avg_price_str": _q8s(execution.avg_price),
        "mode": intent.mode,
        "status": execution.status,
        "intent_id": str(intent.intent_id),
    }
    _deliver_webhook(intent.actor, str(row["webhook_url"]), str(row["secret"]), "fill", payload)


def _after_fill_side_effects(execution: ExecutionResult, intent: OrderIntent) -> None:
    _record_fill(execution, intent.symbol, intent.side)
    _update_position(execution, intent)
    with contextlib.suppress(Exception):
        _notify_fill(intent, execution)


def _q8(value: Decimal) -> Decimal:
    return value.quantize(DECIMAL_8)


def _q8s(value: Decimal) -> str:
    return f"{_q8(value):.8f}"


def _maybe_open_scorecard_outcome(
    execution: ExecutionResult,
    intent: OrderIntent,
    new_qty_after_buy: Decimal,
) -> None:
    """Record an opened outcome when a scorecard-sourced BUY produces fill."""
    _ = new_qty_after_buy
    if intent.side != "buy":
        return
    if intent.source.origin != "scorecard" or not intent.source.scorecard_id:
        return
    if execution.avg_price is None or execution.filled_qty <= 0:
        return
    with connect() as conn:
        row = conn.execute(
            "select source from scorecards where scorecard_id = ?",
            (intent.source.scorecard_id,),
        ).fetchone()
    source = row["source"] if row else "unknown"
    opened_qty = _q8(execution.filled_qty)
    opened_avg_cost = _q8(execution.avg_price)
    opened_cost_basis = _q8(opened_qty * opened_avg_cost)
    with connect() as conn:
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
                str(uuid4()),
                intent.source.scorecard_id,
                intent.actor,
                intent.symbol,
                source,
                intent.side,
                str(intent.intent_id),
                _now().isoformat(),
                _q8s(opened_qty),
                _q8s(opened_avg_cost),
                _q8s(opened_cost_basis),
            ),
        )
        conn.commit()


def _maybe_close_scorecard_outcomes(
    actor: str,
    symbol: str,
    realized_delta: Decimal,
    new_qty: Decimal,
) -> None:
    if new_qty != Decimal("0"):
        return
    with connect() as conn:
        rows = conn.execute(
            "select outcome_id, opened_cost_basis from scorecard_outcomes "
            "where actor = ? and symbol = ? and status = 'open'",
            (actor, symbol),
        ).fetchall()
    if not rows:
        return
    total_basis = sum(Decimal(row["opened_cost_basis"]) for row in rows) or Decimal("1")
    closed_at = _now().isoformat()
    notes = "split-attribution" if len(rows) > 1 else None
    closed_ids: list[str] = []
    with connect() as conn:
        for row in rows:
            basis = Decimal(row["opened_cost_basis"])
            share = (basis / total_basis) if total_basis else Decimal("0")
            attributed = _q8(realized_delta * share)
            return_pct = _q8(attributed / basis) if basis > 0 else Decimal("0")
            conn.execute(
                "update scorecard_outcomes set status = 'closed', "
                "closed_at = ?, closed_realized_pnl = ?, closed_return_pct = ?, "
                "notes = ? where outcome_id = ?",
                (closed_at, _q8s(attributed), _q8s(return_pct), notes, row["outcome_id"]),
            )
            closed_ids.append(str(row["outcome_id"]))
        conn.commit()

    for outcome_id in closed_ids:
        outcome = _load_outcome_for_reflection(outcome_id)
        if outcome and _push_outcome_reflection(outcome):
            _mark_outcome_reflected(outcome_id)


def _load_outcome_for_reflection(outcome_id: str) -> dict[str, object] | None:
    with connect() as conn:
        row = conn.execute(
            "select outcome_id, scorecard_id, actor, symbol, source, action, "
            "opened_intent_id, opened_at, opened_qty, opened_avg_cost, "
            "opened_cost_basis, status, closed_at, closed_realized_pnl, "
            "closed_return_pct, notes, reflected_at from scorecard_outcomes where outcome_id = ?",
            (outcome_id,),
        ).fetchone()
    if row is None:
        return None
    return _outcome_row_to_dict(row)


def _scorecard_metadata(scorecard_id: str) -> dict[str, str]:
    with connect() as conn:
        row = conn.execute(
            "select payload_json from scorecards where scorecard_id = ?", (scorecard_id,)
        ).fetchone()
    if row is None:
        return {}
    try:
        payload = json.loads(str(row["payload_json"]))
    except json.JSONDecodeError:
        return {}
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    return metadata if isinstance(metadata, dict) else {}


def _compute_alpha_return(
    raw_return: Decimal, benchmark_symbol: str | None, benchmark_open: str | None
) -> tuple[Decimal, str | None]:
    if not benchmark_symbol or benchmark_symbol == "self":
        return _q8(raw_return), "benchmark unavailable; using raw return as alpha"
    if not benchmark_open:
        return _q8(raw_return), "benchmark open price unavailable; using raw return as alpha"
    try:
        open_price = Decimal(str(benchmark_open))
        if open_price <= 0:
            raise InvalidOperation
    except (InvalidOperation, ValueError):
        return _q8(raw_return), "benchmark open price invalid; using raw return as alpha"
    close_price, _source = _mark_for_symbol_str(benchmark_symbol)
    if close_price is None:
        return _q8(raw_return), "benchmark close price unavailable; using raw return as alpha"
    benchmark_return = (close_price - open_price) / open_price
    return _q8(raw_return - benchmark_return), None


def _holding_days(opened_at: str | None, closed_at: str | None) -> int:
    try:
        opened = datetime.fromisoformat(str(opened_at).replace("Z", "+00:00"))
        closed = datetime.fromisoformat(str(closed_at).replace("Z", "+00:00"))
    except ValueError:
        return 0
    return max(0, (closed.date() - opened.date()).days)


def _push_outcome_reflection(outcome: dict[str, object]) -> bool:
    metadata = _scorecard_metadata(str(outcome.get("scorecard_id", "")))
    closed_return = str(outcome.get("closed_return_pct") or "0")
    try:
        raw_return = _q8(Decimal(closed_return))
    except (InvalidOperation, ValueError):
        raw_return = Decimal("0")
    benchmark_symbol = metadata.get("benchmark_symbol")
    alpha_return, alpha_note = _compute_alpha_return(
        raw_return, benchmark_symbol, metadata.get("benchmark_open_price")
    )
    payload = {
        "ticker": metadata.get("ta_ticker") or outcome.get("symbol"),
        "trade_date": metadata.get("ta_date") or str(outcome.get("opened_at", ""))[:10],
        "raw_return": _q8s(raw_return),
        "alpha_return": _q8s(alpha_return),
        "holding_days": _holding_days(
            str(outcome.get("opened_at") or ""), str(outcome.get("closed_at") or "")
        ),
        "provider": metadata.get("provider"),
        "benchmark_name": benchmark_symbol or "paper-position-baseline",
    }
    if alpha_note:
        payload["alpha_note"] = alpha_note
    try:
        response = httpx.post(
            f"{ANALYSIS_ADAPTER_URL}/reflect/outcome",
            json=payload,
            timeout=REFLECT_TIMEOUT_SEC,
        )
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError):
        return False
    return bool(isinstance(body, dict) and body.get("ok") and body.get("reflected"))


def _mark_outcome_reflected(outcome_id: str) -> None:
    with connect() as conn:
        conn.execute(
            "update scorecard_outcomes set reflected_at = ? where outcome_id = ?",
            (_now().isoformat(), outcome_id),
        )
        conn.commit()


def _update_position(execution: ExecutionResult, intent: OrderIntent) -> None:
    if execution.avg_price is None or execution.filled_qty == Decimal("0"):
        return
    fill_qty = _q8(execution.filled_qty)
    fill_price = execution.avg_price
    with connect() as conn:
        row = conn.execute(
            "select qty, avg_cost, total_cost, realized_pnl from paper_positions "
            "where actor = ? and symbol = ?",
            (intent.actor, intent.symbol),
        ).fetchone()
        old_qty = Decimal(row["qty"]) if row else Decimal("0")
        old_avg = Decimal(row["avg_cost"]) if row else Decimal("0")
        old_realized = Decimal(row["realized_pnl"]) if row else Decimal("0")

        if intent.side == "buy":
            new_qty = _q8(old_qty + fill_qty)
            total_cost = _q8((old_qty * old_avg) + (fill_qty * fill_price))
            avg_cost = _q8(total_cost / new_qty) if new_qty else Decimal("0")
            realized = old_realized
            realized_delta = Decimal("0")
        else:
            sell_qty = min(fill_qty, old_qty)
            new_qty = _q8(old_qty - sell_qty)
            realized_delta = _q8(sell_qty * (fill_price - old_avg))
            realized = _q8(old_realized + realized_delta)
            avg_cost = _q8(old_avg if new_qty else Decimal("0"))
            total_cost = _q8(new_qty * avg_cost)

        conn.execute(
            """
            insert into paper_positions
            (actor, symbol, qty, avg_cost, total_cost, realized_pnl, last_updated)
            values (?, ?, ?, ?, ?, ?, ?)
            on conflict(actor, symbol) do update set
                qty = excluded.qty,
                avg_cost = excluded.avg_cost,
                total_cost = excluded.total_cost,
                realized_pnl = excluded.realized_pnl,
                last_updated = excluded.last_updated
            """,
            (
                intent.actor,
                intent.symbol,
                _q8s(new_qty),
                _q8s(avg_cost),
                _q8s(total_cost),
                _q8s(realized),
                _now().isoformat(),
            ),
        )
        if realized_delta != Decimal("0"):
            conn.execute(
                "insert into daily_pnl (actor, date, realized_delta, symbol, created_at) "
                "values (?,?,?,?,?)",
                (
                    intent.actor,
                    _today(),
                    _q8s(realized_delta),
                    intent.symbol,
                    _now().isoformat(),
                ),
            )
        conn.commit()
        try:
            if intent.side == "buy":
                _maybe_open_scorecard_outcome(execution, intent, new_qty)
            else:
                _maybe_close_scorecard_outcomes(
                    intent.actor, intent.symbol, realized_delta, new_qty
                )
        except Exception:
            pass


def _mark_for_symbol_str(symbol: str) -> tuple[Decimal | None, str | None]:
    return _mark_for_symbol(symbol)


def _mark_for_symbol(symbol: str) -> tuple[Decimal | None, str | None]:
    try:
        response = httpx.get(f"{_market_url()}/ticker", params={"symbol": symbol}, timeout=3.0)
        response.raise_for_status()
        payload = response.json()
        return Decimal(str(payload["price"])), str(payload.get("source", "binance"))
    except (httpx.HTTPError, KeyError, ValueError):
        return None, None


def _parse_nl_to_intent_fields(message: str) -> dict[str, object]:
    """Call the Claude API and return the parsed JSON dict."""
    response = httpx.post(
        CLAUDE_API_URL,
        headers={
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": HERMES_MODEL,
            "max_tokens": 256,
            "system": _HERMES_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": message}],
        },
        timeout=15.0,
    )
    response.raise_for_status()
    payload = response.json()
    text = str(payload["content"][0]["text"]).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Claude response was not a JSON object")
    return {str(key): value for key, value in parsed.items()}


def _build_intent_from_nl(nl: NLIntentRequest, fields: dict[str, object]) -> OrderIntent:
    """Construct an OrderIntent from extracted NL fields."""
    side_raw = str(fields["side"])
    if side_raw not in {"buy", "sell"}:
        raise ValueError("side must be buy or sell")
    side = cast(Literal["buy", "sell"], side_raw)

    order_type_raw = str(fields["order_type"])
    if order_type_raw not in {"market", "limit"}:
        raise ValueError("order_type must be market or limit")
    order_type = cast(Literal["market", "limit"], order_type_raw)

    quantity_kind_raw = str(fields["quantity_kind"])
    if quantity_kind_raw not in {"base", "quote"}:
        raise ValueError("quantity_kind must be base or quote")
    quantity_kind = cast(Literal["base", "quote"], quantity_kind_raw)

    limit_price = (
        Decimal(str(fields["limit_price"])) if fields.get("limit_price") is not None else None
    )

    return OrderIntent(
        intent_id=uuid4(),
        request_id=nl.request_id or uuid4(),
        idempotency_key=nl.idempotency_key,
        actor=nl.actor,
        created_at=_now(),
        mode=nl.mode,
        venue="binance_spot",
        symbol=str(fields["symbol"]),
        side=side,
        order_type=order_type,
        quantity=Quantity(
            kind=quantity_kind,
            value=Decimal(str(fields["quantity_value"])),
        ),
        limit_price=limit_price,
        time_in_force="GTC",
        reduce_only=False,
        leverage=None,
        stop_loss=None,
        take_profit=None,
        source=Source(
            origin="user_nl",
            scorecard_id=None,
            hermes_message_id=nl.hermes_message_id,
        ),
        client_confirmation_required=False,
    )


def _consume_live_unlock_or_error(
    token: str, actor: str, dry: bool, intent_id: UUID | None = None
) -> JSONResponse | None:
    """Validate or consume a single-use live-unlock token."""
    if not token:
        return JSONResponse(status_code=403, content={"code": "LIVE_UNLOCK_REQUIRED"})
    now_iso = _now().isoformat()
    with connect() as conn:
        row = conn.execute(
            "select actor, expires_at, consumed_at, bound_intent_id "
            "from live_unlock_tokens where token = ?",
            (token,),
        ).fetchone()
    if row is None:
        return JSONResponse(status_code=403, content={"code": "INVALID_LIVE_UNLOCK"})
    if row["actor"] != actor:
        return JSONResponse(status_code=403, content={"code": "LIVE_UNLOCK_ACTOR_MISMATCH"})
    if row["consumed_at"] is not None:
        return JSONResponse(status_code=403, content={"code": "LIVE_UNLOCK_ALREADY_USED"})
    if row["expires_at"] < now_iso:
        return JSONResponse(status_code=410, content={"code": "LIVE_UNLOCK_EXPIRED"})
    if row["bound_intent_id"] is not None:
        if intent_id is None or row["bound_intent_id"] != str(intent_id):
            return JSONResponse(status_code=403, content={"code": "LIVE_UNLOCK_INTENT_MISMATCH"})
    if not dry:
        with connect() as conn:
            cursor = conn.execute(
                "update live_unlock_tokens set consumed_at = ? "
                "where token = ? and consumed_at is NULL",
                (now_iso, token),
            )
            conn.commit()
        if cursor.rowcount == 0:
            return JSONResponse(status_code=403, content={"code": "LIVE_UNLOCK_ALREADY_USED"})
    return None


def _scorecard_should_mark_consumed(
    response: JSONResponse | dict[str, object],
) -> bool:
    if isinstance(response, dict):
        return True
    code = getattr(response, "status_code", 500)
    if code == 202:
        return True
    if code == 422:
        body = json.loads(bytes(response.body).decode())
        if body.get("code") == "RISK_REJECTED":
            return True
    return False


def _cancel_refresh_request(intent: OrderIntent, execution: ExecutionResult) -> ExecutionRequest:
    return ExecutionRequest(
        execution_id=uuid4(),
        intent_id=intent.intent_id,
        decision_id=execution.decision_id,
        idempotency_key=intent.idempotency_key,
        confirmation_token=None,
        dry_run=False,
        submitted_at=_now(),
    )


def _call_cancel(intent: OrderIntent, execution: ExecutionResult) -> ExecutionResult:
    req = _cancel_refresh_request(intent, execution)
    response = httpx.post(
        f"{_execution_url()}/cancel",
        content=req.model_dump_json(),
        headers={
            "content-type": "application/json",
            "x-mode": intent.mode,
            "x-symbol": intent.symbol,
            "x-venue-order-id": execution.venue_order_id or "",
            "x-order-type": intent.order_type,
        },
        timeout=15.0,
    )
    response.raise_for_status()
    return ExecutionResult.model_validate(response.json())


def _call_refresh(intent: OrderIntent, execution: ExecutionResult) -> ExecutionResult:
    req = _cancel_refresh_request(intent, execution)
    response = httpx.post(
        f"{_execution_url()}/refresh",
        content=req.model_dump_json(),
        headers={
            "content-type": "application/json",
            "x-mode": intent.mode,
            "x-symbol": intent.symbol,
            "x-venue-order-id": execution.venue_order_id or "",
            "x-order-type": intent.order_type,
        },
        timeout=10.0,
    )
    response.raise_for_status()
    return ExecutionResult.model_validate(response.json())


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/notifications/subscribe", response_model=None)
def subscribe_notifications(req: NotificationSubscribeRequest) -> JSONResponse | dict[str, object]:
    if not _is_allowed_webhook_host(req.webhook_url):
        return JSONResponse(status_code=400, content={"code": "WEBHOOK_HOST_NOT_ALLOWED"})
    events = sorted({event for event in req.events if event == "fill"})
    if not events:
        return JSONResponse(status_code=400, content={"code": "NO_SUPPORTED_EVENTS"})
    secret = req.secret or str(uuid4())
    now_iso = _now().isoformat()
    with connect() as conn:
        conn.execute(
            """
            insert into notification_subscriptions
              (actor, webhook_url, secret, events_json, enabled, created_at, updated_at)
            values (?, ?, ?, ?, 1, ?, ?)
            on conflict(actor) do update set
              webhook_url = excluded.webhook_url,
              secret = excluded.secret,
              events_json = excluded.events_json,
              enabled = 1,
              updated_at = excluded.updated_at
            """,
            (req.actor, req.webhook_url, secret, json.dumps(events), now_iso, now_iso),
        )
        conn.commit()
    return {
        "actor": req.actor,
        "webhook_url": req.webhook_url,
        "events": events,
        "secret": secret,
    }


@app.post("/notifications/unsubscribe", response_model=None)
def unsubscribe_notifications(actor: str | None = None) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    with connect() as conn:
        cursor = conn.execute(
            "update notification_subscriptions set enabled = 0, updated_at = ? where actor = ?",
            (_now().isoformat(), actor),
        )
        conn.commit()
    return {"actor": actor, "enabled": False, "changed": cursor.rowcount > 0}


@app.get("/notifications/deliveries", response_model=None)
def list_notification_deliveries(
    actor: str | None = None, limit: int = Query(default=20, ge=1)
) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    clamped = min(limit, NOTIFICATION_HISTORY_LIMIT)
    with connect() as conn:
        rows = conn.execute(
            """
            select event_type, webhook_url, status_code, ok, error_class, created_at
            from notification_deliveries
            where actor = ?
            order by created_at desc, id desc
            limit ?
            """,
            (actor, clamped),
        ).fetchall()
    return {
        "actor": actor,
        "deliveries": [
            {
                "event_type": row["event_type"],
                "webhook_url": row["webhook_url"],
                "status_code": row["status_code"],
                "ok": bool(row["ok"]),
                "error_class": row["error_class"],
                "created_at": row["created_at"],
            }
            for row in rows
        ],
    }


def _live_kill_switch_active() -> bool:
    with connect() as conn:
        row = conn.execute("select killed from live_autonomy_kill where id = 1").fetchone()
    return bool(row and row["killed"])


def _default_live_autonomy(actor: str) -> dict[str, object]:
    return {
        "actor": actor,
        "enabled": False,
        "daily_live_budget_usdt": "0",
        "per_live_trade_max_usdt": "50",
        "max_live_exposure_usdt": "0",
        "current_live_exposure_usdt": "0",
        "daily_live_trade_count_max": 3,
        "min_calibrated_conviction": "0.70",
        "min_closed_outcomes": 20,
        "allowed_sources": "tradingagents",
        "created_at": None,
        "updated_at": None,
    }


def _live_autonomy_row_to_dict(actor: str, row: sqlite3.Row | None) -> dict[str, object]:
    if row is None:
        return _default_live_autonomy(actor)
    max_exposure = str(row["max_live_exposure_usdt"])
    current_exposure = _check_live_exposure_cap(
        actor, Decimal("0"), Decimal(max_exposure)
    )[1]
    return {
        "actor": actor,
        "enabled": bool(row["enabled"]),
        "daily_live_budget_usdt": str(row["daily_live_budget_usdt"]),
        "per_live_trade_max_usdt": str(row["per_live_trade_max_usdt"]),
        "max_live_exposure_usdt": max_exposure,
        "current_live_exposure_usdt": _q8s(current_exposure) if current_exposure else "0",
        "daily_live_trade_count_max": int(row["daily_live_trade_count_max"]),
        "min_calibrated_conviction": str(row["min_calibrated_conviction"]),
        "min_closed_outcomes": int(row["min_closed_outcomes"]),
        "allowed_sources": str(row["allowed_sources"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _check_drawdown_for_live_auto(actor: str) -> str | None:
    try:
        body = get_pnl_today(actor=actor)
    except Exception:
        return "DRAWDOWN_CHECK_UNAVAILABLE"
    if isinstance(body, JSONResponse):
        return "DRAWDOWN_CHECK_UNAVAILABLE"
    try:
        total = Decimal(str(body.get("total_pnl", "0")))
    except (InvalidOperation, ValueError):
        return "DRAWDOWN_CHECK_UNAVAILABLE"
    cap = Decimal(os.getenv("DAILY_DRAWDOWN_HARD_STOP_USDT", "1000"))
    if total <= -cap:
        return "DAILY_DRAWDOWN_BREACHED"
    return None


def _check_live_exposure_cap(
    actor: str, proposed_notional: Decimal, max_cap: Decimal
) -> tuple[bool, Decimal]:
    with connect() as conn:
        rows = conn.execute(
            """
            select p.qty, p.avg_cost
            from paper_positions p
            where p.actor = ?
              and cast(p.qty as real) > 0
              and exists (
                select 1 from intents i
                where json_extract(i.payload_json, '$.actor') = p.actor
                  and json_extract(i.payload_json, '$.symbol') = p.symbol
                  and json_extract(i.payload_json, '$.mode') = 'live'
              )
            """,
            (actor,),
        ).fetchall()
    current = Decimal("0")
    for row in rows:
        current += Decimal(str(row["qty"])) * Decimal(str(row["avg_cost"]))
    current = _q8(current)
    return current + proposed_notional <= max_cap, current


def _eligible_for_live_auto(
    actor: str, scorecard: dict[str, object], settings: dict[str, object]
) -> tuple[bool, str]:
    if not LIVE_AUTONOMY_GLOBAL_ENABLED:
        return False, "LIVE_AUTONOMY_GLOBAL_DISABLED"
    if _live_kill_switch_active():
        return False, "LIVE_AUTONOMY_KILL_SWITCH"
    if not settings.get("enabled"):
        return False, "ACTOR_NOT_OPTED_IN"
    source = str(scorecard.get("source", ""))
    allowed = {s.strip() for s in str(settings.get("allowed_sources", "")).split(",") if s.strip()}
    if source not in allowed:
        return False, "SOURCE_NOT_ALLOWED"
    try:
        conviction = Decimal(str(scorecard.get("conviction", "0")))
        min_conv = Decimal(str(settings.get("min_calibrated_conviction", "0.70")))
    except (InvalidOperation, ValueError):
        return False, "INVALID_CONVICTION"
    if conviction < min_conv:
        return False, "BELOW_MIN_CONVICTION"
    metadata = scorecard.get("metadata") or {}
    if not isinstance(metadata, dict):
        return False, "MISSING_METADATA"
    asset_type = str(metadata.get("asset_type") or "crypto")
    heuristic_raw = metadata.get("heuristic_conviction")
    if heuristic_raw is None:
        return False, "MISSING_HEURISTIC"
    try:
        heuristic = Decimal(str(heuristic_raw))
    except (InvalidOperation, ValueError):
        return False, "INVALID_HEURISTIC"
    bucket = _bucket_label(heuristic)
    if bucket is None:
        return False, "HEURISTIC_OUT_OF_RANGE"
    with connect() as conn:
        cal = conn.execute(
            "select sample_count, calibrated_conviction from conviction_calibration "
            "where source = ? and asset_type = ? and heuristic_bucket = ?",
            (source, asset_type, bucket),
        ).fetchone()
    if cal is None:
        return False, "NO_CALIBRATION_DATA"
    if int(cal["sample_count"]) < int(str(settings.get("min_closed_outcomes", 20))):
        return False, "INSUFFICIENT_SAMPLES"
    today = _today()
    with connect() as conn:
        spend = conn.execute(
            "select spent_usdt, trade_count from live_autonomy_spend where actor = ? and date = ?",
            (actor, today),
        ).fetchone()
    spent = Decimal(str(spend["spent_usdt"])) if spend else Decimal("0")
    count = int(spend["trade_count"]) if spend else 0
    try:
        max_cap = Decimal(str(settings.get("max_live_exposure_usdt", "0")))
        proposed = Decimal(str(settings.get("per_live_trade_max_usdt", "0")))
    except (InvalidOperation, ValueError):
        return False, "INVALID_LIVE_EXPOSURE_CAP"
    if max_cap <= 0:
        return False, "MAX_LIVE_EXPOSURE_NOT_SET"
    allowed_by_cap, current_exposure = _check_live_exposure_cap(actor, proposed, max_cap)
    if not allowed_by_cap:
        return False, f"LIVE_EXPOSURE_CAP_BREACHED:{_q8s(current_exposure)}/{max_cap}"
    budget = Decimal(str(settings.get("daily_live_budget_usdt", "0")))
    if spent >= budget:
        return False, "DAILY_BUDGET_EXHAUSTED"
    if count >= int(str(settings.get("daily_live_trade_count_max", 3))):
        return False, "DAILY_TRADE_COUNT_EXHAUSTED"
    pnl_block = _check_drawdown_for_live_auto(actor)
    if pnl_block is not None:
        return False, pnl_block
    return True, "OK"


def _mint_user_live_unlock_token(actor: str) -> str:
    token = str(uuid4())
    created = _now()
    expires = created + timedelta(minutes=LIVE_UNLOCK_TTL_MIN)
    with connect() as conn:
        conn.execute(
            "insert into live_unlock_tokens "
            "(token, actor, created_at, expires_at, consumed_at, bound_intent_id) "
            "values (?, ?, ?, ?, NULL, NULL)",
            (token, actor, created.isoformat(), expires.isoformat()),
        )
        conn.commit()
    return token


def _mint_auto_unlock_bound_token(actor: str, intent_id: UUID | str) -> str:
    token = str(uuid4())
    created = _now()
    expires = created + timedelta(minutes=2)
    with connect() as conn:
        conn.execute(
            "insert into live_unlock_tokens "
            "(token, actor, created_at, expires_at, consumed_at, bound_intent_id) "
            "values (?, ?, ?, ?, NULL, ?)",
            (token, actor, created.isoformat(), expires.isoformat(), str(intent_id)),
        )
        conn.commit()
    return token


def _record_live_autonomy_spend(actor: str, date: str, amount: Decimal) -> None:
    now_iso = _now().isoformat()
    with connect() as conn:
        row = conn.execute(
            "select spent_usdt, trade_count from live_autonomy_spend where actor = ? and date = ?",
            (actor, date),
        ).fetchone()
        spent = _q8((Decimal(str(row["spent_usdt"])) if row else Decimal("0")) + amount)
        count = (int(row["trade_count"]) if row else 0) + 1
        conn.execute(
            "insert into live_autonomy_spend "
            "(actor, date, spent_usdt, trade_count, last_updated) values (?, ?, ?, ?, ?) "
            "on conflict(actor, date) do update set spent_usdt = excluded.spent_usdt, "
            "trade_count = excluded.trade_count, last_updated = excluded.last_updated",
            (actor, date, _q8s(spent), count, now_iso),
        )
        conn.commit()


def _fire_live_autonomous_trade(actor: str, scorecard_id: str, usdt_budget: Decimal) -> bool:
    intent_id = uuid4()
    token = _mint_auto_unlock_bound_token(actor, intent_id)
    payload = {
        "scorecard_id": scorecard_id,
        "actor": actor,
        "idempotency_key": f"live-auto-{actor}-{scorecard_id[:8]}-{int(time.time())}",
        "usdt_budget": str(usdt_budget),
        "position_fraction": "1.0",
        "mode": "live",
        "intent_id": str(intent_id),
    }
    try:
        response = httpx.post(
            f"{ORCHESTRATOR_SELF_URL}/intents/from_scorecard",
            json=payload,
            headers={"x-live-unlock": token},
            timeout=15.0,
        )
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError):
        return False
    return isinstance(body, dict) and body.get("status") == "executed"


def live_auto_trade_tick(now: datetime | None = None) -> dict[str, object]:
    if not LIVE_AUTONOMY_GLOBAL_ENABLED:
        return {"placed": 0, "skipped": 0, "reason": "GLOBAL_DISABLED"}
    if _live_kill_switch_active():
        return {"placed": 0, "skipped": 0, "reason": "KILL_SWITCH"}
    current = now or _now()
    placed = 0
    skipped = 0
    with connect() as conn:
        actors = conn.execute(
            "select actor, enabled, daily_live_budget_usdt, per_live_trade_max_usdt, "
            "max_live_exposure_usdt, daily_live_trade_count_max, min_calibrated_conviction, "
            "min_closed_outcomes, allowed_sources, created_at, updated_at "
            "from live_autonomy_settings where enabled = 1"
        ).fetchall()
    for actor_row in actors:
        actor = str(actor_row["actor"])
        settings = _live_autonomy_row_to_dict(actor, actor_row)
        with connect() as conn:
            candidates = conn.execute(
                """
                select scorecard_id, payload_json, source from scorecards
                where actor = ? and consumed_by_intent_id is NULL
                  and expires_at > ?
                  and action in ('buy','sell')
                order by created_at asc limit ?
                """,
                (actor, current.isoformat(), LIVE_AUTO_BATCH_LIMIT),
            ).fetchall()
        for row in candidates:
            try:
                payload = json.loads(str(row["payload_json"]))
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(payload, dict):
                skipped += 1
                continue
            eligible, _reason = _eligible_for_live_auto(actor, payload, settings)
            if not eligible:
                skipped += 1
                continue
            today = current.strftime("%Y-%m-%d")
            with connect() as conn:
                spend_row = conn.execute(
                    "select spent_usdt from live_autonomy_spend where actor = ? and date = ?",
                    (actor, today),
                ).fetchone()
            spent = Decimal(str(spend_row["spent_usdt"])) if spend_row else Decimal("0")
            budget = Decimal(str(settings["daily_live_budget_usdt"]))
            per_trade = Decimal(str(settings["per_live_trade_max_usdt"]))
            spend_amount = min(per_trade, budget - spent)
            if spend_amount < Decimal("10"):
                skipped += 1
                continue
            if _fire_live_autonomous_trade(actor, str(row["scorecard_id"]), spend_amount):
                placed += 1
                _record_live_autonomy_spend(actor, today, spend_amount)
                break
            skipped += 1
    return {"placed": placed, "skipped": skipped}


def _record_autonomy_spend(actor: str, date: str, amount: Decimal) -> None:
    now_iso = _now().isoformat()
    with connect() as conn:
        row = conn.execute(
            "select spent_usdt, trade_count from autonomy_spend where actor = ? and date = ?",
            (actor, date),
        ).fetchone()
        if row is None:
            spent = _q8(amount)
            count = 1
        else:
            spent = _q8(Decimal(str(row["spent_usdt"])) + amount)
            count = int(row["trade_count"]) + 1
        conn.execute(
            "insert into autonomy_spend "
            "(actor, date, spent_usdt, trade_count, last_updated) "
            "values (?, ?, ?, ?, ?) "
            "on conflict(actor, date) do update set "
            "spent_usdt = excluded.spent_usdt, "
            "trade_count = excluded.trade_count, "
            "last_updated = excluded.last_updated",
            (actor, date, _q8s(spent), count, now_iso),
        )
        conn.commit()


def _fire_autonomous_trade(actor: str, scorecard_id: str, usdt_budget: Decimal) -> bool:
    payload = {
        "scorecard_id": scorecard_id,
        "actor": actor,
        "idempotency_key": f"auto-{actor}-{scorecard_id[:8]}-{int(time.time())}",
        "usdt_budget": str(usdt_budget),
        "position_fraction": "1.0",
        "mode": "paper",
    }
    try:
        response = httpx.post(
            f"{ORCHESTRATOR_SELF_URL}/intents/from_scorecard",
            json=payload,
            headers={"x-live-unlock": ""},
            timeout=10.0,
        )
        return getattr(response, "status_code", 200) in {200, 202}
    except httpx.HTTPError:
        return False


def auto_trade_tick(now: datetime | None = None) -> dict[str, object]:
    if not AUTO_TRADE_ENABLED:
        return {"placed": 0, "skipped_budget": 0, "skipped_other": 0, "reason": "disabled"}
    current = now or _now()
    today = current.strftime("%Y-%m-%d")
    placed = 0
    skipped_budget = 0
    skipped_other = 0
    with connect() as conn:
        actors = conn.execute(
            "select actor, daily_budget_usdt, min_conviction, per_trade_usdt, allowed_sources "
            "from autonomy_settings where enabled = 1"
        ).fetchall()
    for actor_row in actors:
        try:
            actor = str(actor_row["actor"])
            budget = Decimal(str(actor_row["daily_budget_usdt"]))
            min_conv = Decimal(str(actor_row["min_conviction"]))
            per_trade = Decimal(str(actor_row["per_trade_usdt"]))
        except (InvalidOperation, ValueError):
            skipped_other += 1
            continue
        allowed = {
            item.strip() for item in str(actor_row["allowed_sources"]).split(",") if item.strip()
        }
        with connect() as conn:
            spend_row = conn.execute(
                "select spent_usdt from autonomy_spend where actor = ? and date = ?",
                (actor, today),
            ).fetchone()
        spent = Decimal(str(spend_row["spent_usdt"])) if spend_row else Decimal("0")
        if spent >= budget:
            continue
        with connect() as conn:
            candidates = conn.execute(
                """
                select scorecard_id, payload_json, source from scorecards
                where actor = ? and consumed_by_intent_id is NULL
                  and expires_at > ?
                  and action in ('buy','sell')
                  and cast(json_extract(payload_json, '$.conviction') as real) >= ?
                order by created_at asc limit ?
                """,
                (actor, current.isoformat(), float(min_conv), AUTO_TRADE_BATCH_LIMIT),
            ).fetchall()
        for row in candidates:
            if str(row["source"]) not in allowed:
                skipped_other += 1
                continue
            remaining = budget - spent
            spend = min(per_trade, remaining)
            if spend < Decimal("1"):
                skipped_budget += 1
                continue
            if _fire_autonomous_trade(actor, str(row["scorecard_id"]), spend):
                spent += spend
                placed += 1
                _record_autonomy_spend(actor, today, spend)
            else:
                skipped_other += 1
            if spent >= budget:
                break
    return {"placed": placed, "skipped_budget": skipped_budget, "skipped_other": skipped_other}


@app.post("/calibration/recompute", response_model=None)
def recompute_calibration() -> dict[str, object]:
    with connect() as conn:
        rows = conn.execute(
            """
            select o.source, s.payload_json, o.closed_return_pct
            from scorecard_outcomes o
            join scorecards s on s.scorecard_id = o.scorecard_id
            where o.status = 'closed' and o.reflected_at is not null
            """
        ).fetchall()
    buckets: dict[tuple[str, str, str], list[tuple[Decimal, Decimal]]] = {}
    for row in rows:
        try:
            payload = json.loads(str(row["payload_json"]))
            metadata = payload.get("metadata") or {}
            heuristic_raw = metadata.get("heuristic_conviction") or payload.get("conviction")
            asset_type = str(metadata.get("asset_type") or "crypto")
            heuristic = Decimal(str(heuristic_raw))
            alpha = Decimal(str(row["closed_return_pct"]))
        except (json.JSONDecodeError, InvalidOperation, KeyError, TypeError):
            continue
        bucket = _bucket_label(heuristic)
        if bucket is None:
            continue
        buckets.setdefault((str(row["source"]), asset_type, bucket), []).append((heuristic, alpha))
    now_iso = _now().isoformat()
    written = 0
    with connect() as conn:
        conn.execute("delete from conviction_calibration")
        for (source, asset_type, bucket), samples in buckets.items():
            sample_count = len(samples)
            hit_count = sum(1 for _, alpha in samples if alpha > 0)
            avg_alpha = sum((alpha for _, alpha in samples), Decimal("0")) / Decimal(sample_count)
            empirical = Decimal(hit_count) / Decimal(sample_count)
            k = CALIBRATION_SHRINKAGE_K
            calibrated = (Decimal(hit_count) + k * Decimal("0.5")) / (Decimal(sample_count) + k)
            conn.execute(
                """
                insert into conviction_calibration
                  (source, asset_type, heuristic_bucket, sample_count, hit_count,
                   avg_alpha_return, empirical_hit_rate, calibrated_conviction, updated_at)
                values (?,?,?,?,?,?,?,?,?)
                """,
                (
                    source,
                    asset_type,
                    bucket,
                    sample_count,
                    hit_count,
                    _q8s(avg_alpha),
                    _q8s(empirical),
                    _q8s(calibrated),
                    now_iso,
                ),
            )
            written += 1
        conn.commit()
    return {"buckets_written": written, "rows_considered": len(rows)}


@app.get("/calibration", response_model=None)
def get_calibration(source: str | None = None, asset_type: str | None = None) -> dict[str, object]:
    clauses: list[str] = []
    params: list[object] = []
    if source:
        clauses.append("source = ?")
        params.append(source)
    if asset_type:
        clauses.append("asset_type = ?")
        params.append(asset_type)
    where = (" where " + " and ".join(clauses)) if clauses else ""
    with connect() as conn:
        rows = conn.execute(
            "select source, asset_type, heuristic_bucket, sample_count, hit_count, "
            "avg_alpha_return, empirical_hit_rate, calibrated_conviction, updated_at "
            f"from conviction_calibration{where} "
            "order by source, asset_type, heuristic_bucket",
            params,
        ).fetchall()
    return {
        "shrinkage_k": str(CALIBRATION_SHRINKAGE_K),
        "min_samples": CALIBRATION_MIN_SAMPLES,
        "items": [_calibration_row_to_dict(row) for row in rows],
    }


@app.post("/autonomy/settings", response_model=None)
def update_autonomy(req: AutonomyUpdateRequest) -> JSONResponse | dict[str, object]:
    with connect() as conn:
        existing = conn.execute(
            "select enabled, daily_budget_usdt, min_conviction, per_trade_usdt, "
            "allowed_sources from autonomy_settings where actor = ?",
            (req.actor,),
        ).fetchone()
    current: dict[str, object] = {
        "enabled": int(existing["enabled"]) if existing else 0,
        "daily_budget_usdt": str(existing["daily_budget_usdt"]) if existing else "0",
        "min_conviction": str(existing["min_conviction"]) if existing else "0.65",
        "per_trade_usdt": str(existing["per_trade_usdt"]) if existing else "50",
        "allowed_sources": str(existing["allowed_sources"]) if existing else "tradingagents",
    }
    if req.enabled is not None:
        current["enabled"] = 1 if req.enabled else 0
    try:
        if req.daily_budget_usdt is not None:
            if Decimal(req.daily_budget_usdt) < 0:
                return JSONResponse(status_code=400, content={"code": "INVALID_BUDGET"})
            current["daily_budget_usdt"] = req.daily_budget_usdt
        if req.min_conviction is not None:
            min_conviction = Decimal(req.min_conviction)
            if not (Decimal("0") <= min_conviction <= Decimal("1")):
                return JSONResponse(status_code=400, content={"code": "INVALID_MIN_CONVICTION"})
            current["min_conviction"] = req.min_conviction
        if req.per_trade_usdt is not None:
            if Decimal(req.per_trade_usdt) <= 0:
                return JSONResponse(status_code=400, content={"code": "INVALID_PER_TRADE"})
            current["per_trade_usdt"] = req.per_trade_usdt
    except InvalidOperation:
        return JSONResponse(status_code=400, content={"code": "INVALID_DECIMAL"})
    if req.allowed_sources is not None:
        current["allowed_sources"] = req.allowed_sources
    with connect() as conn:
        conn.execute(
            """
            insert into autonomy_settings
              (actor, enabled, daily_budget_usdt, min_conviction, per_trade_usdt,
               allowed_sources, updated_at)
            values (?,?,?,?,?,?,?)
            on conflict(actor) do update set
              enabled = excluded.enabled,
              daily_budget_usdt = excluded.daily_budget_usdt,
              min_conviction = excluded.min_conviction,
              per_trade_usdt = excluded.per_trade_usdt,
              allowed_sources = excluded.allowed_sources,
              updated_at = excluded.updated_at
            """,
            (
                req.actor,
                current["enabled"],
                current["daily_budget_usdt"],
                current["min_conviction"],
                current["per_trade_usdt"],
                current["allowed_sources"],
                _now().isoformat(),
            ),
        )
        conn.commit()
    return {"actor": req.actor, **current, "enabled": bool(current["enabled"])}


@app.get("/autonomy/settings", response_model=None)
def get_autonomy_settings(actor: str | None = None) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    with connect() as conn:
        row = conn.execute(
            "select enabled, daily_budget_usdt, min_conviction, per_trade_usdt, "
            "allowed_sources, updated_at from autonomy_settings where actor = ?",
            (actor,),
        ).fetchone()
    if row is None:
        return {
            "actor": actor,
            "enabled": False,
            "daily_budget_usdt": "0",
            "min_conviction": "0.65",
            "per_trade_usdt": "50",
            "allowed_sources": "tradingagents",
            "updated_at": None,
        }
    return {
        "actor": actor,
        "enabled": bool(row["enabled"]),
        "daily_budget_usdt": str(row["daily_budget_usdt"]),
        "min_conviction": str(row["min_conviction"]),
        "per_trade_usdt": str(row["per_trade_usdt"]),
        "allowed_sources": str(row["allowed_sources"]),
        "updated_at": row["updated_at"],
    }


@app.get("/autonomy/today", response_model=None)
def get_autonomy_today(actor: str | None = None) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    today = _today()
    with connect() as conn:
        row = conn.execute(
            "select spent_usdt, trade_count from autonomy_spend where actor = ? and date = ?",
            (actor, today),
        ).fetchone()
    return {
        "actor": actor,
        "date": today,
        "spent_usdt": str(row["spent_usdt"]) if row else "0",
        "trade_count": int(row["trade_count"]) if row else 0,
    }


@app.post("/admin/live-autonomy/disable", response_model=None)
def disable_live_autonomy(
    x_ops_token: str = Header(default=""),
) -> JSONResponse | dict[str, object]:
    if not OPS_TOKEN or x_ops_token != OPS_TOKEN:
        return JSONResponse(status_code=403, content={"code": "INVALID_OPS_TOKEN"})
    global LIVE_AUTONOMY_GLOBAL_ENABLED
    LIVE_AUTONOMY_GLOBAL_ENABLED = False
    killed_at = _now().isoformat()
    with connect() as conn:
        conn.execute(
            "update live_autonomy_kill set killed = 1, killed_at = ?, killed_by = ? where id = 1",
            (killed_at, "ops_token"),
        )
        conn.commit()
    return {"killed": True, "killed_at": killed_at}


@app.post("/admin/live-autonomy/enable", response_model=None)
def reenable_live_autonomy(
    x_ops_token: str = Header(default=""),
) -> JSONResponse | dict[str, object]:
    if not OPS_TOKEN or x_ops_token != OPS_TOKEN:
        return JSONResponse(status_code=403, content={"code": "INVALID_OPS_TOKEN"})
    with connect() as conn:
        conn.execute(
            "update live_autonomy_kill set killed = 0, killed_at = ?, "
            "killed_by = NULL where id = 1",
            (_now().isoformat(),),
        )
        conn.commit()
    return {"killed": False, "note": "kill flag cleared; env var still gates"}


@app.post("/live-autonomy/settings", response_model=None)
def update_live_autonomy(req: LiveAutonomyUpdateRequest) -> JSONResponse | dict[str, object]:
    with connect() as conn:
        row = conn.execute(
            "select enabled, daily_live_budget_usdt, per_live_trade_max_usdt, "
            "max_live_exposure_usdt, daily_live_trade_count_max, min_calibrated_conviction, "
            "min_closed_outcomes, allowed_sources, created_at, updated_at "
            "from live_autonomy_settings where actor = ?",
            (req.actor,),
        ).fetchone()
    current = _live_autonomy_row_to_dict(req.actor, row)
    now_iso = _now().isoformat()
    if current["created_at"] is None:
        current["created_at"] = now_iso
    if req.enabled is not None:
        current["enabled"] = req.enabled
    try:
        if req.daily_live_budget_usdt is not None:
            if Decimal(req.daily_live_budget_usdt) < 0:
                return JSONResponse(status_code=400, content={"code": "INVALID_BUDGET"})
            current["daily_live_budget_usdt"] = req.daily_live_budget_usdt
        if req.per_live_trade_max_usdt is not None:
            per_trade = Decimal(req.per_live_trade_max_usdt)
            if per_trade <= 0 or per_trade > Decimal("500"):
                return JSONResponse(status_code=400, content={"code": "INVALID_PER_TRADE"})
            current["per_live_trade_max_usdt"] = req.per_live_trade_max_usdt
        if req.max_live_exposure_usdt is not None:
            max_exposure = Decimal(req.max_live_exposure_usdt)
            if max_exposure < 0:
                return JSONResponse(status_code=400, content={"code": "INVALID_MAX_EXPOSURE"})
            if max_exposure > Decimal("100000"):
                return JSONResponse(status_code=400, content={"code": "MAX_EXPOSURE_TOO_HIGH"})
            current["max_live_exposure_usdt"] = req.max_live_exposure_usdt
        if req.min_calibrated_conviction is not None:
            min_conv = Decimal(req.min_calibrated_conviction)
            if not (Decimal("0.5") <= min_conv <= Decimal("1.0")):
                return JSONResponse(status_code=400, content={"code": "INVALID_MIN_CONVICTION"})
            current["min_calibrated_conviction"] = req.min_calibrated_conviction
    except (InvalidOperation, ValueError):
        return JSONResponse(status_code=400, content={"code": "INVALID_DECIMAL"})
    if req.daily_live_trade_count_max is not None:
        if req.daily_live_trade_count_max < 1 or req.daily_live_trade_count_max > 10:
            return JSONResponse(status_code=400, content={"code": "INVALID_TRADE_COUNT"})
        current["daily_live_trade_count_max"] = req.daily_live_trade_count_max
    if req.min_closed_outcomes is not None:
        if req.min_closed_outcomes < 5:
            return JSONResponse(status_code=400, content={"code": "INVALID_MIN_CLOSED_OUTCOMES"})
        current["min_closed_outcomes"] = req.min_closed_outcomes
    if req.allowed_sources is not None:
        current["allowed_sources"] = req.allowed_sources
    current["updated_at"] = now_iso
    with connect() as conn:
        conn.execute(
            """
            insert into live_autonomy_settings
              (actor, enabled, daily_live_budget_usdt, per_live_trade_max_usdt,
               max_live_exposure_usdt, daily_live_trade_count_max, min_calibrated_conviction,
               min_closed_outcomes, allowed_sources, created_at, updated_at)
            values (?,?,?,?,?,?,?,?,?,?,?)
            on conflict(actor) do update set
              enabled = excluded.enabled,
              daily_live_budget_usdt = excluded.daily_live_budget_usdt,
              per_live_trade_max_usdt = excluded.per_live_trade_max_usdt,
              max_live_exposure_usdt = excluded.max_live_exposure_usdt,
              daily_live_trade_count_max = excluded.daily_live_trade_count_max,
              min_calibrated_conviction = excluded.min_calibrated_conviction,
              min_closed_outcomes = excluded.min_closed_outcomes,
              allowed_sources = excluded.allowed_sources,
              updated_at = excluded.updated_at
            """,
            (
                req.actor,
                1 if current["enabled"] else 0,
                current["daily_live_budget_usdt"],
                current["per_live_trade_max_usdt"],
                current["max_live_exposure_usdt"],
                current["daily_live_trade_count_max"],
                current["min_calibrated_conviction"],
                current["min_closed_outcomes"],
                current["allowed_sources"],
                current["created_at"],
                current["updated_at"],
            ),
        )
        conn.commit()
    return current


@app.get("/live-autonomy/settings", response_model=None)
def get_live_autonomy_settings(actor: str | None = None) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    with connect() as conn:
        row = conn.execute(
            "select enabled, daily_live_budget_usdt, per_live_trade_max_usdt, "
            "max_live_exposure_usdt, daily_live_trade_count_max, min_calibrated_conviction, "
            "min_closed_outcomes, allowed_sources, created_at, updated_at "
            "from live_autonomy_settings where actor = ?",
            (actor,),
        ).fetchone()
    return _live_autonomy_row_to_dict(actor, row)


@app.get("/live-autonomy/today", response_model=None)
def get_live_autonomy_today(actor: str | None = None) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    today = _today()
    with connect() as conn:
        row = conn.execute(
            "select spent_usdt, trade_count from live_autonomy_spend where actor = ? and date = ?",
            (actor, today),
        ).fetchone()
    return {
        "actor": actor,
        "date": today,
        "spent_usdt": str(row["spent_usdt"]) if row else "0",
        "trade_count": int(row["trade_count"]) if row else 0,
    }


@app.post("/watchlist", response_model=None)
def add_watchlist(req: WatchlistAddRequest) -> dict[str, object]:
    now = _now()
    next_run = now + timedelta(minutes=req.cadence_minutes)
    symbol = req.symbol.upper()
    with connect() as conn:
        conn.execute(
            "insert into watchlist_entries "
            "(actor, symbol, asset_type, cadence_minutes, last_run_at, "
            "next_run_at, enabled, created_at) "
            "values (?,?,?,?,NULL,?,?,?) "
            "on conflict(actor, symbol) do update set "
            "asset_type = excluded.asset_type, cadence_minutes = excluded.cadence_minutes, "
            "next_run_at = excluded.next_run_at, enabled = 1",
            (
                req.actor,
                symbol,
                req.asset_type,
                req.cadence_minutes,
                next_run.isoformat(),
                1,
                now.isoformat(),
            ),
        )
        row = conn.execute(
            "select actor, symbol, asset_type, cadence_minutes, last_run_at, "
            "next_run_at, enabled, created_at "
            "from watchlist_entries where actor = ? and symbol = ?",
            (req.actor, symbol),
        ).fetchone()
        conn.commit()
    return {"item": _watchlist_row_to_dict(row)}


@app.get("/watchlist", response_model=None)
def list_watchlist(actor: str = Query(..., min_length=1)) -> dict[str, object]:
    with connect() as conn:
        rows = conn.execute(
            "select actor, symbol, asset_type, cadence_minutes, last_run_at, "
            "next_run_at, enabled, created_at "
            "from watchlist_entries where actor = ? and enabled = 1 order by symbol",
            (actor,),
        ).fetchall()
    return {"items": [_watchlist_row_to_dict(row) for row in rows]}


@app.delete("/watchlist/{symbol}", response_model=None)
def delete_watchlist(symbol: str, actor: str = Query(..., min_length=1)) -> dict[str, bool]:
    with connect() as conn:
        cursor = conn.execute(
            "update watchlist_entries set enabled = 0 where actor = ? and symbol = ?",
            (actor, symbol.upper()),
        )
        conn.commit()
    return {"deleted": cursor.rowcount > 0}


@app.post("/admin/live-unlock", response_model=None)
def issue_live_unlock(
    req: LiveUnlockRequest, x_ops_token: str = Header(default="")
) -> JSONResponse | dict[str, object]:
    if not OPS_TOKEN:
        return JSONResponse(
            status_code=503,
            content={"code": "LIVE_UNLOCK_DISABLED", "detail": "OPS_TOKEN not configured"},
        )
    if x_ops_token != OPS_TOKEN:
        return JSONResponse(status_code=403, content={"code": "INVALID_OPS_TOKEN"})
    token = _mint_user_live_unlock_token(req.actor)
    expires = _now() + timedelta(minutes=LIVE_UNLOCK_TTL_MIN)
    return {"token": token, "actor": req.actor, "expires_at": expires}


@app.post("/scorecards", response_model=None)
def create_scorecard(req: ScorecardCreateRequest) -> JSONResponse | dict[str, object]:
    ttl_min = req.ttl_minutes if req.ttl_minutes is not None else SCORECARD_DEFAULT_TTL_MIN
    if ttl_min <= 0 or ttl_min > 1440:
        return JSONResponse(
            status_code=400,
            content={"code": "INVALID_TTL", "detail": "ttl_minutes must be 1..1440"},
        )
    now = _now()
    try:
        scorecard = Scorecard(
            scorecard_id=uuid4(),
            created_at=now,
            expires_at=now + timedelta(minutes=ttl_min),
            source=req.source,
            actor=req.actor,
            symbol=req.symbol,
            action=req.action,
            conviction=Decimal(req.conviction),
            thesis=req.thesis,
            entry_low=Decimal(req.entry_low) if req.entry_low else None,
            entry_high=Decimal(req.entry_high) if req.entry_high else None,
            stop_loss=Decimal(req.stop_loss) if req.stop_loss else None,
            take_profit=Decimal(req.take_profit) if req.take_profit else None,
            time_horizon=req.time_horizon,
            metadata=req.metadata,
        )
    except (InvalidOperation, ValueError) as exc:
        return JSONResponse(
            status_code=400, content={"code": "INVALID_SCORECARD", "detail": str(exc)}
        )
    with connect() as conn:
        conn.execute(
            "insert into scorecards "
            "(scorecard_id, actor, symbol, action, source, payload_json, "
            "created_at, expires_at, consumed_by_intent_id) "
            "values (?,?,?,?,?,?,?,?,NULL)",
            (
                str(scorecard.scorecard_id),
                scorecard.actor,
                scorecard.symbol,
                scorecard.action,
                scorecard.source,
                scorecard.model_dump_json(),
                scorecard.created_at.isoformat(),
                scorecard.expires_at.isoformat(),
            ),
        )
        conn.commit()
    return scorecard.model_dump()


@app.get("/scorecards/{scorecard_id}", response_model=None)
def get_scorecard(scorecard_id: UUID) -> JSONResponse | Scorecard:
    with connect() as conn:
        row = conn.execute(
            "select payload_json from scorecards where scorecard_id = ?",
            (str(scorecard_id),),
        ).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={"code": "SCORECARD_NOT_FOUND"})
    return Scorecard.model_validate_json(row["payload_json"])


@app.get("/scorecards", response_model=None)
def list_scorecards(
    actor: str | None = None,
    symbol: str | None = None,
    active_only: bool = False,
    limit: int = Query(default=50, ge=1),
) -> dict[str, object]:
    clauses: list[str] = []
    params: list[object] = []
    if actor:
        clauses.append("actor = ?")
        params.append(actor)
    if symbol:
        clauses.append("symbol = ?")
        params.append(symbol)
    if active_only:
        clauses.append("expires_at > ?")
        clauses.append("consumed_by_intent_id is NULL")
        params.append(_now().isoformat())
    where = (" where " + " and ".join(clauses)) if clauses else ""
    with connect() as conn:
        rows = conn.execute(
            f"select payload_json, consumed_by_intent_id from scorecards{where} "
            "order by created_at desc limit ?",
            [*params, min(limit, 200)],
        ).fetchall()
    items: list[dict[str, object]] = []
    for row in rows:
        sc = Scorecard.model_validate_json(row["payload_json"])
        items.append(
            {
                "scorecard": sc,
                "consumed_by_intent_id": row["consumed_by_intent_id"],
                "is_expired": sc.expires_at < _now(),
            }
        )
    return {"items": items, "total": len(items)}


@app.get("/scorecard-outcomes", response_model=None)
def list_outcomes(
    actor: str | None = None,
    symbol: str | None = None,
    source: str | None = None,
    status: Literal["open", "closed"] | None = None,
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, object]:
    clauses: list[str] = []
    params: list[object] = []
    for col, val in (
        ("actor", actor),
        ("symbol", symbol),
        ("source", source),
        ("status", status),
    ):
        if val is not None:
            clauses.append(f"{col} = ?")
            params.append(val)
    where = (" where " + " and ".join(clauses)) if clauses else ""
    with connect() as conn:
        rows = conn.execute(
            f"""
            select outcome_id, scorecard_id, actor, symbol, source, action,
                   opened_intent_id, opened_at, opened_qty, opened_avg_cost,
                   opened_cost_basis, status, closed_at, closed_realized_pnl,
                   closed_return_pct, notes, reflected_at
            from scorecard_outcomes{where}
            order by opened_at desc
            limit ?
            """,
            [*params, limit],
        ).fetchall()
    return {
        "items": [_outcome_row_to_dict(row) for row in rows],
        "attribution_rule": (
            "When the aggregate position closes with multiple open outcomes, "
            "realized PnL is split proportionally by opened_cost_basis. "
            "Rows with notes='split-attribution' indicate this case."
        ),
    }


@app.get("/scorecard-outcomes/summary", response_model=None)
def outcomes_summary(actor: str | None = None, since: str | None = None) -> dict[str, object]:
    clauses_closed: list[str] = ["status = 'closed'"]
    params_closed: list[object] = []
    if actor:
        clauses_closed.append("actor = ?")
        params_closed.append(actor)
    if since:
        clauses_closed.append("closed_at >= ?")
        params_closed.append(since)
    where_closed = " where " + " and ".join(clauses_closed)
    with connect() as conn:
        closed_rows = conn.execute(
            f"select source, closed_realized_pnl from scorecard_outcomes{where_closed}",
            params_closed,
        ).fetchall()
        open_clauses = ["status = 'open'"]
        open_params: list[object] = []
        if actor:
            open_clauses.append("actor = ?")
            open_params.append(actor)
        open_rows = conn.execute(
            f"select source, count(*) as n from scorecard_outcomes "
            f"where {' and '.join(open_clauses)} group by source",
            open_params,
        ).fetchall()
        pending_reflection_clauses = ["status = 'closed'", "reflected_at is null"]
        pending_reflection_params: list[object] = []
        if actor:
            pending_reflection_clauses.append("actor = ?")
            pending_reflection_params.append(actor)
        if since:
            pending_reflection_clauses.append("closed_at >= ?")
            pending_reflection_params.append(since)
        pending_reflection_rows = conn.execute(
            f"select source, count(*) as n from scorecard_outcomes "
            f"where {' and '.join(pending_reflection_clauses)} group by source",
            pending_reflection_params,
        ).fetchall()

    by_source: dict[str, _OutcomeSummaryBucket] = {}
    for row in closed_rows:
        src = str(row["source"])
        pnl = Decimal(str(row["closed_realized_pnl"]))
        bucket = by_source.setdefault(src, _OutcomeSummaryBucket())
        bucket.closed_count += 1
        bucket.total_pnl += pnl
        if pnl > 0:
            bucket.hits += 1
        elif pnl < 0:
            bucket.losses += 1

    pending_reflections: dict[str, int] = {}
    for row in pending_reflection_rows:
        pending_reflections[str(row["source"])] = int(str(row["n"]))
        by_source.setdefault(str(row["source"]), _OutcomeSummaryBucket())

    for row in open_rows:
        bucket = by_source.setdefault(str(row["source"]), _OutcomeSummaryBucket())
        bucket.open_count = int(str(row["n"]))

    summary: dict[str, object] = {}
    for src, bucket in by_source.items():
        summary[src] = {
            "closed_count": bucket.closed_count,
            "open_count": bucket.open_count,
            "hits": bucket.hits,
            "losses": bucket.losses,
            "hit_rate": (
                f"{bucket.hits / bucket.closed_count:.4f}" if bucket.closed_count else "0.0000"
            ),
            "realized_pnl": _q8s(bucket.total_pnl),
            "total_pnl": _q8s(bucket.total_pnl),
            "pending_reflection_count": pending_reflections.get(src, 0),
        }
    return {"actor": actor, "since": since, "by_source": summary}


@app.post("/reflect/pending", response_model=None)
def reflect_pending(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, object]:
    with connect() as conn:
        rows = conn.execute(
            "select outcome_id, scorecard_id, actor, symbol, source, action, "
            "opened_intent_id, opened_at, opened_qty, opened_avg_cost, "
            "opened_cost_basis, status, closed_at, closed_realized_pnl, "
            "closed_return_pct, notes, reflected_at from scorecard_outcomes "
            "where status = 'closed' and reflected_at is null "
            "order by closed_at asc limit ?",
            (limit,),
        ).fetchall()
    attempted = 0
    reflected = 0
    failed = 0
    for row in rows:
        attempted += 1
        outcome = _outcome_row_to_dict(row)
        if _push_outcome_reflection(outcome):
            _mark_outcome_reflected(str(outcome["outcome_id"]))
            reflected += 1
        else:
            failed += 1
    return {"attempted": attempted, "reflected": reflected, "failed": failed}


@app.get("/scorecard-outcomes/{outcome_id}", response_model=None)
def get_outcome(outcome_id: UUID) -> JSONResponse | dict[str, object]:
    with connect() as conn:
        row = conn.execute(
            "select outcome_id, scorecard_id, actor, symbol, source, action, "
            "opened_intent_id, opened_at, opened_qty, opened_avg_cost, "
            "opened_cost_basis, status, closed_at, closed_realized_pnl, "
            "closed_return_pct, notes, reflected_at from scorecard_outcomes where outcome_id = ?",
            (str(outcome_id),),
        ).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={"code": "OUTCOME_NOT_FOUND"})
    return _outcome_row_to_dict(row)


def _outcome_row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    return {
        "outcome_id": row["outcome_id"],
        "scorecard_id": row["scorecard_id"],
        "actor": row["actor"],
        "symbol": row["symbol"],
        "source": row["source"],
        "action": row["action"],
        "opened_intent_id": row["opened_intent_id"],
        "opened_at": row["opened_at"],
        "opened_qty": row["opened_qty"],
        "opened_avg_cost": row["opened_avg_cost"],
        "opened_cost_basis": row["opened_cost_basis"],
        "status": row["status"],
        "closed_at": row["closed_at"],
        "closed_realized_pnl": row["closed_realized_pnl"],
        "closed_return_pct": row["closed_return_pct"],
        "notes": row["notes"],
        "reflected_at": row["reflected_at"],
    }


@app.get("/pnl/today", response_model=None)
def get_pnl_today(actor: str | None = None) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    today = _today()
    with connect() as conn:
        rows = conn.execute(
            "select symbol, realized_delta from daily_pnl where actor = ? and date = ?",
            (actor, today),
        ).fetchall()
        position_rows = conn.execute(
            "select symbol, qty, avg_cost from paper_positions where actor = ?",
            (actor,),
        ).fetchall()

    realized = Decimal("0")
    by_symbol_realized: dict[str, Decimal] = {}
    for row in rows:
        delta = Decimal(row["realized_delta"])
        realized += delta
        by_symbol_realized[row["symbol"]] = (
            by_symbol_realized.get(row["symbol"], Decimal("0")) + delta
        )

    unrealized = Decimal("0")
    by_symbol_unrealized: dict[str, Decimal] = {}
    for row in position_rows:
        qty = Decimal(row["qty"])
        if qty <= 0:
            continue
        mark, _ = _mark_for_symbol(row["symbol"])
        if mark is None:
            continue
        delta = _q8(qty * (mark - Decimal(row["avg_cost"])))
        unrealized += delta
        by_symbol_unrealized[row["symbol"]] = delta

    return {
        "actor": actor,
        "date": today,
        "realized_pnl": _q8s(realized),
        "unrealized_pnl": _q8s(unrealized),
        "total_pnl": _q8s(realized + unrealized),
        "by_symbol": {
            "realized": {k: _q8s(v) for k, v in by_symbol_realized.items()},
            "unrealized": {k: _q8s(v) for k, v in by_symbol_unrealized.items()},
        },
    }


@app.post("/intents/from_nl", response_model=None)
def create_intent_from_nl(
    nl: NLIntentRequest,
    x_live_unlock: str = Header(default=""),
) -> JSONResponse | dict[str, object]:
    if not CLAUDE_API_KEY:
        return JSONResponse(
            status_code=503,
            content={
                "code": "HERMES_UNAVAILABLE",
                "detail": "CLAUDE_API_KEY not configured",
            },
        )

    try:
        fields = _parse_nl_to_intent_fields(nl.message)
    except (httpx.HTTPError, KeyError, ValueError, json.JSONDecodeError) as exc:
        return JSONResponse(
            status_code=503,
            content={"code": "HERMES_UNAVAILABLE", "detail": str(exc)[:300]},
        )

    if "error" in fields:
        return JSONResponse(
            status_code=400,
            content={"code": "HERMES_PARSE_ERROR", "detail": str(fields["error"])},
        )

    try:
        intent = _build_intent_from_nl(nl, fields)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=400,
            content={"code": "HERMES_PARSE_ERROR", "detail": str(exc)[:300]},
        )

    return create_intent(intent, x_live_unlock=x_live_unlock)


@app.post("/intents/from_scorecard", response_model=None)
def create_intent_from_scorecard(
    req: ScorecardIntentRequest,
    x_live_unlock: str = Header(default=""),
) -> JSONResponse | dict[str, object]:
    with connect() as conn:
        row = conn.execute(
            "select payload_json, consumed_by_intent_id, expires_at from scorecards "
            "where scorecard_id = ?",
            (str(req.scorecard_id),),
        ).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={"code": "SCORECARD_NOT_FOUND"})
    if row["consumed_by_intent_id"] is not None:
        return JSONResponse(
            status_code=409,
            content={
                "code": "SCORECARD_ALREADY_CONSUMED",
                "intent_id": row["consumed_by_intent_id"],
            },
        )

    if row["expires_at"] < _now().isoformat():
        return JSONResponse(status_code=410, content={"code": "SCORECARD_EXPIRED"})
    scorecard = Scorecard.model_validate_json(row["payload_json"])
    if scorecard.expires_at < _now():
        return JSONResponse(status_code=410, content={"code": "SCORECARD_EXPIRED"})
    if scorecard.action == "hold":
        return JSONResponse(
            status_code=400,
            content={
                "code": "SCORECARD_ACTION_HOLD",
                "detail": "hold scorecards are informational only",
            },
        )
    if scorecard.actor != req.actor:
        return JSONResponse(status_code=403, content={"code": "SCORECARD_ACTOR_MISMATCH"})

    try:
        budget = Decimal(req.usdt_budget)
        fraction = Decimal(req.position_fraction)
    except (InvalidOperation, ValueError):
        return JSONResponse(
            status_code=400,
            content={
                "code": "INVALID_AMOUNT",
                "detail": "usdt_budget / position_fraction must be decimal strings",
            },
        )
    if budget <= 0 or fraction <= 0 or fraction > 1:
        return JSONResponse(
            status_code=400,
            content={
                "code": "INVALID_AMOUNT",
                "detail": "0 < position_fraction <= 1 and budget > 0",
            },
        )

    notional = (budget * scorecard.conviction * fraction).quantize(Decimal("0.01"))
    if notional <= 0:
        return JSONResponse(
            status_code=400,
            content={
                "code": "ZERO_SIZED_INTENT",
                "detail": "conviction * budget * fraction rounded to zero",
            },
        )

    limit_price: Decimal | None = None
    if req.order_type == "limit":
        if scorecard.action == "buy":
            limit_price = scorecard.entry_low or scorecard.entry_high
        else:
            limit_price = scorecard.entry_high or scorecard.entry_low
        if limit_price is None:
            return JSONResponse(status_code=400, content={"code": "SCORECARD_MISSING_ENTRY_PRICE"})

    side: Literal["buy", "sell"] = scorecard.action
    intent_id = req.intent_id or uuid4()
    intent = OrderIntent(
        intent_id=intent_id,
        request_id=req.request_id or uuid4(),
        idempotency_key=req.idempotency_key,
        actor=req.actor,
        created_at=_now(),
        mode=req.mode,
        venue="binance_spot",
        symbol=scorecard.symbol,
        side=side,
        order_type=req.order_type,
        quantity=Quantity(kind="quote", value=notional),
        limit_price=limit_price,
        time_in_force="GTC",
        reduce_only=False,
        leverage=None,
        stop_loss=scorecard.stop_loss,
        take_profit=scorecard.take_profit,
        source=Source(
            origin="scorecard",
            scorecard_id=str(scorecard.scorecard_id),
            hermes_message_id=None,
        ),
        client_confirmation_required=False,
    )

    response = create_intent(intent, x_live_unlock=x_live_unlock)
    if _scorecard_should_mark_consumed(response):
        with connect() as conn:
            cursor = conn.execute(
                "update scorecards set consumed_by_intent_id = ? "
                "where scorecard_id = ? and consumed_by_intent_id is NULL",
                (str(intent.intent_id), str(scorecard.scorecard_id)),
            )
            conn.commit()
        if cursor.rowcount == 0:
            return JSONResponse(
                status_code=409,
                content={
                    "code": "SCORECARD_RACED",
                    "detail": "scorecard was consumed by a concurrent request",
                    "your_intent_id": str(intent.intent_id),
                },
            )
    return response


@app.post("/intents", response_model=None)
def create_intent(
    intent: OrderIntent, x_live_unlock: str = Header(default="")
) -> JSONResponse | dict[str, object]:
    with connect() as conn:
        existing = conn.execute(
            "select payload_json, decision_json, execution_json, status "
            "from intents where idempotency_key = ?",
            (intent.idempotency_key,),
        ).fetchone()
    if existing is not None:
        return _idempotent_response(existing)

    if intent.mode == "live":
        unlock_error = _consume_live_unlock_or_error(
            str(x_live_unlock), intent.actor, dry=True, intent_id=intent.intent_id
        )
        if unlock_error is not None:
            return unlock_error

    base_qty: str | None = None
    price_str: str | None = None
    if intent.quantity.kind == "quote":
        requested_notional = intent.quantity.value
    else:
        base_qty, price_str = _resolve_qty(intent, _market_url())
        requested_notional = intent.quantity.value * Decimal(price_str)

    exposure_response = _exposure_limit_response(
        intent, _current_exposure(intent.symbol, _today()), requested_notional
    )
    if exposure_response is not None:
        return exposure_response

    risk_url = os.getenv("RISK_ENGINE_URL", "http://risk-engine:8081")
    risk_response = httpx.post(
        f"{risk_url}/validate",
        content=intent.model_dump_json(),
        headers={"content-type": "application/json"},
        timeout=5.0,
    )
    risk_response.raise_for_status()
    decision = RiskDecision.model_validate(risk_response.json())

    try:
        if not decision.approved:
            _persist(intent, decision, None, "rejected")
            return _rejected_response(decision)
        if decision.requires_confirmation:
            _persist(intent, decision, None, "pending_confirmation")
            return _pending_response(intent, decision)

        if base_qty is None:
            base_qty, _ = _resolve_qty(intent, _market_url())
        if intent.mode == "live":
            unlock_error = _consume_live_unlock_or_error(
                str(x_live_unlock), intent.actor, dry=False, intent_id=intent.intent_id
            )
            if unlock_error is not None:
                return unlock_error
        execution = _call_execution(intent, decision, base_qty)
        _persist(intent, decision, execution, "executed")
        _after_fill_side_effects(execution, intent)
        return {
            "status": "executed",
            "intent": intent,
            "decision": decision,
            "execution": execution,
        }
    except DuplicateIntentIdError:
        return JSONResponse(status_code=409, content={"code": "DUPLICATE_INTENT_ID"})


@app.get("/intents")
def list_intents(
    limit: int = Query(default=20, ge=1),
    offset: int = Query(default=0, ge=0),
    mode: str | None = None,
) -> dict[str, object]:
    clamped_limit = min(limit, 100)
    params: list[object] = []
    where = ""
    if mode is not None:
        where = " where json_extract(payload_json, '$.mode') = ?"
        params.append(mode)
    with connect() as conn:
        total = conn.execute(f"select count(*) from intents{where}", params).fetchone()[0]
        rows = conn.execute(
            f"""
            select payload_json, decision_json, execution_json, status from intents{where}
            order by created_at desc limit ? offset ?
            """,
            [*params, clamped_limit, offset],
        ).fetchall()
    return {"items": [_row_to_item(row) for row in rows], "total": total}


@app.get("/exposure")
def get_exposure(date: str | None = None) -> dict[str, object]:
    target_date = date or _today()
    symbols: dict[str, dict[str, str]] = {}
    with connect() as conn:
        rows = conn.execute(
            "select symbol, side, sum(cast(notional_usdt as real)) as total "
            "from daily_fills where date = ? group by symbol, side",
            (target_date,),
        ).fetchall()
    totals: dict[str, dict[str, Decimal]] = {}
    for row in rows:
        symbol = str(row["symbol"])
        side = str(row["side"])
        total = Decimal(str(row["total"]))
        bucket = totals.setdefault(
            symbol,
            {"side_buy": Decimal("0"), "side_sell": Decimal("0"), "total": Decimal("0")},
        )
        if side == "buy":
            bucket["side_buy"] += total
        elif side == "sell":
            bucket["side_sell"] += total
        bucket["total"] += total
    for symbol, values in totals.items():
        symbols[symbol] = {key: str(value) for key, value in values.items()}
    return {"date": target_date, "limit_usdt": str(_daily_limit()), "symbols": symbols}


@app.post("/intents/{intent_id}/confirm", response_model=None)
def confirm_intent(
    intent_id: UUID,
    request: ConfirmationRequest,
    x_live_unlock: str = Header(default=""),
) -> JSONResponse | dict[str, object]:
    if request.intent_id != intent_id:
        return JSONResponse(status_code=400, content={"code": "INTENT_ID_MISMATCH"})
    with connect() as conn:
        row = conn.execute(
            "select payload_json, decision_json, execution_json, status "
            "from intents where intent_id=?",
            (str(intent_id),),
        ).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={"code": "INTENT_NOT_FOUND"})

    if row["status"] == "canceled":
        return JSONResponse(status_code=410, content={"code": "INTENT_CANCELED"})
    intent = OrderIntent.model_validate_json(row["payload_json"])
    decision = RiskDecision.model_validate_json(row["decision_json"])
    if not decision.requires_confirmation:
        return JSONResponse(status_code=409, content={"code": "CONFIRMATION_NOT_REQUIRED"})
    if row["execution_json"] is not None:
        return JSONResponse(status_code=409, content={"code": "ALREADY_EXECUTED"})
    if request.confirmation_token != decision.confirmation_token:
        return JSONResponse(status_code=403, content={"code": "INVALID_CONFIRMATION_TOKEN"})
    if decision.confirmation_expires_at is not None and decision.confirmation_expires_at < _now():
        return JSONResponse(status_code=410, content={"code": "CONFIRMATION_EXPIRED"})

    if intent.mode == "live":
        unlock_error = _consume_live_unlock_or_error(
            str(x_live_unlock), intent.actor, dry=False, intent_id=intent.intent_id
        )
        if unlock_error is not None:
            return unlock_error
    base_qty, _ = _resolve_qty(intent, _market_url())
    execution = _call_execution(intent, decision, base_qty)
    with connect() as conn:
        conn.execute(
            "update intents set execution_json = ?, status = ? where intent_id = ?",
            (execution.model_dump_json(), "executed", str(intent_id)),
        )
        conn.commit()
    _after_fill_side_effects(execution, intent)
    return {"intent": intent, "decision": decision, "execution": execution}


@app.delete("/intents/{intent_id}", response_model=None)
def cancel_intent(intent_id: UUID) -> Response | JSONResponse:
    with connect() as conn:
        row = conn.execute(
            "select payload_json, decision_json, execution_json, status "
            "from intents where intent_id = ?",
            (str(intent_id),),
        ).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={"code": "INTENT_NOT_FOUND"})

    status = str(row["status"])
    if status == "pending_confirmation":
        with connect() as conn:
            conn.execute(
                "update intents set status = ? where intent_id = ?",
                ("canceled", str(intent_id)),
            )
            conn.commit()
        return Response(status_code=204)

    if status == "executed" and row["execution_json"] is not None:
        execution = ExecutionResult.model_validate_json(row["execution_json"])
        intent = OrderIntent.model_validate_json(row["payload_json"])
        if execution.status == "open" and intent.mode == "live" and execution.venue_order_id:
            canceled = _call_cancel(intent, execution)
            with connect() as conn:
                conn.execute(
                    "update intents set execution_json = ? where intent_id = ?",
                    (canceled.model_dump_json(), str(intent_id)),
                )
                conn.commit()
            _after_fill_side_effects(canceled, intent)
            return JSONResponse(
                status_code=200,
                content=jsonable_encoder(
                    {
                        "status": status,
                        "intent": intent,
                        "decision": RiskDecision.model_validate_json(row["decision_json"]),
                        "execution": canceled,
                    }
                ),
            )

    return JSONResponse(
        status_code=409, content={"code": "CANNOT_CANCEL", "current_status": status}
    )


@app.get("/paper/positions", response_model=None)
def get_paper_positions(actor: str | None = None) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    with connect() as conn:
        rows = conn.execute(
            "select symbol, qty, avg_cost, total_cost, realized_pnl from paper_positions "
            "where actor = ? order by symbol",
            (actor,),
        ).fetchall()

    mark_cache: dict[str, tuple[Decimal | None, str | None]] = {}
    positions: list[dict[str, object]] = []
    for row in rows:
        qty = Decimal(row["qty"])
        avg_cost = Decimal(row["avg_cost"])
        total_cost = Decimal(row["total_cost"])
        realized_pnl = Decimal(row["realized_pnl"])
        mark_price: Decimal | None = None
        mark_source: str | None = None
        mark_value: Decimal | None = None
        unrealized_pnl: Decimal | None = None
        if qty > 0:
            symbol = str(row["symbol"])
            mark_cache.setdefault(symbol, _mark_for_symbol(symbol))
            mark_price, mark_source = mark_cache[symbol]
            if mark_price is not None:
                mark_value = _q8(qty * mark_price)
                unrealized_pnl = _q8(qty * (mark_price - avg_cost))
        positions.append(
            {
                "symbol": row["symbol"],
                "qty": _q8s(qty),
                "avg_cost": _q8s(avg_cost),
                "total_cost": _q8s(total_cost),
                "realized_pnl": _q8s(realized_pnl),
                "mark_price": _q8s(mark_price) if mark_price is not None else None,
                "mark_value": _q8s(mark_value) if mark_value is not None else None,
                "unrealized_pnl": _q8s(unrealized_pnl) if unrealized_pnl is not None else None,
                "mark_source": mark_source,
            }
        )
    return {"actor": actor, "positions": positions}


@app.post("/intents/{intent_id}/refresh", response_model=None)
def refresh_intent(intent_id: UUID) -> JSONResponse | dict[str, object]:
    with connect() as conn:
        row = conn.execute(
            "select payload_json, decision_json, execution_json, status "
            "from intents where intent_id=?",
            (str(intent_id),),
        ).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={"code": "INTENT_NOT_FOUND"})

    item = _row_to_item(row)
    intent = item["intent"]
    execution = item["execution"]

    if (
        row["status"] == "executed"
        and isinstance(intent, OrderIntent)
        and isinstance(execution, ExecutionResult)
        and execution.status == "open"
        and intent.mode == "live"
        and execution.venue_order_id
    ):
        refreshed = _call_refresh(intent, execution)
        if refreshed.status != "open":
            with connect() as conn:
                conn.execute(
                    "update intents set execution_json = ? where intent_id = ?",
                    (refreshed.model_dump_json(), str(intent_id)),
                )
                conn.commit()
            _after_fill_side_effects(refreshed, intent)
            item["execution"] = refreshed

    return item


@app.get("/intents/{intent_id}", response_model=None)
def get_intent(intent_id: UUID) -> JSONResponse | dict[str, object]:
    with connect() as conn:
        row = conn.execute(
            "select payload_json, decision_json, execution_json, status "
            "from intents where intent_id=?",
            (str(intent_id),),
        ).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={"code": "INTENT_NOT_FOUND"})
    return _row_to_item(row)


def _persist(
    intent: OrderIntent, decision: RiskDecision, execution: ExecutionResult | None, status: str
) -> None:
    try:
        with connect() as conn:
            conn.execute(
                """
                insert into intents
                (intent_id,idempotency_key,payload_json,decision_json,execution_json,created_at,status)
                values(?,?,?,?,?,?,?)
                """,
                (
                    str(intent.intent_id),
                    intent.idempotency_key,
                    intent.model_dump_json(),
                    decision.model_dump_json(),
                    execution.model_dump_json() if execution else None,
                    _now().isoformat(),
                    status,
                ),
            )
            conn.commit()
    except sqlite3.IntegrityError as exc:
        if "intent_id" in str(exc):
            raise DuplicateIntentIdError from exc
        raise
