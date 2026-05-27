from __future__ import annotations

import json
import os
import re
import threading
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

import httpx
from db import connect, db_path
from fastapi import FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="analysis-adapter", version="0.1.0")

TA_BRIDGE_URL = os.getenv("TA_BRIDGE_URL", "http://tradingagents-bridge:18181")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8080")
TA_DEFAULT_PROVIDER = os.getenv("TA_DEFAULT_PROVIDER", "deepseek")
TA_TIMEOUT_SEC = float(os.getenv("TA_TIMEOUT_SEC", "900"))
SCORECARD_TTL_MIN = int(os.getenv("ANALYSIS_SCORECARD_TTL_MIN", "60"))
JobStatus = Literal["queued", "running", "succeeded", "failed"]
AnalystName = Literal["market", "social", "news", "fundamentals"]

CRYPTO_NAME_TO_SYMBOL = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "ether": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "sol": "SOL",
    "binancecoin": "BNB",
    "bnb": "BNB",
    "xrp": "XRP",
    "ripple": "XRP",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    "cardano": "ADA",
    "ada": "ADA",
}
QUOTE_ASSETS = ("USDT", "USDC", "USD", "BUSD")


def _normalize_ta_ticker(symbol: str, asset_type: str) -> str:
    """Convert user/trading crypto symbols to TradingAgents-friendly tickers."""
    cleaned = symbol.strip()
    if asset_type != "crypto":
        return cleaned.upper()
    compact = re.sub(r"[^A-Za-z0-9]", "", cleaned).upper()
    if not compact:
        return cleaned
    base = CRYPTO_NAME_TO_SYMBOL.get(cleaned.strip().lower(), compact)
    for quote in QUOTE_ASSETS:
        if compact.endswith(quote) and len(compact) > len(quote):
            base = compact[: -len(quote)]
            break
    if base.endswith("USD") and "-" not in cleaned:
        base = base[:-3]
    return f"{base}-USD"


def _default_analysts() -> list[AnalystName]:
    return ["market", "news"]


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    asset_type: Literal["stock", "crypto"] = "crypto"
    analysts: list[AnalystName] = Field(default_factory=_default_analysts)
    provider: str | None = None
    dry_run: bool = False


AnalyzeRequest.model_rebuild(_types_namespace={"Literal": Literal, "AnalystName": AnalystName})


def _now() -> datetime:
    return datetime.now(UTC)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    _ = request
    return JSONResponse(status_code=400, content=jsonable_encoder({"detail": exc.errors()}))


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    db_path()
    return {"status": "ready"}


@app.post("/analyze", response_model=None)
def analyze(req: AnalyzeRequest) -> dict[str, object]:
    job_id = str(uuid4())
    requested_at = _now().isoformat()
    with connect() as conn:
        conn.execute(
            "insert into analysis_jobs "
            "(job_id, actor, symbol, asset_type, requested_at, status) "
            "values (?,?,?,?,?,?)",
            (job_id, req.actor, req.symbol, req.asset_type, requested_at, "queued"),
        )
        conn.commit()
    worker = threading.Thread(
        target=_run_analysis_job,
        args=(job_id, req),
        daemon=True,
        name=f"analysis-{job_id[:8]}",
    )
    worker.start()
    return {"job_id": job_id, "status": "queued", "requested_at": requested_at}


@app.get("/jobs/{job_id}", response_model=None)
def get_job(job_id: UUID) -> JSONResponse | dict[str, object]:
    with connect() as conn:
        row = conn.execute(
            "select actor, symbol, asset_type, requested_at, finished_at, "
            "status, scorecard_id, error from analysis_jobs where job_id = ?",
            (str(job_id),),
        ).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={"code": "JOB_NOT_FOUND"})
    return {
        "job_id": str(job_id),
        "actor": row["actor"],
        "symbol": row["symbol"],
        "asset_type": row["asset_type"],
        "requested_at": row["requested_at"],
        "finished_at": row["finished_at"],
        "status": row["status"],
        "scorecard_id": row["scorecard_id"],
        "error": row["error"],
    }


@app.get("/jobs", response_model=None)
def list_jobs(
    actor: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, object]:
    clauses: list[str] = []
    params: list[object] = []
    if actor:
        clauses.append("actor = ?")
        params.append(actor)
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = (" where " + " and ".join(clauses)) if clauses else ""
    with connect() as conn:
        rows = conn.execute(
            f"select job_id, actor, symbol, status, scorecard_id, requested_at, finished_at "
            f"from analysis_jobs{where} order by requested_at desc limit ?",
            [*params, limit],
        ).fetchall()
    return {
        "items": [
            {
                "job_id": row["job_id"],
                "actor": row["actor"],
                "symbol": row["symbol"],
                "status": row["status"],
                "scorecard_id": row["scorecard_id"],
                "requested_at": row["requested_at"],
                "finished_at": row["finished_at"],
            }
            for row in rows
        ]
    }


def _run_analysis_job(job_id: str, req: AnalyzeRequest) -> None:
    """Worker thread entrypoint. Never raises."""
    _mark_status(job_id, "running")
    try:
        raw = _stub_ta_response(req) if req.dry_run else _call_ta_bridge(req)
        try:
            scorecard_payload = _translate_to_scorecard_payload(req, raw)
            scorecard_id = _post_scorecard(scorecard_payload)
            _mark_success(job_id, scorecard_id, raw)
        except Exception as exc:  # noqa: BLE001
            _mark_failure(job_id, str(exc)[:500], raw)
    except Exception as exc:  # noqa: BLE001
        _mark_failure(job_id, str(exc)[:500])


def _call_ta_bridge(req: AnalyzeRequest) -> dict[str, object]:
    payload: dict[str, object] = {
        "ticker": _normalize_ta_ticker(req.symbol, req.asset_type),
        "date": _now().strftime("%Y-%m-%d"),
        "provider": req.provider or TA_DEFAULT_PROVIDER,
        "asset_type": req.asset_type,
        "analysts": list(req.analysts),
        "dry_run": False,
    }
    response = httpx.post(
        f"{TA_BRIDGE_URL}/analyze",
        json=payload,
        timeout=TA_TIMEOUT_SEC,
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("TA bridge returned non-dict body")
    if not body.get("ok", False):
        raise ValueError(f"TA bridge reported failure: {body.get('error', 'unknown')}")
    return body


def _stub_ta_response(req: AnalyzeRequest) -> dict[str, object]:
    """Deterministic stub used when dry_run=True. Bypasses real TA."""
    return {
        "ok": True,
        "dry_run": True,
        "ticker": _normalize_ta_ticker(req.symbol, req.asset_type),
        "provider": req.provider or TA_DEFAULT_PROVIDER,
        "decision": "BUY",
        "final_trade_decision": "FINAL TRANSACTION PROPOSAL: **BUY**",
        "reports": {
            "market": "(dry-run stub) bullish technicals",
            "sentiment": "(dry-run stub) positive social",
            "news": "(dry-run stub) no negative news",
            "fundamentals": "(dry-run stub) healthy on-chain metrics",
        },
    }


def _translate_to_scorecard_payload(
    req: AnalyzeRequest, raw: dict[str, object]
) -> dict[str, object]:
    decision = str(raw.get("decision", "")).strip().upper()
    if decision not in {"BUY", "HOLD", "SELL"}:
        final_text = str(raw.get("final_trade_decision", ""))
        for token in ("BUY", "SELL", "HOLD"):
            if f"**{token}**" in final_text.upper():
                decision = token
                break
    if decision not in {"BUY", "HOLD", "SELL"}:
        raise ValueError("Could not extract a Buy/Hold/Sell action from TA output")

    action = decision.lower()
    reports = raw.get("reports", {}) or {}
    if not isinstance(reports, dict):
        reports = {}
    conviction = _derive_conviction(action, reports)
    metadata = {f"report_{k}": _truncate(str(v), 2000) for k, v in reports.items() if v}
    metadata["ta_decision"] = decision
    metadata["asset_type"] = req.asset_type
    metadata["provider"] = str(raw.get("provider", req.provider or TA_DEFAULT_PROVIDER))

    thesis_parts: list[str] = []
    if isinstance(reports.get("market"), str):
        thesis_parts.append("Market: " + _truncate(str(reports["market"]), 600))
    if isinstance(reports.get("news"), str):
        thesis_parts.append("News: " + _truncate(str(reports["news"]), 600))
    thesis = " | ".join(thesis_parts) or f"TradingAgents decided {decision} for {req.symbol}"

    return {
        "actor": req.actor,
        "symbol": req.symbol,
        "action": action,
        "source": "tradingagents",
        "conviction": f"{conviction:.4f}",
        "thesis": _truncate(thesis, 3900),
        "entry_low": None,
        "entry_high": None,
        "stop_loss": None,
        "take_profit": None,
        "time_horizon": "swing",
        "ttl_minutes": SCORECARD_TTL_MIN,
        "metadata": metadata,
    }


def _derive_conviction(action: str, reports: dict[str, object]) -> float:
    if action == "hold":
        return 0.30
    score = 0.50
    for key in ("market", "social", "news", "fundamentals", "sentiment"):
        value = reports.get(key)
        if isinstance(value, str) and value.strip():
            score += 0.10
    return min(0.90, score)


def _truncate(value: str, max_len: int) -> str:
    return value if len(value) <= max_len else value[: max_len - 1] + "..."


def _post_scorecard(payload: dict[str, object]) -> str:
    response = httpx.post(
        f"{ORCHESTRATOR_URL}/scorecards",
        json=payload,
        timeout=10.0,
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or "scorecard_id" not in body:
        raise ValueError(f"Orchestrator did not return a scorecard_id: {body}")
    return str(body["scorecard_id"])


def _mark_status(job_id: str, status: JobStatus) -> None:
    with connect() as conn:
        conn.execute("update analysis_jobs set status = ? where job_id = ?", (status, job_id))
        conn.commit()


def _mark_success(job_id: str, scorecard_id: str, raw: dict[str, object]) -> None:
    with connect() as conn:
        conn.execute(
            "update analysis_jobs set status = 'succeeded', scorecard_id = ?, "
            "raw_response_json = ?, finished_at = ? where job_id = ?",
            (scorecard_id, json.dumps(raw, default=str), _now().isoformat(), job_id),
        )
        conn.commit()


def _mark_failure(job_id: str, error: str, raw: dict[str, object] | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "update analysis_jobs set status = 'failed', error = ?, "
            "raw_response_json = ?, finished_at = ? where job_id = ?",
            (
                error,
                json.dumps(raw, default=str) if raw is not None else None,
                _now().isoformat(),
                job_id,
            ),
        )
        conn.commit()
