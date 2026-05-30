from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from typing import Any, cast

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from exchange_client import ExchangeClient, ExchangeName, SUPPORTED_EXCHANGES

logger = logging.getLogger(__name__)
REQUEST_TIMEOUT_SEC = float(os.getenv("EXCHANGE_TIMEOUT_SEC", "10"))


class BalanceItem(BaseModel):
    exchange: ExchangeName
    asset: str
    free: str
    used: str
    total: str


class BalancesResponse(BaseModel):
    balances: list[BalanceItem]
    exchanges: dict[ExchangeName, bool]
    errors: dict[ExchangeName, str] = Field(default_factory=dict)


client = ExchangeClient.from_env()
app = FastAPI(title="exchange-bridge", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


async def _with_timeout(func: Callable[..., Any], *args: object) -> object:
    return await asyncio.wait_for(
        asyncio.to_thread(func, *args),
        timeout=REQUEST_TIMEOUT_SEC,
    )


def _error_code(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "TIMEOUT"
    return type(exc).__name__


@app.get("/readyz", response_model=None)
async def readyz() -> JSONResponse | dict[str, object]:
    exchanges: dict[ExchangeName, bool] = {
        exchange_name: False for exchange_name in SUPPORTED_EXCHANGES
    }
    configured = client.configured_exchanges()
    if not configured:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "exchanges": exchanges},
        )

    for exchange in configured:
        try:
            exchanges[exchange] = bool(await _with_timeout(client.is_ready, exchange))
        except Exception as exc:
            logger.warning(
                "exchange readiness probe failed exchange=%s error=%s",
                exchange,
                type(exc).__name__,
            )
            exchanges[exchange] = False

    status = "ready" if any(exchanges.values()) else "not_ready"
    status_code = 200 if status == "ready" else 503
    return JSONResponse(status_code=status_code, content={"status": status, "exchanges": exchanges})


@app.get("/balances", response_model=BalancesResponse)
async def balances(
    exchange: ExchangeName | None = Query(default=None),
) -> dict[str, object]:
    exchanges: dict[ExchangeName, bool] = {
        exchange_name: False for exchange_name in SUPPORTED_EXCHANGES
    }
    errors: dict[ExchangeName, str] = {}
    requested = (exchange,) if exchange is not None else SUPPORTED_EXCHANGES
    payload: list[dict[str, str]] = []

    for exchange_name in requested:
        if not client.is_configured(exchange_name):
            continue
        try:
            balances_for_exchange = await _with_timeout(client.fetch_balances, exchange_name)
        except Exception as exc:
            logger.warning(
                "exchange balance fetch failed exchange=%s error=%s",
                exchange_name,
                type(exc).__name__,
            )
            exchanges[exchange_name] = False
            errors[exchange_name] = _error_code(exc)
            continue
        exchanges[exchange_name] = True
        payload.extend(cast(list[dict[str, str]], balances_for_exchange))

    return {"balances": payload, "exchanges": exchanges, "errors": errors}
