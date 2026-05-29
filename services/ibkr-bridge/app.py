from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ibkr_client import IBKRClient, IBKRConfig, PlaceOrderRequest

logger = logging.getLogger(__name__)


class BridgeOrderResponse(BaseModel):
    id: str
    status: Literal["pending", "submitted", "filled", "partial", "canceled", "rejected", "error"]
    fills: list[dict[str, str]]
    avg_price: str | None
    filled_qty: str
    remaining_qty: str
    error: str | None
    raw_order_ref: str | None


class TickerResponse(BaseModel):
    symbol: str
    price: str
    source: str = "ibkr"


def _config_from_env() -> IBKRConfig:
    return IBKRConfig(
        host=os.getenv("IBKR_GATEWAY_HOST", "host.docker.internal"),
        port=int(os.getenv("IBKR_GATEWAY_PORT", "4002")),
        client_id=int(os.getenv("IBKR_CLIENT_ID", "1")),
        timeout_sec=float(os.getenv("IBKR_CONNECT_TIMEOUT_SEC", "10")),
        allow_live_port=os.getenv("IBKR_ALLOW_LIVE_PORT", "false").lower() == "true",
    )


client = IBKRClient(_config_from_env())


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        client.connect()
    except RuntimeError as exc:
        logger.error("IBKR bridge startup failed: %s", exc)
        raise
    except Exception:
        # healthz stays up; readyz exposes disconnected state until Gateway/TWS appears.
        pass
    try:
        yield
    finally:
        client.disconnect()


app = FastAPI(title="ibkr-bridge", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz", response_model=None)
def readyz() -> JSONResponse | dict[str, str]:
    if not client.is_ready():
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return {"status": "ready"}


def _ensure_ready() -> None:
    if not client.is_ready():
        raise HTTPException(status_code=503, detail={"code": "IBKR_NOT_READY"})


@app.post("/orders", response_model=BridgeOrderResponse)
def place_order(request: PlaceOrderRequest) -> dict[str, object]:
    _ensure_ready()
    try:
        return client.place_order(request)
    except ValueError as exc:
        code = str(exc)
        if code == "INVALID_SYMBOL":
            raise HTTPException(status_code=400, detail={"code": code}) from exc
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_ORDER", "message": code},
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"code": "IBKR_ORDER_FAILED"}) from exc


@app.get("/orders/{order_id}", response_model=BridgeOrderResponse)
def get_order(order_id: str) -> dict[str, object]:
    _ensure_ready()
    payload = client.get_order(order_id)
    if payload is None:
        raise HTTPException(status_code=404, detail={"code": "ORDER_NOT_FOUND"})
    return payload


@app.delete("/orders/{order_id}", response_model=BridgeOrderResponse)
def cancel_order(order_id: str) -> dict[str, object]:
    _ensure_ready()
    payload = client.cancel_order(order_id)
    if payload is None:
        raise HTTPException(status_code=404, detail={"code": "ORDER_NOT_FOUND"})
    return payload


@app.get("/tickers/{symbol}", response_model=TickerResponse)
def ticker(symbol: str = Path(min_length=1)) -> dict[str, str]:
    _ensure_ready()
    try:
        return client.ticker(symbol)
    except ValueError as exc:
        code = str(exc)
        if code == "INVALID_SYMBOL":
            raise HTTPException(status_code=400, detail={"code": code}) from exc
        raise HTTPException(status_code=503, detail={"code": "MARKET_DATA_UNAVAILABLE"}) from exc
