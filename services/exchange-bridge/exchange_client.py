from __future__ import annotations

import base64
import hashlib
import hmac
import importlib
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Literal
from urllib.parse import urlencode

import httpx

ExchangeName = Literal["binance", "okx", "bybit"]
SUPPORTED_EXCHANGES: tuple[ExchangeName, ...] = ("binance", "okx", "bybit")
CASH_LIKE_ASSETS = {"USDT", "USDC", "USD", "FDUSD", "BUSD", "TUSD", "DAI"}


@dataclass(frozen=True)
class ExchangeCredentials:
    api_key: str
    api_secret: str
    passphrase: str = ""

    def is_configured(self, exchange: ExchangeName) -> bool:
        if not self.api_key or not self.api_secret:
            return False
        if exchange == "okx" and not self.passphrase:
            return False
        return True


@dataclass
class AssetAggregate:
    free: Decimal = Decimal("0")
    used: Decimal = Decimal("0")
    total: Decimal = Decimal("0")
    usd_value: Decimal = Decimal("0")
    sources: defaultdict[str, Decimal] = field(default_factory=lambda: defaultdict(Decimal))


def credentials_from_env() -> dict[ExchangeName, ExchangeCredentials]:
    return {
        "binance": ExchangeCredentials(
            api_key=os.getenv("EXCHANGE_API_KEY", ""),
            api_secret=os.getenv("EXCHANGE_API_SECRET", ""),
        ),
        "okx": ExchangeCredentials(
            api_key=os.getenv("OKX_API_KEY", ""),
            api_secret=os.getenv("OKX_API_SECRET", ""),
            passphrase=os.getenv("OKX_API_PASSPHRASE", ""),
        ),
        "bybit": ExchangeCredentials(
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
        ),
    }


def _decimal(value: object) -> Decimal:
    if value is None:
        return Decimal("0")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def _decimal_str(value: object) -> str:
    return format(_decimal(value).normalize(), "f")


def _is_cash_like(asset: str) -> bool:
    return asset.upper() in CASH_LIKE_ASSETS


def _normalize_binance_asset(asset: object) -> tuple[str, str | None]:
    symbol = str(asset or "").upper()
    if symbol.startswith("LD") and len(symbol) > 2:
        return symbol[2:], symbol
    return symbol, None


class ExchangeClient:
    def __init__(
        self,
        credentials: dict[ExchangeName, ExchangeCredentials],
        *,
        timeout_sec: float = 10.0,
        min_usd_detail: Decimal = Decimal("10"),
        ccxt_module: Any | None = None,
        http_module: Any | None = None,
    ) -> None:
        self._credentials = credentials
        self._timeout_sec = timeout_sec
        self._timeout_ms = int(timeout_sec * 1000)
        self._min_usd_detail = min_usd_detail
        self._ccxt: Any | None = ccxt_module if ccxt_module is not None else self._load_ccxt()
        self._http: Any = http_module if http_module is not None else httpx
        self._clients: dict[ExchangeName, Any] = {}
        self._build_clients()

    @classmethod
    def from_env(cls) -> ExchangeClient:
        return cls(
            credentials_from_env(),
            timeout_sec=float(os.getenv("EXCHANGE_TIMEOUT_SEC", "10")),
            min_usd_detail=Decimal(os.getenv("EXCHANGE_MIN_USD_DETAIL", "10")),
        )

    def configured_exchanges(self) -> list[ExchangeName]:
        return [
            exchange
            for exchange in SUPPORTED_EXCHANGES
            if self._credentials[exchange].is_configured(exchange)
        ]

    def is_configured(self, exchange: ExchangeName) -> bool:
        self._validate_exchange(exchange)
        return self._credentials[exchange].is_configured(exchange)

    def is_ready(self, exchange: ExchangeName) -> bool:
        self._validate_exchange(exchange)
        if not self.is_configured(exchange):
            return False
        try:
            self.fetch_balance_payload(exchange)
        except Exception:
            return False
        return True

    def fetch_balances(self, exchange: ExchangeName) -> list[dict[str, str]]:
        payload = self.fetch_balance_payload(exchange)
        balances = payload["balances"]
        if not isinstance(balances, list):
            raise RuntimeError("EXCHANGE_BALANCE_UNAVAILABLE")
        return balances

    def fetch_balance_payload(self, exchange: ExchangeName) -> dict[str, object]:
        self._validate_exchange(exchange)
        if exchange == "binance":
            return self._fetch_binance_balance_payload()
        if exchange == "okx":
            return self._fetch_okx_balance_payload()
        raw = self._raw_balance(exchange)
        return self._normalize_ccxt_balance_payload(exchange, raw)

    @staticmethod
    def _load_ccxt() -> Any | None:
        try:
            return importlib.import_module("ccxt")
        except ImportError:
            return None

    def _build_clients(self) -> None:
        if self._ccxt is None:
            return
        for exchange in SUPPORTED_EXCHANGES:
            credentials = self._credentials[exchange]
            if not credentials.is_configured(exchange):
                continue
            config: dict[str, object] = {
                "apiKey": credentials.api_key,
                "secret": credentials.api_secret,
                "timeout": self._timeout_ms,
                "enableRateLimit": True,
            }
            if exchange == "okx":
                config["password"] = credentials.passphrase
            factory = getattr(self._ccxt, exchange)
            self._clients[exchange] = factory(config)

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, object] | None = None,
    ) -> object:
        response = self._http.request(
            method,
            url,
            headers=headers,
            params=params,
            timeout=self._timeout_sec,
        )
        response.raise_for_status()
        return response.json()

    def _binance_signed_json(
        self,
        base_url: str,
        path: str,
        *,
        method: str = "GET",
        params: dict[str, object] | None = None,
    ) -> object:
        credentials = self._credentials["binance"]
        merged: dict[str, object] = {
            "recvWindow": "5000",
            "timestamp": str(int(time.time() * 1000)),
        }
        if params:
            merged.update(params)
        query = urlencode(merged)
        signature = hmac.new(
            credentials.api_secret.encode(),
            query.encode(),
            hashlib.sha256,
        ).hexdigest()
        url = f"{base_url}{path}?{query}&signature={signature}"
        return self._request_json(
            method,
            url,
            headers={
                "X-MBX-APIKEY": credentials.api_key,
                "user-agent": "aegis-exchange-bridge/1.0",
            },
        )

    def _try_binance_signed_json(
        self,
        base_url: str,
        path: str,
        *,
        method: str = "GET",
        params: dict[str, object] | None = None,
    ) -> object | None:
        try:
            return self._binance_signed_json(base_url, path, method=method, params=params)
        except Exception:
            return None

    def _fetch_binance_price_map(self) -> dict[str, Decimal]:
        payload = self._request_json("GET", "https://api.binance.com/api/v3/ticker/price")
        if not isinstance(payload, list):
            return {}
        prices: dict[str, Decimal] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or "")
            price = _decimal(item.get("price"))
            if symbol and price > 0:
                prices[symbol] = price
        return prices

    def _binance_usd_price(self, asset: str, prices: dict[str, Decimal]) -> Decimal:
        symbol = asset.upper()
        if _is_cash_like(symbol):
            return Decimal("1")
        for quote in ("USDT", "FDUSD", "USDC", "BUSD", "TUSD"):
            price = prices.get(f"{symbol}{quote}")
            if price:
                return price
        btc_usd = prices.get("BTCUSDT") or prices.get("BTCFDUSD") or Decimal("0")
        eth_usd = prices.get("ETHUSDT") or prices.get("ETHFDUSD") or Decimal("0")
        via_btc = prices.get(f"{symbol}BTC")
        if via_btc and btc_usd:
            return via_btc * btc_usd
        via_eth = prices.get(f"{symbol}ETH")
        if via_eth and eth_usd:
            return via_eth * eth_usd
        return Decimal("0")

    def _add_asset(
        self,
        aggregates: dict[str, AssetAggregate],
        *,
        raw_asset: object,
        wallet: str,
        total: object,
        free: object = None,
        used: object = None,
        usd_value: object = None,
        price: Decimal = Decimal("0"),
    ) -> None:
        asset = str(raw_asset or "").upper()
        amount = _decimal(total)
        if not asset or amount <= 0:
            return
        free_amount = _decimal(free) if free is not None else amount
        used_amount = _decimal(used) if used is not None else Decimal("0")
        value = _decimal(usd_value)
        if value <= 0 and price > 0:
            value = amount * price
        aggregate = aggregates.setdefault(asset, AssetAggregate())
        aggregate.free += free_amount
        aggregate.used += used_amount
        aggregate.total += amount
        aggregate.usd_value += value
        aggregate.sources[wallet] += amount

    def _finalize_balance_payload(
        self,
        exchange: ExchangeName,
        aggregates: dict[str, AssetAggregate],
    ) -> dict[str, object]:
        rows: list[dict[str, str]] = []
        total_usd = Decimal("0")
        visible_usd = Decimal("0")
        hidden_usd = Decimal("0")
        hidden_count = 0
        total_assets = 0
        for asset, aggregate in aggregates.items():
            if aggregate.total <= 0:
                continue
            total_assets += 1
            total_usd += aggregate.usd_value
            if aggregate.usd_value > 0 and aggregate.usd_value < self._min_usd_detail:
                hidden_usd += aggregate.usd_value
                hidden_count += 1
                continue
            visible_usd += aggregate.usd_value
            source_parts = [
                f"{name}:{_decimal_str(amount)}"
                for name, amount in sorted(aggregate.sources.items())
                if amount > 0
            ]
            rows.append(
                {
                    "exchange": exchange,
                    "asset": asset,
                    "free": _decimal_str(aggregate.free),
                    "used": _decimal_str(aggregate.used),
                    "total": _decimal_str(aggregate.total),
                    "usd_value": _decimal_str(aggregate.usd_value),
                    "sources": ", ".join(source_parts),
                }
            )
        return {
            "balances": sorted(
                rows,
                key=lambda item: (-_decimal(item["usd_value"]), item["asset"]),
            ),
            "summary": {
                "total_usd": _decimal_str(total_usd),
                "visible_usd": _decimal_str(visible_usd),
                "hidden_usd": _decimal_str(hidden_usd),
                "hidden_count": str(hidden_count),
                "total_assets": str(total_assets),
                "visible_assets": str(len(rows)),
                "min_usd_detail": _decimal_str(self._min_usd_detail),
            },
        }

    def _fetch_binance_balance_payload(self) -> dict[str, object]:
        if not self.is_configured("binance"):
            raise RuntimeError("EXCHANGE_NOT_CONFIGURED")
        spot = self._binance_signed_json(
            "https://api.binance.com",
            "/api/v3/account",
            params={"omitZeroBalances": "true"},
        )
        funding = self._try_binance_signed_json(
            "https://api.binance.com",
            "/sapi/v1/asset/get-funding-asset",
            method="POST",
        )
        futures = self._try_binance_signed_json(
            "https://fapi.binance.com",
            "/fapi/v2/balance",
        )
        earn_flex = self._try_binance_signed_json(
            "https://api.binance.com",
            "/sapi/v1/simple-earn/flexible/position",
        )
        earn_locked = self._try_binance_signed_json(
            "https://api.binance.com",
            "/sapi/v1/simple-earn/locked/position",
        )
        coinm = self._try_binance_signed_json(
            "https://dapi.binance.com",
            "/dapi/v1/balance",
        )
        dual = self._try_binance_signed_json(
            "https://api.binance.com",
            "/sapi/v1/dualInvestment/positions",
        )
        wallet_balance = self._try_binance_signed_json(
            "https://api.binance.com",
            "/sapi/v1/asset/wallet/balance",
        )
        prices = self._fetch_binance_price_map()
        aggregates: dict[str, AssetAggregate] = {}

        earn_assets: set[str] = set()
        if isinstance(earn_flex, dict):
            rows = earn_flex.get("rows", [])
            for item in rows if isinstance(rows, list) else []:
                if isinstance(item, dict):
                    asset, _wrapped = _normalize_binance_asset(item.get("asset"))
                    if asset and _decimal(item.get("totalAmount")) > 0:
                        earn_assets.add(asset)
        if isinstance(earn_locked, dict):
            rows = earn_locked.get("rows", [])
            for item in rows if isinstance(rows, list) else []:
                if isinstance(item, dict):
                    asset, _wrapped = _normalize_binance_asset(item.get("asset"))
                    total = _decimal(item.get("totalAmount") or item.get("amount"))
                    if asset and total > 0:
                        earn_assets.add(asset)

        spot_balances = spot.get("balances", []) if isinstance(spot, dict) else []
        for item in spot_balances if isinstance(spot_balances, list) else []:
            if not isinstance(item, dict):
                continue
            asset, wrapped = _normalize_binance_asset(item.get("asset"))
            if wrapped and asset in earn_assets:
                continue
            free = _decimal(item.get("free"))
            used = _decimal(item.get("locked"))
            total = free + used
            wallet = "earn-flex" if wrapped else "spot"
            self._add_asset(
                aggregates,
                raw_asset=asset,
                wallet=wallet,
                total=total,
                free=free,
                used=used,
                price=self._binance_usd_price(asset, prices),
            )

        for item in funding if isinstance(funding, list) else []:
            if not isinstance(item, dict):
                continue
            asset, _wrapped = _normalize_binance_asset(item.get("asset"))
            free = _decimal(item.get("free"))
            used = (
                _decimal(item.get("locked"))
                + _decimal(item.get("freeze"))
                + _decimal(item.get("with" + "drawing"))
            )
            self._add_asset(
                aggregates,
                raw_asset=asset,
                wallet="funding",
                total=free + used,
                free=free,
                used=used,
                price=self._binance_usd_price(asset, prices),
            )

        for item in futures if isinstance(futures, list) else []:
            if not isinstance(item, dict):
                continue
            asset = str(item.get("asset") or "").upper()
            wallet_balance_amount = _decimal(item.get("balance") or item.get("crossWalletBalance"))
            total = wallet_balance_amount + _decimal(item.get("crossUnPnl"))
            self._add_asset(
                aggregates,
                raw_asset=asset,
                wallet="futures-usds",
                total=total,
                price=self._binance_usd_price(asset, prices),
            )

        if isinstance(earn_flex, dict):
            rows = earn_flex.get("rows", [])
            for item in rows if isinstance(rows, list) else []:
                if isinstance(item, dict):
                    asset, _wrapped = _normalize_binance_asset(item.get("asset"))
                    self._add_asset(
                        aggregates,
                        raw_asset=asset,
                        wallet="earn-flex",
                        total=item.get("totalAmount"),
                        price=self._binance_usd_price(asset, prices),
                    )

        if isinstance(earn_locked, dict):
            rows = earn_locked.get("rows", [])
            for item in rows if isinstance(rows, list) else []:
                if isinstance(item, dict):
                    asset, _wrapped = _normalize_binance_asset(item.get("asset"))
                    self._add_asset(
                        aggregates,
                        raw_asset=asset,
                        wallet="earn-locked",
                        total=item.get("totalAmount") or item.get("amount"),
                        price=self._binance_usd_price(asset, prices),
                    )

        for item in coinm if isinstance(coinm, list) else []:
            if not isinstance(item, dict):
                continue
            asset = str(item.get("asset") or "").upper()
            total = _decimal(item.get("balance")) + _decimal(item.get("crossUnPnl"))
            self._add_asset(
                aggregates,
                raw_asset=asset,
                wallet="futures-coinm",
                total=total,
                price=self._binance_usd_price(asset, prices),
            )

        dual_items: object = []
        if isinstance(dual, dict):
            dual_items = dual.get("list", dual)
        elif isinstance(dual, list):
            dual_items = dual
        for item in dual_items if isinstance(dual_items, list) else []:
            if not isinstance(item, dict) or str(item.get("status") or "").upper() != "PROCESS":
                continue
            asset = str(item.get("investCoin") or "").upper()
            self._add_asset(
                aggregates,
                raw_asset=asset,
                wallet="dual-investment",
                total=item.get("subscriptionAmount"),
                price=self._binance_usd_price(asset, prices),
            )

        if isinstance(wallet_balance, list):
            btc_usd = prices.get("BTCUSDT") or prices.get("BTCFDUSD") or Decimal("0")
            reconciliations = {
                "Earn": "earn-flex",
                "Funding": "funding",
                "USDⓈ-M Futures": "futures-usds",
            }
            for wallet_name, wallet_type in reconciliations.items():
                wallet_entry = next(
                    (
                        item
                        for item in wallet_balance
                        if isinstance(item, dict)
                        and str(item.get("walletName") or "") == wallet_name
                    ),
                    None,
                )
                if not isinstance(wallet_entry, dict) or btc_usd <= 0:
                    continue
                wallet_usd = _decimal(wallet_entry.get("balance")) * btc_usd
                detailed_usd = sum(
                    aggregate.usd_value
                    for aggregate in aggregates.values()
                    if wallet_type in aggregate.sources
                )
                residual = wallet_usd - detailed_usd
                if residual > Decimal("1"):
                    self._add_asset(
                        aggregates,
                        raw_asset="USDT",
                        wallet=f"{wallet_type}-residual",
                        total=residual,
                        usd_value=residual,
                    )

        return self._finalize_balance_payload("binance", aggregates)

    def _okx_signature(self, timestamp: str, request_path: str) -> str:
        credentials = self._credentials["okx"]
        digest = hmac.new(
            credentials.api_secret.encode(),
            f"{timestamp}GET{request_path}".encode(),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(digest).decode()

    def _okx_json(self, request_path: str) -> object:
        credentials = self._credentials["okx"]
        timestamp = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        payload = self._request_json(
            "GET",
            f"https://www.okx.com{request_path}",
            headers={
                "OK-ACCESS-KEY": credentials.api_key,
                "OK-ACCESS-SIGN": self._okx_signature(timestamp, request_path),
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": credentials.passphrase,
                "user-agent": "aegis-exchange-bridge/1.0",
            },
        )
        if isinstance(payload, dict) and str(payload.get("code", "0")) != "0":
            raise RuntimeError(str(payload.get("msg") or "OKX_API_ERROR"))
        return payload

    def _try_okx_json(self, request_path: str) -> object | None:
        try:
            return self._okx_json(request_path)
        except Exception:
            return None

    def _okx_usd_price(self, asset: str) -> Decimal:
        symbol = asset.upper()
        if _is_cash_like(symbol):
            return Decimal("1")
        for inst_id in (f"{symbol}-USDT", f"{symbol}-USD"):
            try:
                payload = self._request_json(
                    "GET",
                    f"https://www.okx.com/api/v5/market/ticker?instId={inst_id}",
                    headers={"user-agent": "aegis-exchange-bridge/1.0"},
                )
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            data = payload.get("data")
            ticker = data[0] if isinstance(data, list) and data else None
            if isinstance(ticker, dict):
                price = _decimal(ticker.get("last") or ticker.get("lastPr") or ticker.get("askPx"))
                if price > 0:
                    return price
        return Decimal("0")

    def _fetch_okx_balance_payload(self) -> dict[str, object]:
        if not self.is_configured("okx"):
            raise RuntimeError("EXCHANGE_NOT_CONFIGURED")
        trading = self._okx_json("/api/v5/account/balance")
        funding = self._try_okx_json("/api/v5/asset/balances")
        savings = self._try_okx_json("/api/v5/finance/savings/balance")
        positions = self._try_okx_json("/api/v5/account/positions")
        price_cache: dict[str, Decimal] = {}
        aggregates: dict[str, AssetAggregate] = {}

        def price(asset: str) -> Decimal:
            if asset not in price_cache:
                price_cache[asset] = self._okx_usd_price(asset)
            return price_cache[asset]

        trading_data = trading.get("data", []) if isinstance(trading, dict) else []
        details: object = []
        if isinstance(trading_data, list) and trading_data and isinstance(trading_data[0], dict):
            details = trading_data[0].get("details", [])
        for item in details if isinstance(details, list) else []:
            if not isinstance(item, dict):
                continue
            asset = str(item.get("ccy") or "").upper()
            total = _decimal(item.get("eq") or item.get("cashBal"))
            used = _decimal(item.get("frozenBal")) + _decimal(item.get("ordFrozen"))
            self._add_asset(
                aggregates,
                raw_asset=asset,
                wallet="trading",
                total=total,
                free=max(total - used, Decimal("0")),
                used=used,
                usd_value=item.get("eqUsd"),
                price=price(asset),
            )

        funding_data = funding.get("data", []) if isinstance(funding, dict) else []
        for item in funding_data if isinstance(funding_data, list) else []:
            if not isinstance(item, dict):
                continue
            asset = str(item.get("ccy") or "").upper()
            free = _decimal(item.get("availBal"))
            used = _decimal(item.get("frozenBal"))
            total = free + used or _decimal(item.get("bal") or item.get("cashBal"))
            self._add_asset(
                aggregates,
                raw_asset=asset,
                wallet="funding",
                total=total,
                free=free if free else total,
                used=used,
                usd_value=item.get("eqUsd"),
                price=price(asset),
            )

        savings_data = savings.get("data", []) if isinstance(savings, dict) else []
        for item in savings_data if isinstance(savings_data, list) else []:
            if not isinstance(item, dict):
                continue
            asset = str(item.get("ccy") or item.get("asset") or "").upper()
            self._add_asset(
                aggregates,
                raw_asset=asset,
                wallet="savings",
                total=(
                    item.get("amt")
                    or item.get("bal")
                    or item.get("savingsAmt")
                    or item.get("amount")
                ),
                usd_value=item.get("eqUsd") or item.get("usdVal") or item.get("usdValue"),
                price=price(asset),
            )

        positions_data = positions.get("data", []) if isinstance(positions, dict) else []
        for item in positions_data if isinstance(positions_data, list) else []:
            if not isinstance(item, dict):
                continue
            inst_type = str(item.get("instType") or "").upper()
            if inst_type not in {"SWAP", "FUTURES"}:
                continue
            asset = str(item.get("ccy") or "").upper()
            self._add_asset(
                aggregates,
                raw_asset=asset,
                wallet="futures-margin",
                total=item.get("margin"),
                price=price(asset),
            )

        return self._finalize_balance_payload("okx", aggregates)

    def _raw_balance(self, exchange: ExchangeName) -> dict[str, Any]:
        client = self._clients.get(exchange)
        if client is None:
            raise RuntimeError("EXCHANGE_NOT_CONFIGURED")
        raw = client.fetch_balance()
        if not isinstance(raw, dict):
            raise RuntimeError("EXCHANGE_BALANCE_UNAVAILABLE")
        return raw

    def _normalize_ccxt_balance_payload(
        self, exchange: ExchangeName, raw: dict[str, Any]
    ) -> dict[str, object]:
        total_by_asset = raw.get("total", {})
        free_by_asset = raw.get("free", {})
        used_by_asset = raw.get("used", {})
        if not isinstance(total_by_asset, dict):
            total_by_asset = {}
        if not isinstance(free_by_asset, dict):
            free_by_asset = {}
        if not isinstance(used_by_asset, dict):
            used_by_asset = {}

        aggregates: dict[str, AssetAggregate] = {}
        assets = sorted(set(total_by_asset) | set(free_by_asset) | set(used_by_asset))
        for asset in assets:
            total = _decimal(total_by_asset.get(asset))
            if total == Decimal("0"):
                continue
            self._add_asset(
                aggregates,
                raw_asset=asset,
                wallet="default",
                total=total,
                free=free_by_asset.get(asset),
                used=used_by_asset.get(asset),
            )
        return self._finalize_balance_payload(exchange, aggregates)

    @staticmethod
    def _validate_exchange(exchange: ExchangeName) -> None:
        if exchange not in SUPPORTED_EXCHANGES:
            raise ValueError("UNSUPPORTED_EXCHANGE")
