from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

ExchangeName = Literal["binance", "okx", "bybit"]
SUPPORTED_EXCHANGES: tuple[ExchangeName, ...] = ("binance", "okx", "bybit")


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


class ExchangeClient:
    def __init__(
        self,
        credentials: dict[ExchangeName, ExchangeCredentials],
        *,
        timeout_sec: float = 10.0,
        ccxt_module: Any | None = None,
    ) -> None:
        self._credentials = credentials
        self._timeout_ms = int(timeout_sec * 1000)
        self._ccxt: Any | None = ccxt_module if ccxt_module is not None else self._load_ccxt()
        self._clients: dict[ExchangeName, Any] = {}
        self._build_clients()

    @classmethod
    def from_env(cls) -> ExchangeClient:
        return cls(
            credentials_from_env(),
            timeout_sec=float(os.getenv("EXCHANGE_TIMEOUT_SEC", "10")),
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
        if exchange not in self._clients:
            return False
        try:
            self._raw_balance(exchange)
        except Exception:
            return False
        return True

    def fetch_balances(self, exchange: ExchangeName) -> list[dict[str, str]]:
        self._validate_exchange(exchange)
        raw = self._raw_balance(exchange)
        return self._normalize_balance(exchange, raw)

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

    def _raw_balance(self, exchange: ExchangeName) -> dict[str, Any]:
        client = self._clients.get(exchange)
        if client is None:
            raise RuntimeError("EXCHANGE_NOT_CONFIGURED")
        raw = client.fetch_balance()
        if not isinstance(raw, dict):
            raise RuntimeError("EXCHANGE_BALANCE_UNAVAILABLE")
        return raw

    def _normalize_balance(
        self, exchange: ExchangeName, raw: dict[str, Any]
    ) -> list[dict[str, str]]:
        total_by_asset = raw.get("total", {})
        free_by_asset = raw.get("free", {})
        used_by_asset = raw.get("used", {})
        if not isinstance(total_by_asset, dict):
            total_by_asset = {}
        if not isinstance(free_by_asset, dict):
            free_by_asset = {}
        if not isinstance(used_by_asset, dict):
            used_by_asset = {}

        assets = sorted(set(total_by_asset) | set(free_by_asset) | set(used_by_asset))
        balances: list[dict[str, str]] = []
        for asset in assets:
            total = _decimal(total_by_asset.get(asset))
            if total == Decimal("0"):
                continue
            balances.append(
                {
                    "exchange": exchange,
                    "asset": str(asset),
                    "free": _decimal_str(free_by_asset.get(asset)),
                    "used": _decimal_str(used_by_asset.get(asset)),
                    "total": _decimal_str(total),
                }
            )
        return balances

    @staticmethod
    def _validate_exchange(exchange: ExchangeName) -> None:
        if exchange not in SUPPORTED_EXCHANGES:
            raise ValueError("UNSUPPORTED_EXCHANGE")
