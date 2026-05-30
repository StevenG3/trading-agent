# Exchange Bridge

Small internal FastAPI wrapper for read-only balances across Binance, OKX, and
Bybit. Binance and OKX use exchange-native account endpoints so spot, funding,
earn/savings, and futures margin balances are aggregated by asset before the
dashboard renders them. Bybit still uses the ccxt default balance endpoint.

Default endpoint:

```text
GET /balances
GET /balances?exchange=binance
GET /healthz
GET /readyz
```

Environment variables:

```text
EXCHANGE_API_KEY=<<unset>>
EXCHANGE_API_SECRET=<<unset>>
OKX_API_KEY=<<unset>>
OKX_API_SECRET=<<unset>>
OKX_API_PASSPHRASE=<<unset>>
BYBIT_API_KEY=<<unset>>
BYBIT_API_SECRET=<<unset>>
EXCHANGE_TIMEOUT_SEC=10
EXCHANGE_MIN_USD_DETAIL=10
```

`/healthz` reports process health. `/readyz` returns `503` when no exchange is
configured or none of the configured exchanges can be reached.

`EXCHANGE_MIN_USD_DETAIL` hides per-asset rows whose estimated USD value is
below the threshold. The default is `10`.
