# Exchange Bridge

Small internal FastAPI wrapper around `ccxt` for read-only balances across
Binance, OKX, and Bybit. Credentials are read only from environment variables.

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
```

`/healthz` reports process health. `/readyz` returns `503` when no exchange is
configured or none of the configured exchanges can be reached.
