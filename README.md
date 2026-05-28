# Trading Agent Phase 1

> This system can place real orders in later phases. Phase 1 ships with all trading paths disabled. Do not enable live mode until Phase 9.

Phase 1 creates four local HTTP service skeletons for paper-only order-intent validation and simulated execution. Live trading is disabled, no exchange keys are required or used, and no service connects to Binance or any other exchange.

## Run

```bash
cd deploy
docker compose up --build
```

Only `orchestrator` is configured to publish a host port: `8080`. On the current server, another local Nginx already owns `127.0.0.1:8080`, so use container-internal self-tests or change the host port deliberately if you want local host access.

## Paper intent example

```bash
curl -s http://127.0.0.1:8080/intents \
  -H 'content-type: application/json' \
  -d '{
    "intent_id":"11111111-1111-4111-8111-111111111111",
    "request_id":"22222222-2222-4222-8222-222222222222",
    "idempotency_key":"demo-paper-1",
    "actor":"user_1",
    "created_at":"2026-05-25T00:00:00Z",
    "mode":"paper",
    "venue":"binance_spot",
    "symbol":"BTCUSDT",
    "side":"buy",
    "order_type":"market",
    "quantity":{"kind":"quote","value":"100"},
    "limit_price":null,
    "time_in_force":"GTC",
    "reduce_only":false,
    "leverage":null,
    "stop_loss":null,
    "take_profit":null,
    "source":{"origin":"manual_api","scorecard_id":null,"hermes_message_id":null},
    "client_confirmation_required":false
  }'
```

## Safety

Live trading is disabled in Phase 1. `mode="live"` is rejected before execution. The execution service only returns simulated fills from fixture market data. Exchange API keys are placeholders only and are not read by the services.


## Phase 2

Phase 2 keeps live trading disabled. Paper intents below the confirmation threshold execute immediately with simulated fills; medium paper intents return a confirmation token and must be confirmed before execution.

Create a paper order that requires confirmation:

```bash
curl -s http://127.0.0.1:8080/intents \
  -H 'content-type: application/json' \
  -d '{
    "intent_id":"55555555-5555-4555-8555-555555555555",
    "request_id":"66666666-6666-4666-8666-666666666666",
    "idempotency_key":"demo-paper-confirm-1",
    "actor":"user_1",
    "created_at":"2026-05-25T00:00:00Z",
    "mode":"paper",
    "venue":"binance_spot",
    "symbol":"BTCUSDT",
    "side":"buy",
    "order_type":"market",
    "quantity":{"kind":"quote","value":"1000"},
    "limit_price":null,
    "time_in_force":"GTC",
    "reduce_only":false,
    "leverage":null,
    "stop_loss":null,
    "take_profit":null,
    "source":{"origin":"manual_api","scorecard_id":null,"hermes_message_id":null},
    "client_confirmation_required":false
  }'
```

A `202` response includes `confirmation_token`. Confirm it with:

```bash
curl -s http://127.0.0.1:8080/intents/55555555-5555-4555-8555-555555555555/confirm \
  -H 'content-type: application/json' \
  -d '{
    "intent_id":"55555555-5555-4555-8555-555555555555",
    "confirmation_token":"<token-from-202-response>"
  }'
```

List intents:

```bash
curl -s 'http://127.0.0.1:8080/intents?limit=20&offset=0&mode=paper'
```

Live trading remains disabled in Phase 2. `mode="live"` returns `LIVE_DISABLED_PHASE1` and never reaches execution.


## Phase 3

Phase 3 adds idempotency, quote-to-base quantity calculation, per-symbol daily exposure tracking, and `/exposure`.

A quote-kind paper order now resolves quantity from market price before simulated execution. For example, 100 USDT at a 50000 USDT/BTC price fills `0.00200000` BTC.

Idempotency is keyed by `idempotency_key`: sending the same key again returns the original cached result without calling risk again. A different `intent_id` with the same `idempotency_key` still returns the cached original result.

Current exposure can be inspected with:

```bash
curl -s 'http://127.0.0.1:8080/exposure'
curl -s 'http://127.0.0.1:8080/exposure?date=2026-05-25'
```

`PER_SYMBOL_DAILY_LIMIT_USDT` controls the daily per-symbol cap. Live trading remains disabled in Phase 3.

## Phase 4

Phase 4 renames the durable SQLite file to `trading.sqlite`. Existing `phase1.sqlite` files are migrated automatically on first service start when the new file is absent.

Pending confirmation intents can be canceled before execution:

```bash
curl -i -X DELETE http://127.0.0.1:8080/intents/<intent-id>
```

Canceled idempotency keys are terminal and return `INTENT_CANCELED` on resubmission.

Paper executions now update `/paper/positions` for each actor. Open positions are marked from market data when available; if marking fails, mark fields are returned as `null` without failing the request.

```bash
curl -s 'http://127.0.0.1:8080/paper/positions?actor=user_1'
```

`PER_SYMBOL_DAILY_LIMIT_USDT` is also reported in risk decisions under `hard_caps_applied.per_symbol_exposure`. Live trading remains disabled in Phase 4.

## Phase 5: Live Trading

Live trading sends real Binance Spot market orders and can spend real funds. Paper mode remains the safe default and does not require exchange credentials.

To enable live trading, set `LIVE_TRADING_ENABLED=true`, `EXCHANGE_API_KEY`, and `EXCHANGE_API_SECRET` in `.env`, then restart the stack. Keep the gate false until credentials and risk limits are verified.

Phase 5 supports only live `order_type=market`; live limit orders are rejected by the risk-engine with `LIMIT_ORDER_LIVE_UNSUPPORTED`. The confirmation flow still applies: orders at or above `CONFIRMATION_THRESHOLD_USDT` require explicit confirmation before execution.

## Phase 9: Scorecards, Drawdown, Live Unlock

Phase 9 adds a scorecard signal layer. A human analyst, Hermes chat, or a future TradingAgents adapter can create a structured thesis first, then convert that scorecard into an order intent:

```bash
curl -s http://127.0.0.1:8080/scorecards \
  -H 'content-type: application/json' \
  -d '{
    "actor":"user_1",
    "symbol":"BTCUSDT",
    "action":"buy",
    "source":"manual",
    "conviction":"0.5",
    "thesis":"Breakout above 100k support retest",
    "entry_low":"95000.00",
    "entry_high":"100000.00",
    "stop_loss":"90000.00",
    "take_profit":"110000.00",
    "time_horizon":"swing",
    "ttl_minutes":60
  }'
```

```bash
curl -s http://127.0.0.1:8080/intents/from_scorecard \
  -H 'content-type: application/json' \
  -d '{
    "scorecard_id":"<scorecard_id>",
    "actor":"user_1",
    "idempotency_key":"scorecard-demo-1",
    "mode":"paper",
    "usdt_budget":"200",
    "position_fraction":"1.0",
    "order_type":"market"
  }'
```

Position size is `conviction x usdt_budget x position_fraction`. For example, conviction `0.5` and budget `200` creates a `100` USDT quote-sized intent.

Live trading now has two gates. Gate 1 is still the operator-controlled environment setup: `LIVE_TRADING_ENABLED=true` plus `EXCHANGE_API_KEY` and `EXCHANGE_API_SECRET`. Gate 2 is a short-lived single-use live unlock token:

```bash
curl -s http://127.0.0.1:8080/admin/live-unlock \
  -H 'content-type: application/json' \
  -H 'x-ops-token: <OPS_TOKEN>' \
  -d '{"actor":"user_1"}'
```

Every live intent must carry the returned token in `x-live-unlock`. Paper trading does not require this header. By default `OPS_TOKEN` is unset, so no live token can be minted.

The daily drawdown circuit breaker uses `DAILY_DRAWDOWN_HARD_STOP_USDT=1000` by default. When an actor's total PnL for the UTC date is `<= -1000`, risk-engine rejects further orders for that actor with `DAILY_DRAWDOWN_BREACHED`. The bucket naturally resets at UTC midnight.

Inspect today's PnL:

```bash
curl -s 'http://127.0.0.1:8080/pnl/today?actor=user_1'
```

Live trading still supports Binance Spot only. IBKR, equities, futures, margin, and leverage are deliberately deferred.

## Phase 10: TradingAgents analysis adapter

The `analysis-adapter` service is the HTTP seam between the external TradingAgents deployment in `/home/gggqqy/apps/tradingagents-official/` and this platform's scorecard layer. It never imports TradingAgents modules and never talks to execution-service; it calls the TradingAgents bridge over HTTP, translates the result into a scorecard, then posts that scorecard to orchestrator.

Start a TA-driven analysis with:

```bash
curl -X POST http://localhost:8085/analyze \
  -H "content-type: application/json" \
  -d '{"actor":"user_1","symbol":"BTCUSDT","asset_type":"crypto"}'
```

The request returns immediately with `{"job_id":"...","status":"queued"}`. Poll `GET /jobs/{job_id}` until `status="succeeded"`; the resulting `scorecard_id` can then be consumed through orchestrator's existing `/intents/from_scorecard` path. Live order creation still requires `LIVE_TRADING_ENABLED` plus a valid single-use `x-live-unlock` token.

TradingAgents analysts are originally equity-focused. Crypto support depends on TradingAgents' own dataflows; this adapter does not compensate for missing crypto-native data sources, so treat crypto reports with appropriate skepticism until upstream data sources improve.

The conviction heuristic is intentionally simple: hold maps to `0.30`; buy/sell starts at `0.50` and adds `0.10` for each populated analyst report, capped at `0.90`. Future phases can refine this from TA debate margins and historical hit rate.

## Phase 12: Scorecard Outcomes

Phase 12 adds a forward-only `scorecard_outcomes` table that records whether scorecard-sourced BUY signals ultimately made or lost money. A filled scorecard BUY opens an outcome row. When the aggregate `(actor, symbol)` position closes, every open outcome for that pair is closed with realized PnL and return percentage.

Attribution is approximate by design: if multiple scorecards overlap on the same `(actor, symbol)`, the close PnL is split proportionally by each row's `opened_cost_basis`; those rows carry `notes="split-attribution"`. Manual orders and natural-language orders without a scorecard reference do not create outcome rows.

Examples:

```bash
curl 'http://localhost:8080/scorecard-outcomes?actor=user_1&status=closed'
curl 'http://localhost:8080/scorecard-outcomes/summary?actor=user_1'
curl 'http://localhost:8080/scorecard-outcomes/11111111-1111-4111-8111-111111111111'
```


### Outcome reflection

Closed scorecard outcomes are pushed back through the analysis adapter to the TradingAgents bridge so future analyses can learn from paper-trading results. The reflection payload includes raw return plus benchmark-adjusted alpha when the original scorecard captured a benchmark open price. If benchmark pricing is unavailable, alpha falls back to raw return and the payload carries an `alpha_note`. If the bridge is unavailable, `reflected_at` stays empty and operators can retry with `POST /reflect/pending?limit=50`. The outcomes summary includes `pending_reflection_count` per source.

### Watchlists and scheduled analysis

The orchestrator can keep an actor-scoped watchlist and periodically ask the analysis adapter to create fresh scorecards. Use `POST /watchlist` with `actor`, `symbol`, `asset_type`, and `cadence_minutes` (15-1440), `GET /watchlist?actor=...` to inspect active entries, and `DELETE /watchlist/{symbol}?actor=...` to disable one. Scheduler knobs live in `deploy/.env.example`: `SCHEDULER_ENABLED`, `SCHEDULER_TICK_SEC`, and `SCHEDULER_BATCH_LIMIT`.


### Phase 15 paper autonomy

The orchestrator can recompute conviction calibration from reflected closed scorecard outcomes with `POST /calibration/recompute`; analysis-adapter records both `heuristic_conviction` and `calibrated_conviction` in scorecard metadata. Paper autonomy is opt-in per actor via `/autonomy/settings`, bounded by `daily_budget_usdt`, `per_trade_usdt`, `min_conviction`, and `allowed_sources`. Auto-trades hard-code `mode=paper` and call the existing `/intents/from_scorecard` path through `ORCHESTRATOR_SELF_URL`.

## Phase 16: Gated Live Autonomy

Phase 16 adds live-autonomy controls, but the default remains fully off. A live autonomous order can fire only when all gates pass: `LIVE_TRADING_ENABLED=true`, exchange credentials are present, `LIVE_AUTONOMY_GLOBAL_ENABLED=true`, the actor has opted in through `/live-autonomy/settings`, calibration sample thresholds pass, daily spend and trade-count limits have room, drawdown is not breached, and the global live-autonomy kill switch is not engaged.

Phase 18 also adds a protective watchdog for open scorecard outcomes. When enabled with `STOP_LOSS_WATCHDOG_ENABLED=true`, the scheduler checks open outcomes in batches, compares the current mark price with the scorecard `stop_loss` and `take_profit`, and submits a reduce-only sell through the same `/intents` path. Live protective sells mint an intent-bound unlock token internally; row-level failures are counted and never crash the scheduler.

Operator controls:

```bash
curl -s -X POST http://127.0.0.1:8080/live-autonomy/settings   -H 'content-type: application/json'   -d '{"actor":"tg_5175667339","enabled":true,"daily_live_budget_usdt":"100","per_live_trade_max_usdt":"25","min_calibrated_conviction":"0.75"}'

curl -s 'http://127.0.0.1:8080/live-autonomy/settings?actor=tg_5175667339'
curl -s 'http://127.0.0.1:8080/live-autonomy/today?actor=tg_5175667339'
curl -s -X POST http://127.0.0.1:8080/admin/live-autonomy/disable
curl -s -X POST http://127.0.0.1:8080/admin/live-autonomy/enable
```

The autonomous live path mints a two-minute, single-use, actor-scoped unlock token internally. Because `/intents/from_scorecard` still generates `intent_id` internally, Phase 16 auto-minted tokens are not pre-bound to an intent id; manual live unlock tokens remain single-use and can be bound by future schema work.
