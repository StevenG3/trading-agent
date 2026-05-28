from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def db_path() -> Path:
    data_dir = Path(os.getenv("DATA_DIR", "/tmp/trading-agent-data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    new_path = data_dir / "trading.sqlite"
    old_path = data_dir / "phase1.sqlite"
    if not new_path.exists() and old_path.exists():
        old_path.replace(new_path)
    return new_path


def connect() -> sqlite3.Connection:
    path = db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        create table if not exists intents (
            intent_id text primary key,
            payload_json text not null,
            decision_json text,
            execution_json text,
            created_at text not null
        )
        """
    )
    try:
        conn.execute("alter table intents add column status text not null default 'pending'")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("alter table intents add column idempotency_key text not null default ''")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    conn.execute(
        """
        update intents
        set idempotency_key = json_extract(payload_json, '$.idempotency_key')
        where idempotency_key = ''
        """
    )
    conn.execute(
        """
        create table if not exists daily_fills (
            fill_id text primary key,
            date text not null,
            symbol text not null,
            side text not null,
            notional_usdt text not null,
            created_at text not null
        )
        """
    )
    conn.execute("create index if not exists idx_intents_created_at on intents(created_at desc)")
    conn.execute(
        "create unique index if not exists idx_intents_idempotency_key on intents(idempotency_key)"
    )
    conn.execute(
        """
        create table if not exists paper_positions (
            actor text not null,
            symbol text not null,
            qty text not null default '0',
            avg_cost text not null default '0',
            total_cost text not null default '0',
            realized_pnl text not null default '0',
            last_updated text not null,
            primary key (actor, symbol)
        )
        """
    )
    for column, default in (
        ("paper_qty", "0"),
        ("paper_avg_cost", "0"),
        ("live_qty", "0"),
        ("live_avg_cost", "0"),
    ):
        try:
            conn.execute(
                f"alter table paper_positions add column {column} text not null default '{default}'"
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass
    conn.execute(
        """
        update paper_positions
        set paper_qty = qty
        where paper_qty = '0' and cast(qty as real) != 0
        """
    )
    conn.execute(
        """
        update paper_positions
        set paper_avg_cost = avg_cost
        where paper_avg_cost = '0' and cast(qty as real) != 0
        """
    )
    conn.execute(
        "create index if not exists idx_daily_fills_date_symbol on daily_fills(date, symbol)"
    )
    conn.execute(
        """
        create table if not exists scorecards (
            scorecard_id text primary key,
            actor text not null,
            symbol text not null,
            action text not null,
            source text not null,
            payload_json text not null,
            created_at text not null,
            expires_at text not null,
            consumed_by_intent_id text
        )
        """
    )
    conn.execute(
        "create index if not exists idx_scorecards_actor_symbol on scorecards(actor, symbol)"
    )
    conn.execute("create index if not exists idx_scorecards_expires_at on scorecards(expires_at)")
    conn.execute(
        """
        create table if not exists daily_pnl (
            actor text not null,
            date text not null,
            realized_delta text not null,
            symbol text not null,
            created_at text not null
        )
        """
    )
    conn.execute("create index if not exists idx_daily_pnl_actor_date on daily_pnl(actor, date)")
    conn.execute(
        """
        create table if not exists live_unlock_tokens (
            token text primary key,
            actor text not null,
            created_at text not null,
            expires_at text not null,
            consumed_at text
        )
        """
    )
    conn.execute(
        "create index if not exists idx_live_unlock_expires on live_unlock_tokens(expires_at)"
    )
    conn.execute(
        """
        create table if not exists watchlist_entries (
            actor text not null,
            symbol text not null,
            asset_type text not null,
            cadence_minutes integer not null,
            last_run_at text,
            next_run_at text not null,
            enabled integer not null default 1,
            created_at text not null,
            primary key (actor, symbol)
        )
        """
    )
    conn.execute(
        "create index if not exists idx_watchlist_due on watchlist_entries(enabled, next_run_at)"
    )

    conn.execute(
        """
        create table if not exists conviction_calibration (
            source text not null,
            asset_type text not null,
            heuristic_bucket text not null,
            sample_count integer not null,
            hit_count integer not null,
            avg_alpha_return text not null,
            empirical_hit_rate text not null,
            calibrated_conviction text not null,
            updated_at text not null,
            primary key (source, asset_type, heuristic_bucket)
        )
        """
    )
    conn.execute(
        """
        create table if not exists autonomy_settings (
            actor text primary key,
            enabled integer not null default 0,
            daily_budget_usdt text not null default '0',
            min_conviction text not null default '0.65',
            per_trade_usdt text not null default '50',
            allowed_sources text not null default 'tradingagents',
            updated_at text not null
        )
        """
    )
    conn.execute(
        """
        create table if not exists autonomy_spend (
            actor text not null,
            date text not null,
            spent_usdt text not null default '0',
            trade_count integer not null default 0,
            last_updated text not null,
            primary key (actor, date)
        )
        """
    )

    try:
        conn.execute("alter table live_unlock_tokens add column bound_intent_id text")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    conn.execute(
        """
        create table if not exists live_autonomy_settings (
            actor text primary key,
            enabled integer not null default 0,
            daily_live_budget_usdt text not null default '0',
            per_live_trade_max_usdt text not null default '50',
            max_live_exposure_usdt text not null default '0',
            daily_live_trade_count_max integer not null default 3,
            min_calibrated_conviction text not null default '0.70',
            min_closed_outcomes integer not null default 20,
            allowed_sources text not null default 'tradingagents',
            created_at text not null,
            updated_at text not null
        )
        """
    )
    try:
        conn.execute(
            "alter table live_autonomy_settings "
            "add column max_live_exposure_usdt text not null default '0'"
        )
        conn.commit()
    except sqlite3.OperationalError:
        pass
    conn.execute(
        """
        create table if not exists live_autonomy_spend (
            actor text not null,
            date text not null,
            spent_usdt text not null default '0',
            trade_count integer not null default 0,
            last_updated text not null,
            primary key (actor, date)
        )
        """
    )
    conn.execute(
        """
        create table if not exists live_autonomy_kill (
            id integer primary key check (id = 1),
            killed integer not null default 0,
            killed_at text,
            killed_by text
        )
        """
    )
    conn.execute("insert or ignore into live_autonomy_kill (id, killed) values (1, 0)")
    conn.execute(
        """
        create table if not exists notification_subscriptions (
            actor text primary key,
            webhook_url text not null,
            secret text not null,
            events_json text not null,
            enabled integer not null default 1,
            created_at text not null,
            updated_at text not null
        )
        """
    )
    conn.execute(
        """
        create table if not exists notification_deliveries (
            id integer primary key autoincrement,
            actor text not null,
            event_type text not null,
            webhook_url text not null,
            status_code integer,
            ok integer not null,
            error_class text,
            created_at text not null
        )
        """
    )
    conn.execute(
        "create index if not exists idx_notification_deliveries_actor_created "
        "on notification_deliveries(actor, created_at desc)"
    )
    conn.execute(
        """
        create table if not exists scorecard_outcomes (
            outcome_id text primary key,
            scorecard_id text not null,
            actor text not null,
            symbol text not null,
            source text not null,
            action text not null,
            opened_intent_id text not null,
            opened_at text not null,
            opened_qty text not null,
            opened_avg_cost text not null,
            opened_cost_basis text not null,
            status text not null,
            closed_at text,
            closed_realized_pnl text,
            closed_return_pct text,
            notes text
        )
        """
    )
    conn.execute(
        "create index if not exists idx_scorecard_outcomes_actor_symbol "
        "on scorecard_outcomes(actor, symbol, status)"
    )
    conn.execute(
        "create index if not exists idx_scorecard_outcomes_scorecard "
        "on scorecard_outcomes(scorecard_id)"
    )
    try:
        conn.execute("alter table scorecard_outcomes add column reflected_at text")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    conn.execute(
        "create index if not exists idx_scorecard_outcomes_reflection "
        "on scorecard_outcomes(status, reflected_at)"
    )
    conn.execute(
        "create index if not exists idx_scorecard_outcomes_source "
        "on scorecard_outcomes(source, status)"
    )
    conn.commit()
