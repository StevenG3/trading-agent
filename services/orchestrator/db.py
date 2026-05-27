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
        "create index if not exists idx_scorecards_actor_symbol "
        "on scorecards(actor, symbol)"
    )
    conn.execute(
        "create index if not exists idx_scorecards_expires_at on scorecards(expires_at)"
    )
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
    conn.execute(
        "create index if not exists idx_daily_pnl_actor_date on daily_pnl(actor, date)"
    )
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
