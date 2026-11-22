"""SQLite job queue + results store."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

STATUSES = (
    "scouted", "ingested", "candidates", "rendered", "packaged", "approved",
    "posted", "submitted", "tracked", "failed", "dead_letter", "killed",
)

SCHEMA = f"""
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sources (
    id          INTEGER PRIMARY KEY,
    url_or_path TEXT NOT NULL,
    title       TEXT,
    duration_s  REAL,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS jobs (
    id            INTEGER PRIMARY KEY,
    kind          TEXT NOT NULL CHECK (kind IN ('source','clip')),
    source_id     INTEGER NOT NULL REFERENCES sources(id),
    parent_job_id INTEGER REFERENCES jobs(id),
    status        TEXT NOT NULL CHECK (status IN ({",".join(repr(s) for s in STATUSES)})),
    agent_lock    TEXT,
    locked_at     TEXT,
    attempts      INTEGER NOT NULL DEFAULT 0,
    qa_loops      INTEGER NOT NULL DEFAULT 0,
    reset_count   INTEGER NOT NULL DEFAULT 0,
    payload       TEXT NOT NULL DEFAULT '{{}}',
    decisions     TEXT NOT NULL DEFAULT '{{}}',
    last_error    TEXT,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, kind, agent_lock);

CREATE TABLE IF NOT EXISTS transitions (
    id          INTEGER PRIMARY KEY,
    job_id      INTEGER NOT NULL REFERENCES jobs(id),
    from_status TEXT,
    to_status   TEXT NOT NULL,
    agent       TEXT NOT NULL,
    reason      TEXT,
    ts          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_transitions_job ON transitions(job_id);

CREATE TABLE IF NOT EXISTS dead_letter (
    id      INTEGER PRIMARY KEY,
    job_id  INTEGER NOT NULL REFERENCES jobs(id),
    agent   TEXT NOT NULL,
    error   TEXT NOT NULL,
    logs    TEXT NOT NULL,
    payload TEXT NOT NULL,
    ts      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS breakers (
    key       TEXT PRIMARY KEY,
    state     TEXT NOT NULL CHECK (state IN ('closed','open','half_open')) DEFAULT 'closed',
    failures  INTEGER NOT NULL DEFAULT 0,
    opened_at REAL
);

CREATE TABLE IF NOT EXISTS playbook_versions (
    version    TEXT PRIMARY KEY,
    yaml       TEXT NOT NULL,
    status     TEXT NOT NULL CHECK (status IN ('active','shadow','retired')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS alerts (
    id       INTEGER PRIMARY KEY,
    kind     TEXT NOT NULL,
    severity TEXT NOT NULL,
    message  TEXT NOT NULL,
    ts       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def loads(text: str | None) -> dict:
    return json.loads(text) if text else {}


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)
