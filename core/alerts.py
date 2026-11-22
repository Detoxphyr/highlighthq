"""Escalation. Alerts are written to /alerts only for the kinds a human must
see: account strikes, payout anomalies, tripped breakers, or a daily failure
rate above the threshold. Everything else goes to the daily digest only."""
from __future__ import annotations

import json
import time
from pathlib import Path

from core.logging import log

ESCALATE_KINDS = {"account_strike", "payout_anomaly", "breaker_tripped", "failure_rate"}
FAILURE_RATE_THRESHOLD = 0.20


def raise_alert(conn, alerts_dir: Path, kind: str, message: str, severity: str = "high", **details) -> Path | None:
    conn.execute("INSERT INTO alerts(kind, severity, message) VALUES (?,?,?)", (kind, severity, message))
    if kind not in ESCALATE_KINDS:
        log("alert_suppressed", kind=kind, message=message)
        return None
    alerts_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = alerts_dir / f"alert-{stamp}-{kind}.json"
    path.write_text(json.dumps({"ts": stamp, "kind": kind, "severity": severity, "message": message, **details}, indent=2))
    log("alert_escalated", level="error", kind=kind, path=str(path), message=message)
    return path


def daily_stats(conn, day: str | None = None) -> dict:
    day = day or time.strftime("%Y-%m-%d", time.gmtime())
    rows = conn.execute(
        "SELECT to_status, COUNT(*) n FROM transitions WHERE substr(ts,1,10)=? GROUP BY to_status", (day,)
    ).fetchall()
    counts = {r["to_status"]: r["n"] for r in rows}
    terminal_bad = conn.execute(
        "SELECT COUNT(DISTINCT job_id) n FROM transitions WHERE substr(ts,1,10)=? "
        "AND to_status IN ('failed','dead_letter')", (day,)).fetchone()["n"]
    touched = conn.execute(
        "SELECT COUNT(DISTINCT job_id) n FROM transitions WHERE substr(ts,1,10)=?", (day,)).fetchone()["n"]
    rate = terminal_bad / touched if touched else 0.0
    return {"day": day, "transitions": counts, "jobs_touched": touched, "jobs_failed": terminal_bad, "failure_rate": rate}


def write_digest(conn, alerts_dir: Path, day: str | None = None) -> Path:
    stats = daily_stats(conn, day)
    alerts = conn.execute(
        "SELECT kind, severity, message, ts FROM alerts WHERE substr(ts,1,10)=? ORDER BY ts", (stats["day"],)
    ).fetchall()
    jobs = conn.execute("SELECT status, kind, COUNT(*) n FROM jobs GROUP BY status, kind ORDER BY kind, status").fetchall()
    lines = [f"# HighlightHQ daily digest {stats['day']}", "",
             f"- Jobs touched: {stats['jobs_touched']}",
             f"- Jobs that failed at least once: {stats['jobs_failed']} ({stats['failure_rate']:.0%})", "",
             "## Job table by status", "", "| kind | status | count |", "|---|---|---|"]
    lines += [f"| {r['kind']} | {r['status']} | {r['n']} |" for r in jobs]
    lines += ["", "## Alerts", ""]
    lines += [f"- {a['ts']} [{a['severity']}] {a['kind']}: {a['message']}" for a in alerts] or ["- none"]
    alerts_dir.mkdir(parents=True, exist_ok=True)
    path = alerts_dir / f"digest-{stats['day']}.md"
    path.write_text("\n".join(lines) + "\n")
    if stats["jobs_touched"] >= 5 and stats["failure_rate"] > FAILURE_RATE_THRESHOLD:
        raise_alert(conn, alerts_dir, "failure_rate",
                    f"daily failure rate {stats['failure_rate']:.0%} exceeds {FAILURE_RATE_THRESHOLD:.0%}", **stats)
    return path
