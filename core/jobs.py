"""Job cards and the status state machine. Agents never call each other; they
only claim jobs in their input status and move them to the next status here."""
from __future__ import annotations

import json
import socket
import os
from dataclasses import dataclass, field

from core.db import dumps, loads
from core.logging import log, tail_for_job

PIPELINE = ["scouted", "ingested", "candidates", "rendered", "packaged", "approved", "posted", "submitted", "tracked"]

ALLOWED = {
    "scouted": {"ingested"},
    "ingested": {"candidates"},
    "candidates": {"rendered"},
    "rendered": {"packaged"},
    # QA Gate: approve, or route back to the responsible agent's input status.
    "packaged": {"approved", "candidates", "rendered"},
    "approved": {"posted"},
    "posted": {"submitted"},
    "submitted": {"tracked"},
    "failed": set(PIPELINE) | {"dead_letter"},  # requeue to the input status or dead-letter
}
ACTIVE = set(PIPELINE) - {"tracked"}
MAX_ATTEMPTS = 3


class InvalidTransition(Exception):
    pass


@dataclass
class Job:
    id: int
    kind: str
    source_id: int
    parent_job_id: int | None
    status: str
    attempts: int
    qa_loops: int
    reset_count: int
    payload: dict = field(default_factory=dict)
    decisions: dict = field(default_factory=dict)
    last_error: str | None = None

    @classmethod
    def from_row(cls, row) -> "Job":
        return cls(id=row["id"], kind=row["kind"], source_id=row["source_id"], parent_job_id=row["parent_job_id"],
                   status=row["status"], attempts=row["attempts"], qa_loops=row["qa_loops"],
                   reset_count=row["reset_count"], payload=loads(row["payload"]), decisions=loads(row["decisions"]),
                   last_error=row["last_error"])


def _check(from_status: str | None, to_status: str) -> None:
    if to_status in ("killed", "failed"):
        return  # any job may be killed or fail
    if from_status is None:
        return
    if to_status not in ALLOWED.get(from_status, set()):
        raise InvalidTransition(f"{from_status} -> {to_status}")


def record_transition(conn, job_id: int, from_status: str | None, to_status: str, agent: str, reason: str = "") -> None:
    conn.execute("INSERT INTO transitions(job_id, from_status, to_status, agent, reason) VALUES (?,?,?,?,?)",
                 (job_id, from_status, to_status, agent, reason))
    log("job_transition", job_id=job_id, from_status=from_status, to_status=to_status, agent=agent, reason=reason)


def add_source(conn, url_or_path: str, title: str | None = None) -> int:
    """Create a source and its job card in 'scouted'."""
    cur = conn.execute("INSERT INTO sources(url_or_path, title) VALUES (?,?)", (url_or_path, title))
    source_id = cur.lastrowid
    return create_job(conn, kind="source", source_id=source_id, status="scouted",
                      payload={"input": url_or_path}, agent="scout")


def create_job(conn, *, kind: str, source_id: int, status: str, payload: dict, decisions: dict | None = None,
               parent_job_id: int | None = None, agent: str) -> int:
    cur = conn.execute(
        "INSERT INTO jobs(kind, source_id, parent_job_id, status, payload, decisions) VALUES (?,?,?,?,?,?)",
        (kind, source_id, parent_job_id, status, dumps(payload), dumps(decisions or {})))
    record_transition(conn, cur.lastrowid, None, status, agent, "created")
    return cur.lastrowid


def claim(conn, status: str, kind: str, agent: str) -> Job | None:
    """Atomically lock the oldest unlocked job in `status`."""
    lock = f"{agent}@{socket.gethostname()}:{os.getpid()}"
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT id FROM jobs WHERE status=? AND kind=? AND agent_lock IS NULL ORDER BY id LIMIT 1",
            (status, kind)).fetchone()
        if not row:
            conn.execute("COMMIT")
            return None
        conn.execute("UPDATE jobs SET agent_lock=?, locked_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=?",
                     (lock, row["id"]))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return get(conn, row["id"])


def get(conn, job_id: int) -> Job:
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if row is None:
        raise KeyError(job_id)
    return Job.from_row(row)


def transition(conn, job: Job, to_status: str, agent: str, reason: str = "", *, payload: dict | None = None,
               decisions: dict | None = None, qa_loops: int | None = None, error: str | None = None,
               reset_attempts: bool = False) -> Job:
    """Move a job, merge payload/decisions, release the lock, log the move."""
    _check(job.status, to_status)
    new_payload = {**job.payload, **(payload or {})}
    new_decisions = {**job.decisions, **(decisions or {})}
    attempts = 0 if reset_attempts else job.attempts
    conn.execute(
        "UPDATE jobs SET status=?, payload=?, decisions=?, qa_loops=?, last_error=?, attempts=?, agent_lock=NULL, "
        "locked_at=NULL, updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=?",
        (to_status, dumps(new_payload), dumps(new_decisions), job.qa_loops if qa_loops is None else qa_loops,
         error if error is not None else (job.last_error if to_status == "failed" else None), attempts, job.id))
    record_transition(conn, job.id, job.status, to_status, agent, reason)
    return get(conn, job.id)


def fail(conn, job: Job, agent: str, error: str, input_status: str) -> Job:
    """Record a failure. Requeue to the agent's input status until MAX_ATTEMPTS,
    then move to dead_letter with the full logs."""
    conn.execute("UPDATE jobs SET attempts=attempts+1 WHERE id=?", (job.id,))
    job = get(conn, job.id)
    job = transition(conn, job, "failed", agent, f"attempt {job.attempts}: {error[:300]}",
                     payload={"failed_from": input_status}, error=error)
    if job.attempts >= MAX_ATTEMPTS:
        conn.execute("INSERT INTO dead_letter(job_id, agent, error, logs, payload) VALUES (?,?,?,?,?)",
                     (job.id, agent, error, json.dumps(tail_for_job(job.id), default=str), dumps(job.payload)))
        return transition(conn, job, "dead_letter", agent, f"exhausted {MAX_ATTEMPTS} attempts")
    return transition(conn, job, input_status, agent, "requeue after failure", error=error)


def kill(conn, job: Job, agent: str, reason: str) -> Job:
    return transition(conn, job, "killed", agent, reason)


def release(conn, job: Job) -> None:
    conn.execute("UPDATE jobs SET agent_lock=NULL, locked_at=NULL WHERE id=?", (job.id,))


def job_table(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT id, kind, parent_job_id, status, attempts, qa_loops, payload, decisions, last_error FROM jobs ORDER BY id"
    ).fetchall()
    out = []
    for r in rows:
        p, d = loads(r["payload"]), loads(r["decisions"])
        out.append({
            "id": r["id"], "kind": r["kind"], "parent": r["parent_job_id"], "status": r["status"],
            "attempts": r["attempts"], "qa_loops": r["qa_loops"],
            "window": f"{p['start']:.1f}-{p['end']:.1f}s" if "start" in p else "",
            "score": p.get("score", ""), "hook_type": d.get("hook_type", ""),
            "qa": (p.get("qa") or {}).get("verdict", ""), "error": (r["last_error"] or "")[:60],
        })
    return out
