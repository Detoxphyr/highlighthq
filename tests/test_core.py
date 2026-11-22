import json

import pytest

from core import jobs as J
from core.alerts import raise_alert, write_digest
from core.breaker import CircuitBreaker
from core.retry import PersistentError, TransientError, retry, with_fallback


def test_state_machine_happy_path(conn):
    jid = J.add_source(conn, "/tmp/x.mp4")
    job = J.get(conn, jid)
    assert job.status == "scouted"
    job = J.transition(conn, job, "ingested", "ingest")
    with pytest.raises(J.InvalidTransition):
        J.transition(conn, job, "approved", "qa_gate")  # cannot skip stages
    job = J.transition(conn, job, "candidates", "moment_finder")
    rows = conn.execute("SELECT from_status, to_status FROM transitions WHERE job_id=? ORDER BY id", (jid,)).fetchall()
    assert [(r[0], r[1]) for r in rows] == [(None, "scouted"), ("scouted", "ingested"), ("ingested", "candidates")]


def test_claim_is_exclusive(conn):
    J.add_source(conn, "a.mp4")
    first = J.claim(conn, "scouted", "source", "ingest")
    second = J.claim(conn, "scouted", "source", "ingest")
    assert first is not None and second is None
    J.release(conn, first)
    assert J.claim(conn, "scouted", "source", "ingest").id == first.id


def test_fail_requeues_then_dead_letters(conn):
    jid = J.add_source(conn, "a.mp4")
    for attempt in range(1, 4):
        job = J.claim(conn, "scouted", "source", "ingest")
        assert job is not None, f"attempt {attempt} should be claimable"
        job = J.fail(conn, job, "ingest", "boom", "scouted")
    assert job.status == "dead_letter"
    dl = conn.execute("SELECT * FROM dead_letter WHERE job_id=?", (jid,)).fetchone()
    assert dl["error"] == "boom" and isinstance(json.loads(dl["logs"]), list)
    assert J.claim(conn, "scouted", "source", "ingest") is None


def test_retry_backoff_and_persistent():
    calls, sleeps = [], []

    @retry(tries=3, base_delay=1, sleep=sleeps.append)
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise TransientError("blip")
        return "ok"

    assert flaky() == "ok" and len(calls) == 3
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]  # exponential

    @retry(tries=3, sleep=sleeps.append)
    def broken():
        calls.append(1)
        raise PersistentError("bad input")

    n = len(calls)
    with pytest.raises(PersistentError):
        broken()
    assert len(calls) == n + 1  # no retry on persistent errors


def test_with_fallback():
    assert with_fallback(lambda: 1, lambda: 2, "x") == (1, "primary")
    assert with_fallback(lambda: 1 / 0, lambda: 2, "x") == (2, "fallback")


def test_breaker_trips_alerts_and_recovers(conn, settings):
    now = [1000.0]
    tripped = []
    br = CircuitBreaker(conn, "platform:tiktok:acct1", threshold=2, cooldown_s=60, clock=lambda: now[0],
                        on_trip=lambda k, f: tripped.append(raise_alert(conn, settings.alerts_dir, "breaker_tripped", k)))
    br.failure()
    assert br.allow()
    br.failure()
    assert not br.allow() and tripped and tripped[0].exists()
    now[0] += 61
    assert br.state == "half_open" and br.allow()
    br.failure()  # failure while half-open re-opens immediately
    assert br.state == "open"
    now[0] += 61
    br.success()
    assert br.state == "closed"


def test_only_escalation_kinds_hit_alerts_dir(conn, settings):
    assert raise_alert(conn, settings.alerts_dir, "render_slow", "meh") is None
    assert raise_alert(conn, settings.alerts_dir, "account_strike", "strike on yt").exists()


def test_digest_flags_high_failure_rate(conn, settings):
    for i in range(6):
        jid = J.add_source(conn, f"{i}.mp4")
        job = J.claim(conn, "scouted", "source", "ingest")
        J.fail(conn, job, "ingest", "boom", "scouted")
    path = write_digest(conn, settings.alerts_dir)
    assert "daily digest" in path.read_text()
    assert list(settings.alerts_dir.glob("alert-*-failure_rate.json"))
