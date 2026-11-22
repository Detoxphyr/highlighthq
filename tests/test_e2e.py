"""End-to-end: generate a short synthetic video, run all Phase 1 agents, then
exercise the QA reject loop and the kill-after-max-loops path."""
import json
from pathlib import Path

import pytest

from agents import qa_gate
from agents.ingest import probe
from core import jobs as J
from run import PHASE1, agent_class, run_pipeline
from scripts.make_test_video import main as make_video


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("vid") / "short_talk.mp4", paragraphs=5)


def test_pipeline_end_to_end(settings, conn, video, monkeypatch):
    monkeypatch.setenv("HQ_WHISPER_MODEL", "/nonexistent/model")  # force the transcription fallback offline
    monkeypatch.setenv("HQ_WHISPER_FALLBACK_MODEL", "/nonexistent/model")
    J.add_source(conn, str(video))
    run_pipeline(settings, conn)
    table = J.job_table(conn)
    clips = [r for r in table if r["kind"] == "clip"]
    assert table[0]["status"] == "candidates" and clips
    approved = [r for r in clips if r["status"] == "approved"]
    assert approved, table
    for r in approved:
        p = J.get(conn, r["id"]).payload
        info = probe(Path(p["clip_path"]))
        assert (info["width"], info["height"]) == (1080, 1920) and info["has_audio"]
        assert 15 <= info["duration"] <= 61
        assert Path(p["checkpoint"]).exists()
        assert set(p["packages"]) == {"youtube_shorts", "instagram_reels", "tiktok", "facebook_reels"}
        assert p["qa"]["verdict"] == "approve"
    # every move is in the transitions table
    n = conn.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]
    assert n == 3 + 4 * len(clips)  # source: created, ingested, candidates; clip: created, rendered, packaged, approved

    # --- QA reject loop: reviewer flags packaging, Packager redoes it, QA approves ---
    clip_id = approved[0]["id"]
    job = J.get(conn, clip_id)
    conn.execute("UPDATE jobs SET status='packaged' WHERE id=?", (clip_id,))
    reviews = iter([{"score": 4, "issues": [{"problem": "title overpromises", "responsible": "packager"}], "summary": "x"},
                    {"score": 8, "issues": [], "summary": "ok"}])
    monkeypatch.setattr(qa_gate, "llm_review", lambda *a, **k: next(reviews))
    qa, packager = agent_class("qa_gate")(settings, conn), agent_class("packager")(settings, conn)
    assert qa.run_once().status == "rendered"
    assert packager.run_once().status == "packaged"
    job = qa.run_once()
    assert job.status == "approved" and job.qa_loops == 1
    assert [h["verdict"] for h in job.payload["qa_history"]][-2:] == ["reject", "approve"]

    # --- kill after max loops: reviewer always blames the editor ---
    conn.execute("UPDATE jobs SET status='packaged', qa_loops=0 WHERE id=?", (clip_id,))
    monkeypatch.setattr(qa_gate, "llm_review", lambda *a, **k: {
        "score": 3, "issues": [{"problem": "captions late", "responsible": "clip_editor"}], "summary": "x"})
    statuses = []
    for _ in range(10):
        n = sum(agent_class(a)(settings, conn).run(once=True) for a in PHASE1)
        statuses.append(J.get(conn, clip_id).status)
        if statuses[-1] == "killed" or n == 0:
            break
    job = J.get(conn, clip_id)
    assert job.status == "killed" and job.qa_loops == 2
    assert "max QA loops" in conn.execute(
        "SELECT reason FROM transitions WHERE job_id=? ORDER BY id DESC LIMIT 1", (clip_id,)).fetchone()[0]


def test_bad_source_goes_to_dead_letter(settings, conn):
    jid = J.add_source(conn, "/definitely/missing.mp4")
    ingest = agent_class("ingest")(settings, conn)
    for _ in range(3):
        ingest.run_once()
    job = J.get(conn, jid)
    assert job.status == "dead_letter" and job.attempts == 3
    dl = conn.execute("SELECT * FROM dead_letter WHERE job_id=?", (jid,)).fetchone()
    assert "not found" in dl["error"] and json.loads(dl["logs"])
