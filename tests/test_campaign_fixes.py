"""Network-free tests for the Windows real-run fixes: peaks input, full-coverage windows, campaign packaging,
QA prompt and campaign rule checks, watermark switch, reframe track and frame-rate mapping."""
import json

import numpy as np
import pytest

from agents import moment_finder as MF
from agents.clip_editor import _filter_graph, plan_track, source_frame_index
from agents.packager import enforce, llm_package, pkg_schema, template_package
from agents.qa_gate import Check, check_campaign, check_rules, decide, llm_review, review_packages
from core import jobs as J
from tests.conftest import FakeLLM, transcript_from

SENTS = [f"Sentence number {i} is right here and it keeps going for a while." for i in range(80)]


# ---------- 1. peaks ----------

def test_load_peaks_formats(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps([{"start": 100, "peak": 112, "score": 3.1}, {"start": 400, "peak": 405}]))
    assert MF.load_peaks(str(f)) == [112.0, 405.0]
    f.write_text(json.dumps({"peaks": [30, 10]}))
    assert MF.load_peaks(str(f)) == [10.0, 30.0]
    f.write_text(json.dumps([50, 20.5]))
    assert MF.load_peaks(str(f)) == [20.5, 50.0]


def test_find_moments_only_near_peaks(playbook):
    pb = {**playbook, "clip": {**playbook["clip"], "min_s": 8, "max_s": 20, "target_s": 14, "top_n": 3,
                               "peak_radius_s": 15}}
    tr = transcript_from(SENTS)                       # ~5 s per sentence, ~400 s total
    peak = 200.0
    chosen, engine = MF.find_moments(tr, {"cuts": [], "loudness_db": []}, pb, peaks=[peak])
    assert engine == "heuristic" and chosen
    assert all(peak - 15 <= w.start and w.end <= peak + 15 for w in chosen)
    no_peaks, _ = MF.find_moments(tr, {"cuts": [], "loudness_db": []}, pb)
    assert any(w.end < peak - 15 or w.start > peak + 15 for w in no_peaks)   # default unchanged: whole video


def test_add_source_stores_peaks(conn):
    job_id = J.add_source(conn, "video.mp4", "t", peaks=[30, 10], source_date="2026-11-19")
    p = J.get(conn, job_id).payload
    assert p["peaks"] == [10.0, 30.0] and p["source_date"] == "2026-11-19"
    assert "peaks" not in J.get(conn, J.add_source(conn, "other.mp4")).payload


def test_build_windows_cap_keeps_whole_video():
    sents = MF.split_sentences(transcript_from(SENTS)["segments"][0]["words"])
    full = MF.build_windows(sents, 8, 40, max_windows=10 ** 6)
    capped = MF.build_windows(sents, 8, 40, max_windows=50)
    assert len(full) > 50 and len(capped) == 50
    assert capped[-1].end == full[-1].end                    # still reaches the end of the video
    assert [w.id for w in capped] == list(range(50))


# ---------- 2. campaign packaging ----------

def test_required_hashtags_first_and_capped(rules):
    pcfg = {"title_max": 50, "caption_max": 280, "hashtags": 2}
    out = enforce("x", {"title": "", "caption": "Kai Cenat goes crazy", "hashtags": ["#gta6", "#kaicenat", "#fyp"]},
                  pcfg, rules, ["#KAICENAT"])
    assert out["hashtags"] == ["#KAICENAT", "#gta6"]          # first, case kept, deduped, counted toward cap
    assert len(out["full_caption"]) <= 280


def test_playbook_campaign_settings(playbook):
    pk = playbook["packaging"]
    assert pk["creator_name"] == "Kai Cenat" and pk["required_hashtags"] == ["#KAICENAT"]
    assert pk["platforms"]["youtube_shorts"]["title_max"] == 50
    assert set(pk["platforms"]) == {"youtube_shorts", "instagram_reels", "tiktok", "x"}
    assert pk["platforms"]["x"]["title_max"] == 0 and pk["platforms"]["x"]["caption_max"] == 280


def test_template_names_creator(playbook):
    raw = template_package("Here is the truth about the heist. It went wrong fast.", "story", playbook)
    assert all("Kai Cenat" in raw[p]["caption"] for p in raw)
    assert "Kai Cenat" in raw["youtube_shorts"]["title"]


def test_llm_package_prompt_and_schema(playbook, rules):
    fake = FakeLLM([{p: {"title": "", "caption": "Kai Cenat", "hashtags": []} for p in playbook["packaging"]["platforms"]}])
    llm_package(fake, "text", "story", playbook, rules, None)
    call = fake.calls[0]
    assert "Kai Cenat" in call["system"] and "streamer" in call["system"] and "positive" in call["system"]
    assert set(call["schema"]["required"]) == set(playbook["packaging"]["platforms"])
    assert pkg_schema(["a", "b"])["required"] == ["a", "b"]


# ---------- 3. QA prompt ----------

def _pkgs():
    return {p: {"title": "T" if p == "youtube_shorts" else "", "full_caption": "Kai Cenat wins #KAICENAT",
                "hashtags": ["#KAICENAT"]} for p in ("youtube_shorts", "instagram_reels", "tiktok", "x")}


def test_review_sends_titles_only_where_they_exist(playbook, rules):
    p = {"text": "hi", "start": 0, "end": 20, "packages": _pkgs()}
    shown = review_packages(p, playbook)
    assert "title" in shown["youtube_shorts"]
    assert all("title" not in shown[k] for k in ("instagram_reels", "tiktok", "x"))
    fake = FakeLLM([{"score": 7, "issues": [], "summary": "", "violations": []}])
    llm_review(fake, p, rules, playbook)
    assert "by design" in fake.calls[0]["system"] and "kai_primary_focus" in fake.calls[0]["system"]
    assert "violations" in fake.calls[0]["schema"]["required"]


# ---------- 4. watermark switch ----------

def test_filter_graph_without_watermark(tmp_path):
    g = _filter_graph(tmp_path / "s.ass", {"enabled": False, "scale": 0.22}, 1080)
    assert "overlay" not in g and "[2:v]" not in g and g.endswith("[v]")
    assert "overlay" in _filter_graph(tmp_path / "s.ass", {"enabled": True, "scale": 0.22}, 1080)


def test_watermark_check_skipped_when_disabled(rules, playbook, tmp_path):
    p = {"text": "hello", "packages": _pkgs(), "render_mode": "tracked"}
    strict = {**rules, "qa": {**rules["qa"], "require_watermark": True}}
    names = [c.name for c in check_rules(p, strict, tmp_path, {**playbook, "watermark": {"enabled": False, "path": "x"}})]
    assert "rules_watermark" not in names
    names = [c.name for c in check_rules(p, strict, tmp_path, {**playbook, "watermark": {"enabled": True, "path": "x"}})]
    assert "rules_watermark" in names


# ---------- campaign rule checks (rules.yaml) ----------

def _by_name(checks):
    return {c.name: c for c in checks}


def test_campaign_rules_in_rules_yaml(rules):
    camp = rules["campaign"]
    assert camp["required_mentions"] == ["Kai Cenat"] and camp["required_hashtags"] == ["#KAICENAT"]
    assert "facebook_reels" not in camp["allowed_platforms"] and "x" in camp["allowed_platforms"]
    assert camp["no_logos"] is True and camp["language"] == "en"


def test_campaign_checks_pass_for_good_packages(rules, playbook):
    checks = check_campaign({"packages": _pkgs(), "watermark": False}, rules, playbook)
    assert checks and all(c.ok for c in checks)


def test_campaign_missing_name(rules, playbook):
    pk = _pkgs()
    pk["tiktok"]["full_caption"] = "This streamer is wild #KAICENAT"
    c = _by_name(check_campaign({"packages": pk, "watermark": False}, rules, playbook))["campaign_mentions"]
    assert not c.ok and "tiktok" in c.detail and c.responsible == "packager"


def test_campaign_missing_hashtag(rules, playbook):
    pk = _pkgs()
    pk["x"]["hashtags"] = ["#gta6"]
    c = _by_name(check_campaign({"packages": pk, "watermark": False}, rules, playbook))["campaign_hashtags"]
    assert not c.ok and "x" in c.detail and c.responsible == "packager"


def test_campaign_disallowed_platform(rules, playbook):
    pk = {**_pkgs(), "facebook_reels": {"title": "", "full_caption": "Kai Cenat #KAICENAT", "hashtags": ["#KAICENAT"]}}
    c = _by_name(check_campaign({"packages": pk, "watermark": False}, rules, playbook))["campaign_platforms"]
    assert not c.ok and "facebook_reels" in c.detail and c.responsible == "kill"


def test_campaign_watermark_while_no_logos(rules, playbook):
    c = _by_name(check_campaign({"packages": _pkgs(), "watermark": True}, rules, playbook))["campaign_no_logos"]
    assert not c.ok and c.responsible == "clip_editor"


def test_campaign_japanese_text(rules, playbook):
    pk = _pkgs()
    pk["tiktok"]["full_caption"] = "Kai Cenat すごい #KAICENAT"
    c = _by_name(check_campaign({"packages": pk, "watermark": False}, rules, playbook))["campaign_language"]
    assert not c.ok and "tiktok" in c.detail


def test_content_rule_violation_kills():
    review = {"score": 9, "issues": [], "summary": "", "violations": ["no_negative_portrayal"]}
    to, reason, _ = decide([Check("a", True, "ok")], review, 6, 0, 2)
    assert to == "killed" and "no_negative_portrayal" in reason


# ---------- 5. reframe track + frame mapping ----------

def test_plan_track_prefers_persistent_face_and_keeps_it_in_crop():
    w, crop_w, n = 1920, 608, 40
    faces = [[] for _ in range(n)]
    for i in range(20, 40):
        faces[i] = [(950.0, 1250.0)]                  # persistent face, center 1100
    faces[3] = [(100.0, 400.0)]                       # isolated false positive
    faces[25] = [(950.0, 1250.0), (1500.0, 1900.0)]   # bigger second detection must not win
    fallback = [1700.0] * n                           # saliency pulled right by a chat overlay
    cx, engine, ratio = plan_track(faces, fallback, ["saliency"] * n, w, crop_w)
    assert engine == "face" and ratio == pytest.approx(20 / 40)
    for i, c in enumerate(cx):                        # whole face box inside the crop, gaps filled from the face
        assert c - crop_w / 2 <= 950 and 1250 <= c + crop_w / 2, (i, c)


def test_plan_track_falls_back_without_faces():
    cx, engine, ratio = plan_track([[] for _ in range(10)], [960.0] * 10, ["saliency"] * 10, 1920, 608)
    assert engine == "saliency" and ratio == 0 and all(abs(c - 960) < 1e-6 for c in cx)


def test_source_frame_index_keeps_real_time():
    assert [source_frame_index(i, 30, 60) for i in range(4)] == [0, 2, 4, 6]   # 60 fps source, 30 fps output
    assert [source_frame_index(i, 30, 30) for i in range(3)] == [0, 1, 2]
    assert source_frame_index(30, 30, 59.94) == 60
