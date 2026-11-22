import pytest

from agents import moment_finder as MF
from agents.clip_editor import build_ass, crop_geometry, snap_to_sentences
from agents.packager import enforce, template_package
from agents.qa_gate import Check, check_cutoff, decide, unsynced_ratio
from core.llm import LLMUnavailable
from tests.conftest import FakeLLM, make_words, transcript_from

SENTS = [
    "Welcome back to the show today.",
    "Here's the truth nobody tells you about practice.",
    "More hours do not make you better at anything at all.",
    "Better feedback makes you better and that is the whole point.",
    "And that is why we keep going back to the basics every single day.",
    "What if you only have ten minutes a day to practice?",
    "Ten minutes is plenty if you spend it on the hardest part of the skill.",
    "Skip the warm up you already know by heart.",
    "Go straight to the thing you keep avoiding.",
    "Thanks for listening, see you next week.",
]


# ---------- Moment Finder ----------

def test_split_and_windows_respect_length(playbook):
    words = make_words(SENTS)
    sents = MF.split_sentences(words)
    assert len(sents) == len(SENTS)
    wins = MF.build_windows(sents, 10, 20)
    assert wins and all(10 <= w.duration <= 20 for w in wins)
    assert all(w.sentences[0].words[0]["s"] == w.start for w in wins)


def test_dangling_opener_scores_lower(playbook):
    words = make_words(SENTS)
    sents = MF.split_sentences(words)
    clean = MF.Window(0, sents[1].start, sents[3].end, sents[1:4])
    dangling = MF.Window(1, sents[4].start, sents[6].end, sents[4:7])
    import numpy as np
    for w in (clean, dangling):
        MF.heuristic_score(w, playbook, np.array([]), 0.5, [])
    assert clean.criteria["self_contained"] > dangling.criteria["self_contained"]
    assert clean.hook_type in ("contrarian", "bold_claim")


def test_select_top_no_overlap():
    ws = [MF.Window(i, s, s + 30, []) for i, s in enumerate([0, 5, 40, 45, 90])]
    for w, sc in zip(ws, [90, 89, 70, 95, 10]):
        w.score = sc
    chosen = MF.select_top(ws, 3, 0.25)
    assert [w.start for w in chosen] == [0, 45, 90]


def test_find_moments_llm_and_fallback(playbook):
    playbook = {**playbook, "clip": {**playbook["clip"], "min_s": 8, "max_s": 20, "target_s": 14, "top_n": 2}}
    tr = transcript_from(SENTS)

    def scorer(prompt):
        import json
        items = json.loads(prompt.split("\n\n", 1)[1])
        return {"scores": [{"id": it["id"], "hook": 9 if "?" in it["opening"] else 4, "self_contained": 7,
                            "emotional_peak": 5, "quotability": 6, "hook_type": "question", "reason": "r"}
                           for it in items]}

    chosen, engine = MF.find_moments(tr, {"cuts": [], "loudness_db": []}, playbook, llm=FakeLLM([scorer]))
    assert engine == "llm" and len(chosen) == 2
    assert any(w.sentences[0].text.startswith("What if") for w in chosen)

    chosen, engine = MF.find_moments(tr, {"cuts": [], "loudness_db": []}, playbook,
                                     llm=FakeLLM([LLMUnavailable("down")]))
    assert engine == "heuristic" and chosen


# ---------- Clip Editor ----------

def test_crop_geometry():
    assert crop_geometry(1920, 1080) == (608, 1080)
    assert crop_geometry(1080, 1920) == (1080, 1920)


def test_ass_word_pop(playbook):
    words = make_words(SENTS[:2])
    ass = build_ass(words, 0.0, 10.0, playbook["subtitles"], 1080, 1920)
    events = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(events) == len(words)  # one event per active word
    assert "\\fscx112" in events[0] and playbook["subtitles"]["highlight_color"] in events[0]


def test_snap_to_sentences():
    words = make_words(SENTS)
    sents = MF.split_sentences(words)
    s, e = snap_to_sentences(words, sents[1].start + 0.8, sents[2].end - 0.9)
    assert s == sents[1].start and e == sents[2].end


# ---------- Packager ----------

def test_enforce_limits(rules):
    pcfg = {"title_max": 20, "caption_max": 60, "hashtags": 9}
    out = enforce("x", {"title": "THIS IS A VERY LOUD TITLE INDEED", "caption": "word " * 40,
                        "hashtags": ["#A", "a", "#b!", "#c", "#d", "#e", "#f", "#g"]}, pcfg, rules)
    assert len(out["title"]) <= 20 and not out["title"].isupper()
    assert len(out["hashtags"]) == rules["content"]["max_hashtags_any_platform"]
    assert out["hashtags"][:2] == ["#a", "#b"]
    assert len(out["full_caption"]) <= 60


def test_template_covers_all_platforms(playbook, rules):
    raw = template_package(" ".join(SENTS[1:4]), "contrarian", playbook)
    assert set(raw) == {"youtube_shorts", "instagram_reels", "tiktok", "facebook_reels"}
    pk = enforce("instagram_reels", raw["instagram_reels"], playbook["packaging"]["platforms"]["instagram_reels"], rules)
    assert pk["title"] == "" and "#reels" in pk["hashtags"]


# ---------- QA Gate ----------

def ok(name):
    return Check(name, True, "fine")


def test_decide_routes():
    good = {"score": 8, "issues": [], "summary": ""}
    assert decide([ok("a")], good, 6, 0, 2)[0] == "approved"
    to, _, fb = decide([Check("cut", False, "mid-sentence", "clip_editor", {"snap_boundaries": True})], good, 6, 0, 2)
    assert to == "candidates" and fb == {"snap_boundaries": True}
    to, _, fb = decide([Check("pkg", False, "banned term", "packager")], good, 6, 1, 2)
    assert to == "rendered" and "banned term" in fb["packaging"]
    assert decide([Check("dup", False, "dup", "kill")], good, 6, 0, 2)[0] == "killed"
    assert decide([Check("cut", False, "x", "clip_editor")], good, 6, 2, 2)[0] == "killed"  # loops exhausted
    low_pkg = {"score": 4, "issues": [{"problem": "clickbait title", "responsible": "packager"}], "summary": "meh"}
    assert decide([ok("a")], low_pkg, 6, 0, 2)[0] == "rendered"
    low_moment = {"score": 3, "issues": [{"problem": "boring", "responsible": "moment"}], "summary": "meh"}
    assert decide([ok("a")], low_moment, 6, 0, 2)[0] == "killed"


def test_unsynced_ratio():
    events = [(0.0, 0.4), (0.5, 0.9), (2.0, 2.4)]
    assert unsynced_ratio(events, [(1.8, 3.0)]) == pytest.approx(1 / 3)
    assert unsynced_ratio(events, []) == 0.0


def test_check_cutoff():
    words = make_words(SENTS)
    sents = MF.split_sentences(words)
    good = {"start": sents[1].start, "end": sents[2].end}
    assert check_cutoff(good, words).ok
    bad = {"start": sents[1].words[3]["s"], "end": sents[2].words[4]["e"]}
    c = check_cutoff(bad, words)
    assert not c.ok and c.fix == {"snap_boundaries": True} and c.responsible == "clip_editor"
