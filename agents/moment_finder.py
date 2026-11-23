"""Agent 2: Moment Finder. ingested (source) -> candidates (source), and creates
one clip job per chosen moment in status 'candidates'.

Builds sentence-aligned windows of playbook clip.min_s..max_s, scores them on
hook (first N seconds), self-contained arc, emotional peak and quotability with
the LLM (claude-sonnet-5-5). Falls back to a deterministic heuristic scorer
when the LLM is unavailable. Picks the top N non-overlapping windows."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np

from agents.base import AgentResult, BaseAgent
from core import jobs as J
from core.llm import LLMUnavailable
from core.logging import log

DANGLING_OPENERS = {"and", "but", "so", "because", "which", "that", "also", "then", "or", "plus", "anyway"}
BACKREFS = ("as i said", "like i said", "earlier", "as mentioned", "going back to")
CRITERIA = ("hook", "self_contained", "emotional_peak", "quotability")


@dataclass
class Sentence:
    start: float
    end: float
    words: list[dict]

    @property
    def text(self) -> str:
        return " ".join(w["w"] for w in self.words)


@dataclass
class Window:
    id: int
    start: float
    end: float
    sentences: list[Sentence]
    criteria: dict = field(default_factory=dict)
    hook_type: str = "story"
    score: float = 0.0
    reason: str = ""

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)

    @property
    def duration(self) -> float:
        return self.end - self.start


def all_words(transcript: dict) -> list[dict]:
    return [w for seg in transcript["segments"] for w in seg["words"]]


def split_sentences(words: list[dict], max_gap: float = 1.2) -> list[Sentence]:
    """Sentences end at terminal punctuation or a long pause."""
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        terminal = re.search(r"[.?!][\"')\]]*$", w["w"]) is not None
        long_gap = nxt is not None and nxt["s"] - w["e"] > max_gap
        if terminal or long_gap or nxt is None:
            out.append(Sentence(cur[0]["s"], cur[-1]["e"], cur))
            cur = []
    return out


def build_windows(sentences: list[Sentence], min_s: float, max_s: float, max_windows: int = 4000) -> list[Window]:
    """All sentence-aligned windows of min_s..max_s. Over max_windows, keep an evenly spaced subset so the
    whole video stays covered (truncating would silently drop everything after the first N windows)."""
    wins: list[Window] = []
    for i in range(len(sentences)):
        for j in range(i, len(sentences)):
            dur = sentences[j].end - sentences[i].start
            if dur > max_s:
                break
            if dur >= min_s:
                wins.append(Window(len(wins), sentences[i].start, sentences[j].end, sentences[i:j + 1]))
    if len(wins) > max_windows:
        keep = np.linspace(0, len(wins) - 1, max_windows).round().astype(int)
        wins = [wins[k] for k in sorted(set(keep.tolist()))]
        for n, w in enumerate(wins):
            w.id = n
    return wins


def near_peaks(windows: list[Window], peaks: list[float], radius_s: float) -> list[Window]:
    """Keep windows that lie entirely within radius_s of some peak (seconds into the source)."""
    keep = [w for w in windows if any(p - radius_s <= w.start and w.end <= p + radius_s for p in peaks)]
    for n, w in enumerate(keep):
        w.id = n
    return keep


def load_peaks(path: str) -> list[float]:
    """Peaks file: kai_scan.py output ([{"peak": s, ...}]), {"peaks": [...]}, or a plain list of seconds."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("peaks", [])
    out = []
    for item in data:
        v = item.get("peak", item.get("start")) if isinstance(item, dict) else item
        if v is not None:
            out.append(float(v))
    return sorted(out)


def overlap_ratio(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-6, min(a[1] - a[0], b[1] - b[0]))


# ---------- heuristic scorer (fallback path) ----------

def classify_hook(text_first: str, strong_openers: list[str]) -> tuple[str, float]:
    t = text_first.lower()
    if "?" in t:
        return "question", 0.9
    if re.search(r"\b\d+(\.\d+)?%?\b|\b(one|two|three|five|ten|hundred|thousand|million)\b", t):
        return "number", 0.75
    if re.search(r"\b(not|never|nobody|wrong|myth|stop)\b", t):
        return "contrarian", 0.8
    if any(o in t for o in strong_openers):
        return "bold_claim", 0.7
    if re.search(r"\b(i was|when i|one day|years ago|last year)\b", t):
        return "story", 0.55
    return "story", 0.3


def heuristic_score(win: Window, playbook: dict, loud: np.ndarray, hop: float, cuts: list[float]) -> None:
    hook_cfg = playbook["hook"]
    first = [w["w"] for w in win.sentences[0].words if w["s"] - win.start <= hook_cfg["window_s"]]
    hook_type, hook = classify_hook(" ".join(first) + (" ?" if win.sentences[0].text.endswith("?")
                                    and win.sentences[0].end - win.start <= hook_cfg["window_s"] + 2 else ""),
                                    hook_cfg["strong_openers"])
    first_word = re.sub(r"\W", "", win.sentences[0].words[0]["w"].lower())
    sc = 0.6
    sc -= 0.35 if first_word in DANGLING_OPENERS else 0
    sc -= 0.25 if any(b in win.text.lower() for b in BACKREFS) else 0
    sc += 0.2 if re.search(r"[.?!]$", win.sentences[-1].words[-1]["w"]) else -0.2
    sc += 0.1 if any(abs(c - win.start) < 1.0 for c in cuts) else 0
    sc += 0.1 if len(win.sentences) >= 3 else 0
    if loud.size:
        a, b = int(win.start / hop), max(int(win.start / hop) + 1, int(win.end / hop))
        seg = loud[a:b]
        peak = float(np.clip((seg.max() - np.median(loud)) / 12.0, 0, 1)) if seg.size else 0.0
        dyn = float(np.clip(seg.std() / 8.0, 0, 1)) if seg.size else 0.0
    else:
        peak = dyn = 0.0
    emo = 0.5 * peak + 0.3 * dyn + 0.2 * min(1.0, win.text.count("!") / 2)
    short = [s for s in win.sentences if 4 <= len(s.words) <= 14]
    quote = 0.3 + 0.3 * min(1, len(short) / 2) + 0.2 * ("you" in win.text.lower().split()) \
        + 0.2 * bool(re.search(r"\bnot\b.*\bbut\b|\bthe (real|only|biggest)\b", win.text.lower()))
    win.criteria = {"hook": round(hook * 10, 2), "self_contained": round(float(np.clip(sc, 0, 1)) * 10, 2),
                    "emotional_peak": round(emo * 10, 2), "quotability": round(min(quote, 1.0) * 10, 2)}
    win.hook_type = hook_type
    win.reason = (f"heuristic: {hook_type} hook, {len(win.sentences)} sentences, "
                  f"peak loudness {peak:.2f}, opener '{first_word}'")


def finalize_score(win: Window, playbook: dict) -> None:
    w = playbook["scoring_weights"]
    base = sum(w[c] * win.criteria[c] for c in CRITERIA) * 10  # 0..100
    target, span = playbook["clip"]["target_s"], playbook["clip"]["max_s"] - playbook["clip"]["min_s"]
    length_factor = 1 - 0.15 * min(1.0, abs(win.duration - target) / max(span, 1))
    win.score = round(base * length_factor, 2)


# ---------- LLM scorer (primary path) ----------

LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "hook": {"type": "number"},
                    "self_contained": {"type": "number"},
                    "emotional_peak": {"type": "number"},
                    "quotability": {"type": "number"},
                    "hook_type": {"type": "string", "enum": ["question", "bold_claim", "number", "story", "contrarian"]},
                    "reason": {"type": "string"},
                },
                "required": ["id", "hook", "self_contained", "emotional_peak", "quotability", "hook_type", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}


def llm_score(llm, windows: list[Window], playbook: dict) -> None:
    hook_s = playbook["hook"]["window_s"]
    system = (
        "You are the Moment Finder for a short-form video clipping pipeline. You score candidate segments of a "
        "long video transcript for their potential as a standalone vertical short.\n"
        "Score each criterion from 0 to 10:\n"
        f"- hook: does something in the first {hook_s} seconds make a scroller stop (question, bold claim, number, "
        "contrarian take, story opener)? Only the opening counts.\n"
        "- self_contained: can a viewer with no context follow it start to finish, with a clear arc and no dangling "
        "references to earlier parts?\n"
        "- emotional_peak: is there a moment of tension, surprise, humor or conviction?\n"
        "- quotability: is there a line people would repeat or share?\n"
        "Be strict: most segments should score below 7. Give a one-sentence reason naming the strongest line."
    )
    creator = (playbook.get("packaging") or {}).get("creator_name", "").strip()
    if creator:
        system += (f"\nThis is a {creator} stream. Favour funny moments, memorable quotes and big reactions from "
                   f"{creator} himself. Score segments down (self_contained and emotional_peak) when {creator} is not "
                   "the focus: reading chat or menus, other people talking, quiet gameplay, or setup chatter.")
    items = []
    for w in windows:
        opening = " ".join(x["w"] for s in w.sentences for x in s.words if x["s"] - w.start <= hook_s)
        items.append({"id": w.id, "duration_s": round(w.duration, 1), "opening": opening, "text": w.text})
    prompt = "Score every candidate below. Return one entry per id.\n\n" + json.dumps(items, indent=1)
    data = llm.complete_json(system, prompt, LLM_SCHEMA, max_tokens=16000)
    by_id = {w.id: w for w in windows}
    seen = 0
    for s in data["scores"]:
        w = by_id.get(s["id"])
        if not w:
            continue
        w.criteria = {c: float(np.clip(s[c], 0, 10)) for c in CRITERIA}
        w.hook_type, w.reason = s["hook_type"], "llm: " + s["reason"]
        seen += 1
    if seen < len(windows) * 0.8:
        raise LLMUnavailable(f"LLM scored only {seen}/{len(windows)} candidates")


def select_top(windows: list[Window], n: int, max_overlap: float) -> list[Window]:
    chosen: list[Window] = []
    for w in sorted(windows, key=lambda x: x.score, reverse=True):
        if all(overlap_ratio((w.start, w.end), (c.start, c.end)) <= max_overlap for c in chosen):
            chosen.append(w)
        if len(chosen) == n:
            break
    return sorted(chosen, key=lambda x: x.start)


def peak_windows(sentences: list[Sentence], peaks: list[float], radius_s: float, min_s: float, max_s: float) -> list[Window]:
    """Windows built only from sentences inside the (merged) regions peak +/- radius_s."""
    regions: list[list[float]] = []
    for p in sorted(peaks):
        a, b = p - radius_s, p + radius_s
        if regions and a <= regions[-1][1]:
            regions[-1][1] = max(regions[-1][1], b)
        else:
            regions.append([a, b])
    wins: list[Window] = []
    for a, b in regions:
        inside = [s for s in sentences if a <= s.start and s.end <= b]
        wins += build_windows(inside, min_s, max_s)
    return near_peaks(wins, peaks, radius_s)


def find_moments(transcript: dict, scenes: dict, playbook: dict, llm=None, llm_pool: int = 25,
                 peaks: list[float] | None = None) -> tuple[list[Window], str]:
    cfg = playbook["clip"]
    sentences = split_sentences(all_words(transcript))
    if peaks:
        windows = peak_windows(sentences, peaks, float(cfg.get("peak_radius_s", 45)), cfg["min_s"], cfg["max_s"])
    else:
        windows = build_windows(sentences, cfg["min_s"], cfg["max_s"])
    if not windows:
        return [], "none"
    loud = np.array(scenes.get("loudness_db") or [], dtype=float)
    for w in windows:
        heuristic_score(w, playbook, loud, scenes.get("hop_s", 0.5), scenes.get("cuts", []))
        finalize_score(w, playbook)
    engine = "heuristic"
    if llm is not None:
        # Pre-filter with the heuristic to keep the LLM call small, keeping diversity across the video.
        pool = select_top(windows, llm_pool, 0.6)
        try:
            llm_score(llm, pool, playbook)
            for w in pool:
                finalize_score(w, playbook)
            windows, engine = pool, "llm"
        except LLMUnavailable as exc:
            log("fallback", level="warning", what="moment_scoring", error=str(exc))
            # restore heuristic scores for the pool
            for w in pool:
                heuristic_score(w, playbook, loud, scenes.get("hop_s", 0.5), scenes.get("cuts", []))
                finalize_score(w, playbook)
    return select_top(windows, cfg["top_n"], cfg["max_overlap"]), engine


class MomentFinderAgent(BaseAgent):
    name = "moment_finder"
    input_status = "ingested"
    kind = "source"

    def process(self, job):
        transcript = json.loads(open(job.payload["transcript"], encoding="utf-8").read())
        scenes = json.loads(open(job.payload["scenes"], encoding="utf-8").read())
        peaks = job.payload.get("peaks") or None
        chosen, engine = find_moments(transcript, scenes, self.playbook, llm=self.llm, peaks=peaks)
        if not chosen:
            why = "near the given peaks" if peaks else "in the transcript"
            return AgentResult("killed", reason=f"no segment fits the playbook length window {why}")
        ids = []
        for w in chosen:
            ids.append(J.create_job(
                self.conn, kind="clip", source_id=job.source_id, parent_job_id=job.id, status="candidates",
                agent=self.name,
                payload={"start": round(w.start, 3), "end": round(w.end, 3), "score": w.score, "reason": w.reason,
                         "criteria": w.criteria, "text": w.text, "media": job.payload["media"],
                         "transcript": job.payload["transcript"], "scenes": job.payload["scenes"]},
                decisions={"hook_type": w.hook_type, "length_s": round(w.duration, 1),
                           "playbook_version": self.playbook["version"], "scoring_engine": engine}))
        return AgentResult("candidates", reason=f"{len(ids)} candidates via {engine}: clip jobs {ids}",
                           payload={"clip_jobs": ids, "scoring_engine": engine, "used_peaks": bool(peaks)})


def main():  # pragma: no cover
    MomentFinderAgent().run()


if __name__ == "__main__":  # pragma: no cover
    main()
