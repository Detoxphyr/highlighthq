"""Agent 5: QA Gate. packaged -> approved | candidates (Clip Editor) |
rendered (Packager) | killed.

Deterministic checklist (file/format, audio-caption sync, cut-off sentences,
black frames, rules.yaml compliance, duplicates) plus an independent LLM
review. Rejections are routed to the responsible agent's input status with a
machine-readable fix. After rules.qa.max_qa_loops rejections the clip is killed."""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from agents.base import AgentResult, BaseAgent
from agents.ingest import probe
from agents.moment_finder import all_words
from core.db import loads
from core.llm import LLMUnavailable
from core.logging import log
from core.retry import run_cmd

ROUTE = {"clip_editor": "candidates", "packager": "rendered"}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    responsible: str = ""          # clip_editor | packager | kill
    fix: dict = field(default_factory=dict)


# ---------- media helpers ----------

def silences(path: str, noise_db: int = -35, min_d: float = 0.25) -> list[tuple[float, float]]:
    proc = run_cmd(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn",
                    "-af", f"silencedetect=noise={noise_db}dB:d={min_d}", "-f", "null", "-"])
    text = proc.stderr.decode(errors="replace")
    starts = [float(x) for x in re.findall(r"silence_start: (-?[0-9.]+)", text)]
    ends = [float(x) for x in re.findall(r"silence_end: ([0-9.]+)", text)]
    if len(ends) < len(starts):
        ends.append(float("inf"))
    return list(zip(starts, ends))


def black_intervals(path: str) -> list[tuple[float, float]]:
    proc = run_cmd(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-an",
                    "-vf", "blackdetect=d=0.1:pix_th=0.10", "-f", "null", "-"])
    text = proc.stderr.decode(errors="replace")
    return [(float(a), float(b)) for a, b in re.findall(r"black_start:([0-9.]+) black_end:([0-9.]+)", text)]


def ass_events(path: str) -> list[tuple[float, float]]:
    def t(s):
        h, m, sec = s.split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("Dialogue:"):
            parts = line.split(",", 9)
            out.append((t(parts[1]), t(parts[2])))
    return out


def unsynced_ratio(events: list[tuple[float, float]], sil: list[tuple[float, float]], shift: float = 0.0) -> float:
    """Share of caption events whose whole span sits inside detected silence."""
    if not events:
        return 1.0
    bad = 0
    for s, e in events:
        s, e = s + shift, e + shift
        if any(a <= s and e <= b for a, b in sil):
            bad += 1
    return bad / len(events)


# ---------- checks ----------

def check_format(p: dict, rules: dict) -> Check:
    path = p.get("clip_path")
    if not path or not Path(path).exists():
        return Check("format", False, "clip file missing", "clip_editor")
    info = probe(Path(path))
    rc = rules["clip"]
    problems = []
    if not (rc["min_duration_s"] <= info["duration"] <= rc["max_duration_s"]):
        problems.append(f"duration {info['duration']:.1f}s outside {rc['min_duration_s']}-{rc['max_duration_s']}s")
    if (info["width"], info["height"]) != (rc["width"], rc["height"]):
        problems.append(f"resolution {info['width']}x{info['height']} != {rc['width']}x{rc['height']}")
    if not info["has_audio"]:
        problems.append("no audio")
    return Check("format", not problems, "; ".join(problems) or f"{info['width']}x{info['height']}, "
                 f"{info['duration']:.1f}s", "clip_editor" if problems else "")


def check_sync(p: dict, rules: dict) -> Check:
    events = ass_events(p["subtitles"])
    sil = silences(p["clip_path"])
    ratio = unsynced_ratio(events, sil)
    limit = rules["qa"]["max_unsynced_word_ratio"]
    if ratio <= limit:
        return Check("audio_caption_sync", True, f"{ratio:.0%} of {len(events)} caption events in silence")
    shifts = np.round(np.arange(-0.6, 0.61, 0.1), 2)
    best = min(shifts, key=lambda s: unsynced_ratio(events, sil, float(s)))
    return Check("audio_caption_sync", False, f"{ratio:.0%} of caption events land in silence (limit {limit:.0%})",
                 "clip_editor", {"subtitle_shift_s": round(float(p.get("subtitle_shift_s", 0)) + float(best), 2)})


def check_cutoff(p: dict, words: list[dict]) -> Check:
    start, end = p["start"], p["end"]
    inside = [i for i, w in enumerate(words) if w["s"] >= start - 0.05 and w["e"] <= end + 0.05]
    if not inside:
        return Check("no_cutoff_sentences", False, "no words inside the clip window", "kill")
    i0, i1 = inside[0], inside[-1]
    problems = []
    if i0 > 0:
        prev = words[i0 - 1]
        if not re.search(r"[.?!][\"')\]]*$", prev["w"]) and words[i0]["s"] - prev["e"] < 1.2:
            problems.append(f"starts mid-sentence at '{words[i0]['w']}'")
    if not re.search(r"[.?!][\"')\]]*$", words[i1]["w"]):
        nxt = words[i1 + 1] if i1 + 1 < len(words) else None
        if nxt is not None and nxt["s"] - words[i1]["e"] < 1.2:
            problems.append(f"ends mid-sentence at '{words[i1]['w']}'")
    if p.get("render_start", start) > words[i0]["s"] + 0.01 or p.get("render_end", end) < words[i1]["e"] - 0.01:
        problems.append("render window clips the first or last word")
    return Check("no_cutoff_sentences", not problems, "; ".join(problems) or "starts and ends on sentence boundaries",
                 "clip_editor" if problems else "", {"snap_boundaries": True} if problems else {})


def check_black(p: dict, rules: dict) -> Check:
    blacks = black_intervals(p["clip_path"])
    worst = max((b - a for a, b in blacks), default=0.0)
    ok = worst <= rules["qa"]["max_black_s"]
    return Check("no_black_frames", ok, f"longest black interval {worst:.2f}s", "" if ok else "clip_editor",
                 {} if ok else {"snap_boundaries": True})


def check_rules(p: dict, rules: dict, settings_root: Path, playbook: dict) -> list[Check]:
    out = []
    banned = [t.lower() for t in rules["content"]["banned_terms"]]
    spoken = [t for t in banned if t in p["text"].lower()]
    out.append(Check("rules_spoken_content", not spoken, f"banned terms spoken: {spoken}" if spoken else "clean",
                     "kill" if spoken else ""))
    problems = []
    for plat, pkg in (p.get("packages") or {}).items():
        blob = f"{pkg['title']} {pkg['full_caption']}".lower()
        hits = [t for t in banned if t in blob]
        if hits:
            problems.append(f"{plat}: banned terms {hits}")
        if len(pkg["hashtags"]) > rules["content"]["max_hashtags_any_platform"]:
            problems.append(f"{plat}: {len(pkg['hashtags'])} hashtags")
        if rules["content"].get("disallow_all_caps_titles") and pkg["title"] and pkg["title"].isupper():
            problems.append(f"{plat}: all-caps title")
        if not pkg["full_caption"]:
            problems.append(f"{plat}: empty caption")
    out.append(Check("rules_packaging", not problems, "; ".join(problems) or "all platforms within rules",
                     "packager" if problems else ""))
    if rules["qa"].get("require_watermark"):
        wm = settings_root / playbook["watermark"]["path"]
        ok = wm.exists() and p.get("render_mode") in ("tracked", "static_center")
        out.append(Check("rules_watermark", ok, "watermark overlaid" if ok else "watermark missing",
                         "" if ok else "clip_editor"))
    return out


def check_duplicate(conn, job, rules: dict) -> Check:
    rows = conn.execute(
        "SELECT id, source_id, payload FROM jobs WHERE kind='clip' AND id<>? AND status IN "
        "('rendered','packaged','approved','posted','submitted','tracked')", (job.id,)).fetchall()
    me = job.payload
    for r in rows:
        other = loads(r["payload"])
        if "start" not in other:
            continue
        if r["source_id"] == job.source_id:
            inter = max(0.0, min(me["end"], other["end"]) - max(me["start"], other["start"]))
            union = max(me["end"], other["end"]) - min(me["start"], other["start"])
            if union > 0 and inter / union > rules["qa"]["duplicate_overlap"]:
                return Check("duplicate", False, f"overlaps clip {r['id']} by {inter / union:.0%}", "kill")
        sim = difflib.SequenceMatcher(None, me["text"][:600].lower(), other.get("text", "")[:600].lower()).ratio()
        if sim > rules["qa"]["duplicate_text_similarity"]:
            return Check("duplicate", False, f"text {sim:.0%} similar to clip {r['id']}", "kill")
    return Check("duplicate", True, "no duplicate found")


REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "description": "1-10 overall quality as a standalone short"},
        "issues": {"type": "array", "items": {"type": "object", "properties": {
            "problem": {"type": "string"},
            "responsible": {"type": "string", "enum": ["moment", "clip_editor", "packager"]}},
            "required": ["problem", "responsible"], "additionalProperties": False}},
        "summary": {"type": "string"},
    },
    "required": ["score", "issues", "summary"],
    "additionalProperties": False,
}


def llm_review(llm, p: dict, rules: dict) -> dict:
    system = ("You are an independent QA reviewer for short vertical clips cut from long videos. You did not choose "
              "or package this clip. Judge it strictly as a viewer scrolling a feed would: does it hook in the first "
              "3 seconds, make sense without context, end cleanly, and are the titles/captions accurate to what is "
              "said? Score 1-10. List concrete issues and who is responsible: 'moment' (bad choice of segment), "
              "'clip_editor' (boundaries, captions, framing), 'packager' (titles, captions, hashtags).")
    prompt = json.dumps({"transcript": p["text"], "duration_s": round(p["end"] - p["start"], 1),
                         "packages": {k: {"title": v["title"], "caption": v["full_caption"]}
                                      for k, v in (p.get("packages") or {}).items()}}, indent=1, ensure_ascii=False)
    return llm.complete_json(system, prompt, REVIEW_SCHEMA, max_tokens=4000, effort="medium")


def heuristic_review(p: dict) -> dict:
    c = p.get("criteria") or {}
    score = round((c.get("hook", 5) + c.get("self_contained", 5) + c.get("quotability", 5)) / 3)
    return {"score": int(score), "issues": [], "summary": "LLM unavailable; heuristic review from Moment Finder criteria"}


def decide(checks: list[Check], review: dict, min_score: int, qa_loops: int, max_loops: int) -> tuple[str, str, dict]:
    """Return (to_status, reason, qa_feedback)."""
    failed = [c for c in checks if not c.ok]
    if review["score"] < min_score:
        resp = {i["responsible"] for i in review.get("issues", [])}
        target = "packager" if resp == {"packager"} else ("clip_editor" if resp == {"clip_editor"} else "kill")
        failed.append(Check("llm_review", False, f"score {review['score']} < {min_score}: {review['summary']}", target,
                            {"packaging": "; ".join(i["problem"] for i in review["issues"])} if target == "packager" else {}))
    if not failed:
        return "approved", f"all checks passed, review score {review['score']}", {}
    kills = [c for c in failed if c.responsible == "kill"]
    if kills:
        return "killed", "; ".join(f"{c.name}: {c.detail}" for c in kills), {}
    if qa_loops >= max_loops:
        return "killed", f"max QA loops ({max_loops}) reached; last: " + "; ".join(c.detail for c in failed), {}
    editor = [c for c in failed if c.responsible == "clip_editor"]
    group = editor or [c for c in failed if c.responsible == "packager"]
    feedback: dict = {}
    for c in group:
        feedback.update(c.fix)
    if not editor:
        feedback.setdefault("packaging", "; ".join(c.detail for c in group))
    target = "clip_editor" if editor else "packager"
    return ROUTE[target], f"rejected to {target}: " + "; ".join(f"{c.name}: {c.detail}" for c in group), feedback


class QAGateAgent(BaseAgent):
    name = "qa_gate"
    input_status = "packaged"
    kind = "clip"

    def process(self, job):
        p = job.payload
        words = all_words(json.loads(Path(p["transcript"]).read_text(encoding="utf-8")))
        checks = [check_format(p, self.rules)]
        if checks[0].ok:
            checks += [check_sync(p, self.rules), check_black(p, self.rules)]
        checks += [check_cutoff(p, words), *check_rules(p, self.rules, self.settings.assets_dir.parent, self.playbook),
                   check_duplicate(self.conn, job, self.rules)]
        try:
            review, engine = llm_review(self.llm, p, self.rules), "llm"
        except LLMUnavailable as exc:
            log("fallback", level="warning", what="qa_review", job_id=job.id, error=str(exc))
            review, engine = heuristic_review(p), "heuristic"
        to_status, reason, feedback = decide(checks, review, self.rules["qa"]["min_llm_score"], job.qa_loops,
                                             self.rules["qa"]["max_qa_loops"])
        verdict = {"approved": "approve", "killed": "kill"}.get(to_status, "reject")
        report = {"verdict": verdict, "reason": reason, "review_engine": engine, "review": review,
                  "checks": [asdict(c) for c in checks], "loop": job.qa_loops}
        Path(p["clip_path"]).with_name("qa.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        history = p.get("qa_history", []) + [{"verdict": verdict, "reason": reason}]
        return AgentResult(to_status, reason=reason,
                           payload={"qa": report, "qa_feedback": feedback, "qa_history": history},
                           qa_loops=job.qa_loops + (1 if verdict == "reject" else 0))


def main():  # pragma: no cover
    QAGateAgent().run()


if __name__ == "__main__":  # pragma: no cover
    main()
