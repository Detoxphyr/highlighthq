"""Agent 4: Packager. rendered -> packaged.

Writes per-platform title, caption and hashtags (YouTube Shorts, Instagram
Reels, TikTok, Facebook Reels) following playbook.yaml packaging limits. Uses
the LLM, falls back to templates. Limits are always enforced after generation."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from agents.base import AgentResult, BaseAgent
from core.llm import LLMUnavailable
from core.logging import log

STOPWORDS = set("""a an the and or but so to of in on at for with is are was were be been it this that you your i me my we
our they them he she his her its as by from not no do does did have has had just like really very can could would should
will about what when where who why how there here then than because if into out up down over more most some any all one
get got going know think thing things people dont don't im i'm it's that's youre you're""".split())


def keywords(text: str, n: int = 6) -> list[str]:
    toks = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in text.split()]
    toks = [t for t in toks if len(t) > 3 and t not in STOPWORDS and not t.isdigit()]
    return [w for w, _ in Counter(toks).most_common(n)]


def first_sentence(text: str) -> str:
    m = re.match(r"(.+?[.?!])(\s|$)", text.strip())
    return (m.group(1) if m else text.strip()).strip()


def truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:-")
    return cut + "…"


def normalize_tag(tag: str) -> str:
    t = re.sub(r"[^A-Za-z0-9_]", "", tag.lstrip("#"))
    return f"#{t.lower()}" if t else ""


def enforce(platform: str, pkg: dict, pcfg: dict, rules: dict) -> dict:
    cap = min(pcfg["hashtags"], rules["content"]["max_hashtags_any_platform"])
    tags, seen = [], set()
    for t in pkg.get("hashtags", []):
        t = normalize_tag(t)
        if t and t not in seen:
            tags.append(t)
            seen.add(t)
    title = truncate(pkg.get("title", "").strip(), pcfg["title_max"])
    if rules["content"].get("disallow_all_caps_titles") and title.isupper():
        title = title.capitalize()
    tag_str = " ".join(tags[:cap])
    body_limit = pcfg["caption_max"] - (len(tag_str) + 2 if tag_str else 0)
    caption = truncate(pkg.get("caption", "").strip(), body_limit)
    return {"platform": platform, "title": title, "caption": caption, "hashtags": tags[:cap],
            "full_caption": (caption + ("\n\n" + tag_str if tag_str else "")).strip()}


def template_package(text: str, hook_type: str, playbook: dict) -> dict[str, dict]:
    pk = playbook["packaging"]
    hook = first_sentence(text)
    kws = keywords(text)
    out = {}
    for platform, pcfg in pk["platforms"].items():
        tags = pcfg.get("extra_tags", []) + pk["base_hashtags"] + [f"#{k}" for k in kws]
        title = hook if hook_type in ("question", "bold_claim", "contrarian", "number") else (
            kws[0].capitalize() + ": " + hook if kws else hook)
        caption = f"“{truncate(hook, 140)}” {pk['cta']}"
        out[platform] = {"title": title, "caption": caption, "hashtags": tags}
    return out


PKG_SCHEMA = {
    "type": "object",
    "properties": {p: {"type": "object", "properties": {
        "title": {"type": "string"}, "caption": {"type": "string"},
        "hashtags": {"type": "array", "items": {"type": "string"}}},
        "required": ["title", "caption", "hashtags"], "additionalProperties": False}
        for p in ("youtube_shorts", "instagram_reels", "tiktok", "facebook_reels")},
    "required": ["youtube_shorts", "instagram_reels", "tiktok", "facebook_reels"],
    "additionalProperties": False,
}


def llm_package(llm, text: str, hook_type: str, playbook: dict, rules: dict, feedback: str | None) -> dict[str, dict]:
    pk = playbook["packaging"]
    limits = {p: {"title_max_chars": c["title_max"], "caption_max_chars": c["caption_max"], "hashtags": c["hashtags"],
                  "must_include_tags": c.get("extra_tags", [])} for p, c in pk["platforms"].items()}
    system = (
        "You write packaging for short vertical video clips. For each platform write a title, a caption and "
        f"hashtags. Caption style: {pk['caption_style']} (open with the clip's hook, end with this CTA: \"{pk['cta']}\"). "
        "Titles must be accurate to what is said in the clip: no clickbait the clip does not deliver. A title_max of 0 "
        "means the platform has no title field: return an empty string. Never use these terms: "
        + ", ".join(rules["content"]["banned_terms"]) + "."
    )
    prompt = (f"Hook type: {hook_type}\nPlatform limits: {json.dumps(limits)}\n"
              f"Base hashtags you may use: {pk['base_hashtags']}\n\nClip transcript:\n{text}")
    if feedback:
        prompt += f"\n\nA reviewer rejected the previous packaging: {feedback}. Fix that."
    return llm.complete_json(system, prompt, PKG_SCHEMA, max_tokens=4000, effort="low")


class PackagerAgent(BaseAgent):
    name = "packager"
    input_status = "rendered"
    kind = "clip"

    def process(self, job):
        p = job.payload
        text, hook_type = p["text"], job.decisions.get("hook_type", "story")
        feedback = (p.get("qa_feedback") or {}).get("packaging")
        try:
            raw, engine = llm_package(self.llm, text, hook_type, self.playbook, self.rules, feedback), "llm"
        except LLMUnavailable as exc:
            log("fallback", level="warning", what="packaging", job_id=job.id, error=str(exc))
            raw, engine = template_package(text, hook_type, self.playbook), "template"
        pcfgs = self.playbook["packaging"]["platforms"]
        packages = {plat: enforce(plat, raw.get(plat, {}), pcfgs[plat], self.rules) for plat in pcfgs}
        out = Path(p["clip_path"]).with_name("packages.json")
        out.write_text(json.dumps(packages, indent=2, ensure_ascii=False))
        return AgentResult("packaged", reason=f"{len(packages)} platforms via {engine}",
                           payload={"packages": packages, "packages_path": str(out), "packaging_engine": engine},
                           decisions={"caption_style": self.playbook["packaging"]["caption_style"],
                                      "packaging_engine": engine})


def main():  # pragma: no cover
    PackagerAgent().run()


if __name__ == "__main__":  # pragma: no cover
    main()
