"""Agent 3: Clip Editor. candidates -> rendered.

Renders a 9:16 clip: face-tracked reframe (OpenCV Haar cascade, falling back to
motion/saliency tracking, then center), burned-in word-by-word animated
subtitles (ASS via libass), watermark overlay from assets/watermark.png.
Writes the mp4 plus a checkpoint frame. If the tracked render path fails, it
falls back to a static center-crop ffmpeg render."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from agents.base import AgentResult, BaseAgent
from agents.moment_finder import all_words, split_sentences
from core.logging import log
from core.retry import PersistentError, run_cmd, with_fallback

LEAD_IN_S, TAIL_S = 0.15, 0.4


# ---------- reframe track ----------

_FACE = None


def face_detector():
    global _FACE
    if _FACE is None:
        _FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _FACE


def crop_geometry(src_w: int, src_h: int) -> tuple[int, int]:
    """Largest 9:16 box that fits the source."""
    if src_w / src_h > 9 / 16:
        return int(round(src_h * 9 / 16)) // 2 * 2, src_h
    return src_w, int(round(src_w * 16 / 9)) // 2 * 2


def compute_track(media: str, start: float, end: float, sample_fps: float = 4.0, smoothing: float = 0.15) -> dict:
    """Return {'times': [...], 'cx': [...], 'engine': face|saliency|center} with cx = crop center x in px."""
    cap = cv2.VideoCapture(media)
    if not cap.isOpened():
        raise PersistentError(f"cannot open {media}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    times, raw, sources = [], [], []
    prev_small = None
    det = face_detector()
    t = start
    while t <= end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        scale = 480 / max(w, 1)
        small = cv2.resize(frame, (480, int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        cx = None
        if len(faces):
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            cx, src = (x + fw / 2) / scale, "face"
        else:
            # saliency: motion energy if available, else edge density
            if prev_small is not None:
                energy = cv2.absdiff(gray, prev_small).astype(np.float32)
            else:
                energy = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
            col = energy.sum(axis=0)
            if col.sum() > 1e-3 * energy.size:
                cx, src = float((col * np.arange(col.size)).sum() / col.sum()) / scale, "saliency"
            else:
                cx, src = w / 2, "center"
        prev_small = gray
        times.append(t)
        raw.append(cx)
        sources.append(src)
        t += 1 / sample_fps
    cap.release()
    if not raw:
        raise PersistentError("no frames decoded for reframe track")
    crop_w, _ = crop_geometry(w, h)
    sm, cur = [], raw[0]
    for v in raw:
        cur = cur + smoothing * (v - cur)
        sm.append(float(np.clip(cur, crop_w / 2, w - crop_w / 2)))
    engine = max(set(sources), key=sources.count)
    return {"times": times, "cx": sm, "engine": engine, "face_ratio": sources.count("face") / len(sources),
            "src_w": w, "src_h": h}


# ---------- subtitles ----------

def _ass_time(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _esc(text: str) -> str:
    return text.replace("{", "(").replace("}", ")").replace("\\", "/")


def build_ass(words: list[dict], clip_start: float, clip_end: float, cfg: dict, width: int, height: int) -> str:
    """Word-pop karaoke: chunks of N words; the active word is highlighted and pops."""
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Pop,{cfg['font']},{cfg['font_size']},{cfg['primary_color']},&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{cfg['outline']},2,2,60,60,{cfg['margin_v']},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    clip_words = [w for w in words if w["s"] >= clip_start - 0.05 and w["e"] <= clip_end + 0.05]
    n = max(1, int(cfg.get("words_per_chunk", 3)))
    lines = []
    for ci in range(0, len(clip_words), n):
        chunk = clip_words[ci:ci + n]
        chunk_end = chunk[-1]["e"]
        for k, w in enumerate(chunk):
            s = w["s"] - clip_start
            e = (chunk[k + 1]["s"] if k + 1 < len(chunk) else chunk_end) - clip_start
            if cfg.get("style") == "plain":
                text = " ".join(_esc(x["w"]) for x in chunk)
            else:
                parts = []
                for j, x in enumerate(chunk):
                    token = _esc(x["w"].upper())
                    if j == k:
                        parts.append("{\\c" + cfg["highlight_color"] + "\\fscx112\\fscy112\\t(0,90,\\fscx100\\fscy100)}"
                                     + token + "{\\r}")
                    else:
                        parts.append(token)
                text = " ".join(parts)
            lines.append(f"Dialogue: 0,{_ass_time(s)},{_ass_time(max(e, s + 0.05))},Pop,,0,0,0,,{text}")
    return header + "\n".join(lines) + "\n"


def snap_to_sentences(words: list[dict], start: float, end: float) -> tuple[float, float]:
    sents = split_sentences(words)
    starts = [s.start for s in sents if s.start <= start + 0.5] or [start]
    ends = [s.end for s in sents if s.end >= end - 0.5] or [end]
    return max(starts), min(ends)


# ---------- render ----------

def _filter_graph(ass_path: Path, wm_cfg: dict, out_w: int) -> str:
    wm_w = int(out_w * wm_cfg.get("scale", 0.22)) // 2 * 2
    ass = str(ass_path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    return (f"[0:v]ass='{ass}'[v1];"
            f"[2:v]scale={wm_w}:-1,format=rgba,colorchannelmixer=aa={wm_cfg.get('opacity', 0.8)}[wm];"
            f"[v1][wm]overlay=W-w-40:60,format=yuv420p[v]")


def render_tracked(media: str, start: float, end: float, track: dict, ass_path: Path, wm_path: Path,
                   out_path: Path, rcfg: dict, wm_cfg: dict, fps: float) -> None:
    W, H = rcfg["width"], rcfg["height"]
    cap = cv2.VideoCapture(media)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    src_w, src_h = track["src_w"], track["src_h"]
    crop_w, crop_h = crop_geometry(src_w, src_h)
    n_frames = int(round((end - start) * fps))
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps:.5f}", "-i", "-",
           "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", media,
           "-i", str(wm_path),
           "-filter_complex", _filter_graph(ass_path, wm_cfg, W),
           "-map", "[v]", "-map", "1:a:0", "-c:v", "libx264", "-preset", rcfg["preset"], "-crf", str(rcfg["crf"]),
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    times, cxs = np.array(track["times"]), np.array(track["cx"])
    last = None
    try:
        for i in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                frame = last if last is not None else np.zeros((src_h, src_w, 3), np.uint8)
            last = frame
            t = start + i / fps
            cx = float(np.interp(t, times, cxs)) if src_w / src_h > 9 / 16 else src_w / 2
            x0 = int(np.clip(cx - crop_w / 2, 0, src_w - crop_w))
            y0 = (src_h - crop_h) // 2 if crop_h < src_h else 0
            crop = frame[y0:y0 + crop_h, x0:x0 + crop_w]
            proc.stdin.write(cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR).tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    finally:
        cap.release()
    rc = proc.wait(timeout=900)
    if rc != 0:
        raise PersistentError(f"ffmpeg render failed: {proc.stderr.read().decode(errors='replace')[-1500:]}")


def render_static(media: str, start: float, end: float, ass_path: Path, wm_path: Path, out_path: Path,
                  rcfg: dict, wm_cfg: dict) -> None:
    """Fallback: pure ffmpeg center crop."""
    W, H = rcfg["width"], rcfg["height"]
    graph = (f"[0:v]crop='min(iw,ih*9/16)':'min(ih,iw*16/9)',scale={W}:{H},setsar=1[c];"
             + _filter_graph(ass_path, wm_cfg, W).replace("[0:v]", "[c]"))
    run_cmd(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", media,
             "-i", media, "-i", str(wm_path), "-filter_complex", graph, "-map", "[v]", "-map", "0:a:0",
             "-c:v", "libx264", "-preset", rcfg["preset"], "-crf", str(rcfg["crf"]), "-c:a", "aac", "-b:a", "160k",
             "-movflags", "+faststart", str(out_path)], timeout=1800)


class ClipEditorAgent(BaseAgent):
    name = "clip_editor"
    input_status = "candidates"
    kind = "clip"

    def process(self, job):
        p = job.payload
        transcript = json.loads(Path(p["transcript"]).read_text(encoding="utf-8"))
        scenes = json.loads(Path(p["scenes"]).read_text(encoding="utf-8"))
        words = all_words(transcript)
        start, end = p["start"], p["end"]
        feedback = p.get("qa_feedback") or {}
        if feedback.get("snap_boundaries"):
            start, end = snap_to_sentences(words, start, end)
        r_start = max(0.0, start - LEAD_IN_S)
        r_end = min(scenes["duration"], end + TAIL_S)
        subs_shift = float(feedback.get("subtitle_shift_s", 0.0))

        out_dir = self.job_dir(job) / f"clip_{job.id}"
        out_dir.mkdir(exist_ok=True)
        rcfg, scfg, wcfg = self.playbook["render"], self.playbook["subtitles"], self.playbook["watermark"]
        wm_path = self.settings.assets_dir.parent / wcfg["path"]
        if not wm_path.exists():
            raise PersistentError(f"watermark missing: {wm_path}")
        ass_path = out_dir / "subs.ass"
        ass_path.write_text(build_ass(words, r_start - subs_shift, r_end - subs_shift, scfg, rcfg["width"], rcfg["height"]), encoding="utf-8")
        out_path = out_dir / "clip.mp4"
        fps = min(float(scenes.get("fps") or rcfg["fps"]), float(rcfg["fps"]))

        track = {"engine": "center"}

        def tracked():
            nonlocal track
            track = compute_track(p["media"], r_start, r_end, smoothing=rcfg["reframe_smoothing"])
            render_tracked(p["media"], r_start, r_end, track, ass_path, wm_path, out_path, rcfg, wcfg, fps)
            return "tracked"

        def static():
            render_static(p["media"], r_start, r_end, ass_path, wm_path, out_path, rcfg, wcfg)
            return "static_center"

        mode, _ = with_fallback(tracked, static, "render", job_id=job.id)
        ckpt = out_dir / "checkpoint.jpg"
        run_cmd(["ffmpeg", "-y", "-v", "error", "-ss", f"{(r_end - r_start) / 2:.2f}", "-i", str(out_path),
                 "-frames:v", "1", "-q:v", "3", str(ckpt)])
        log("clip_rendered", job_id=job.id, mode=mode, reframe=track.get("engine"), path=str(out_path))
        return AgentResult(
            "rendered", reason=f"{mode} render, reframe={track.get('engine')}",
            payload={"clip_path": str(out_path), "checkpoint": str(ckpt), "subtitles": str(ass_path),
                     "render_start": round(r_start, 3), "render_end": round(r_end, 3), "start": round(start, 3),
                     "end": round(end, 3), "subtitle_shift_s": subs_shift, "render_mode": mode,
                     "reframe_engine": track.get("engine"), "face_ratio": round(track.get("face_ratio", 0.0), 2)},
            decisions={"length_s": round(r_end - r_start, 1), "reframe": track.get("engine"),
                       "subtitle_style": scfg["style"]})


def main():  # pragma: no cover
    ClipEditorAgent().run()


if __name__ == "__main__":  # pragma: no cover
    main()
