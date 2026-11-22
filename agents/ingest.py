"""Agent 1: Ingest. scouted -> ingested.

Downloads the source (yt-dlp for URLs, copy for local files), transcribes with
faster-whisper word timestamps, detects scene cuts, and measures loudness.

Fallbacks on persistent tool failure:
- download: yt-dlp -> direct HTTP fetch for direct media URLs
- transcription: configured whisper model -> smaller whisper model -> sidecar
  transcript next to the source (<name>.transcript.json or <name>.srt)
- scene detection: ffmpeg scdet -> fixed-interval pseudo cuts
"""
from __future__ import annotations

import json
import re
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from agents.base import AgentResult, BaseAgent
from core.logging import log
from core.retry import PersistentError, TransientError, retry, run_cmd, with_fallback


def probe(path: Path) -> dict:
    out = run_cmd(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    info = json.loads(out.stdout)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    fps = 30.0
    if v and v.get("avg_frame_rate", "0/0") != "0/0":
        n, d = v["avg_frame_rate"].split("/")
        fps = float(n) / float(d) if float(d) else 30.0
    return {"duration": float(info["format"]["duration"]), "width": v and v["width"], "height": v and v["height"],
            "fps": fps, "has_audio": a is not None}


@retry(tries=3, base_delay=2.0)
def download_ytdlp(url: str, dest_dir: Path) -> Path:
    try:
        run_cmd(["yt-dlp", "--no-playlist", "-f", "bv*[height<=1080]+ba/b[height<=1080]/b", "--merge-output-format", "mp4",
                 "-o", str(dest_dir / "source.%(ext)s"), url], timeout=1800)
    except PersistentError as exc:
        msg = str(exc).lower()
        if any(k in msg for k in ("timed out", "connection", "429", "temporarily", "reset")):
            raise TransientError(str(exc)) from exc
        raise
    files = sorted(dest_dir.glob("source.*"))
    if not files:
        raise PersistentError("yt-dlp produced no file")
    return files[0]


@retry(tries=3, base_delay=2.0)
def download_direct(url: str, dest_dir: Path) -> Path:
    dest = dest_dir / ("source" + (Path(url.split("?")[0]).suffix or ".mp4"))
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except OSError as exc:
        raise TransientError(str(exc)) from exc
    return dest


def fetch_source(src: str, dest_dir: Path) -> tuple[Path, str]:
    if re.match(r"^https?://", src):
        return with_fallback(lambda: download_ytdlp(src, dest_dir), lambda: download_direct(src, dest_dir),
                             "download", src=src)
    path = Path(src).expanduser().resolve()
    if not path.exists():
        raise PersistentError(f"source file not found: {path}")
    dest = dest_dir / f"source{path.suffix}"
    if path != dest:
        shutil.copy2(path, dest)
    return dest, "local"


# ---------- transcription ----------

def transcribe_whisper(path: Path, model_name: str) -> dict:
    from faster_whisper import WhisperModel  # imported lazily; heavy
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(path), word_timestamps=True, vad_filter=True, beam_size=5)
    out = []
    for seg in segments:
        words = [{"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3), "p": round(w.probability, 3)}
                 for w in (seg.words or []) if w.word.strip()]
        out.append({"start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text.strip(), "words": words})
    return {"engine": f"faster-whisper:{model_name}", "language": info.language, "segments": out}


def _srt_time(t: str) -> float:
    h, m, rest = t.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def words_from_text(text: str, start: float, end: float) -> list[dict]:
    """Spread words across [start, end] proportionally to their length."""
    toks = text.split()
    if not toks:
        return []
    weights = np.array([len(t) + 1 for t in toks], dtype=float)
    edges = start + (end - start) * np.concatenate([[0], np.cumsum(weights) / weights.sum()])
    return [{"w": t, "s": round(edges[i], 3), "e": round(edges[i + 1], 3), "p": None} for i, t in enumerate(toks)]


def transcribe_sidecar(source_original: str) -> dict:
    base = Path(source_original).expanduser()
    j = base.with_suffix(".transcript.json")
    if j.exists():
        data = json.loads(j.read_text())
        data.setdefault("engine", "sidecar:json")
        return data
    srt = base.with_suffix(".srt")
    if srt.exists():
        segs = []
        for block in re.split(r"\n\s*\n", srt.read_text().strip()):
            lines = block.strip().splitlines()
            if len(lines) < 3 or "-->" not in lines[1]:
                continue
            a, b = [_srt_time(x.strip()) for x in lines[1].split("-->")]
            text = " ".join(lines[2:]).strip()
            segs.append({"start": a, "end": b, "text": text, "words": words_from_text(text, a, b)})
        return {"engine": "sidecar:srt", "language": "en", "segments": segs}
    raise PersistentError("no sidecar transcript found")


def transcribe(path: Path, original: str, settings) -> dict:
    def primary():
        return transcribe_whisper(path, settings.whisper_model)

    def secondary():
        try:
            return transcribe_whisper(path, settings.whisper_fallback_model)
        except Exception as exc:  # noqa: BLE001
            log("fallback", level="warning", what="whisper_small", error=str(exc))
            return transcribe_sidecar(original)

    result, used = with_fallback(primary, secondary, "transcription")
    result["path_used"] = used
    if not any(s["words"] for s in result["segments"]):
        raise PersistentError("transcript has no words")
    return result


# ---------- scenes + loudness ----------

def detect_scenes_ffmpeg(path: Path, threshold: float = 5.0) -> list[float]:
    proc = run_cmd(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-an",
                    "-vf", f"scale=320:-2,scdet=threshold={threshold}", "-f", "null", "-"], timeout=1800)
    text = proc.stderr.decode(errors="replace")
    return sorted({round(float(m), 3) for m in re.findall(r"lavfi\.scd\.time:\s*([0-9.]+)", text)})


def loudness_curve(path: Path, hop_s: float = 0.5) -> list[float]:
    proc = run_cmd(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                   timeout=1800)
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    hop = int(16000 * hop_s)
    n = len(pcm) // hop
    if n == 0:
        return []
    frames = pcm[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    return [round(float(x), 2) for x in db]


class IngestAgent(BaseAgent):
    name = "ingest"
    input_status = "scouted"
    kind = "source"

    def process(self, job):
        out_dir = self.job_dir(job)
        src = job.payload["input"]
        media, how = fetch_source(src, out_dir)
        info = probe(media)
        if not info["has_audio"]:
            raise PersistentError("source has no audio track")
        transcript = transcribe(media, src, self.settings)
        cuts, scene_path = with_fallback(
            lambda: detect_scenes_ffmpeg(media),
            lambda: [float(t) for t in np.arange(10, info["duration"], 10)],
            "scene_detection", job_id=job.id)
        scenes = {"cuts": cuts, "path_used": scene_path, "loudness_db": loudness_curve(media), "hop_s": 0.5, **info}
        (out_dir / "transcript.json").write_text(json.dumps(transcript, indent=1))
        (out_dir / "scenes.json").write_text(json.dumps(scenes))
        self.conn.execute("UPDATE sources SET duration_s=? WHERE id=?", (info["duration"], job.source_id))
        n_words = sum(len(s["words"]) for s in transcript["segments"])
        return AgentResult(
            "ingested",
            reason=f"{how} download, {transcript['engine']} ({n_words} words), {len(cuts)} scene cuts",
            payload={"media": str(media), "transcript": str(out_dir / "transcript.json"),
                     "scenes": str(out_dir / "scenes.json"), "duration": info["duration"],
                     "transcript_engine": transcript["engine"]},
        )


def main():  # pragma: no cover
    IngestAgent().run()


if __name__ == "__main__":  # pragma: no cover
    main()
