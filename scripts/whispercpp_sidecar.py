"""Transcribe a video with a local whisper.cpp build and write the Ingest agent's sidecar
transcript (<video stem>.transcript.json next to the video).

For machines where faster-whisper cannot load (e.g. Windows Smart App Control blocks the
ctranslate2/PyAV DLLs). Uses real per-word timestamps from whisper.cpp's full JSON output.

usage: python scripts/whispercpp_sidecar.py <video.mp4> [more.mp4 ...]
env:   WHISPER_CPP_DIR (default %LOCALAPPDATA%/Programs/whisper.cpp)
       WHISPER_CPP_MODEL (default ggml-large-v3-turbo-q5_0.bin), WHISPER_CPP_LANG (default en)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WH = Path(os.environ.get("WHISPER_CPP_DIR", Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "whisper.cpp"))
MODEL = os.environ.get("WHISPER_CPP_MODEL", "ggml-large-v3-turbo-q5_0.bin")
LANG = os.environ.get("WHISPER_CPP_LANG", "en")
# 15 s chunks: on loud game audio Whisper can call a whole 30 s window "no speech".
CHUNK_S = float(os.environ.get("WHISPER_CPP_CHUNK_S", "15"))
GAP_S, MAX_SEG_S = 1.2, 12.0


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


def chunk_words(wav: Path, base: Path, offset: float) -> list[dict]:
    subprocess.run([str(WH / "Release" / "whisper-cli.exe"), "-m", str(WH / "models" / MODEL), "-f", str(wav),
                    "-l", LANG, "-ml", "1", "-sow", "-mc", "0", "-bs", "5", "-oj", "-ojf", "-of", str(base)],
                   check=True, capture_output=True)
    data = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    words = []
    for seg in data["transcription"]:
        text = seg["text"].strip()
        if not text or text.startswith(("[_", "<|", "[")):
            continue
        probs = [t["p"] for t in seg.get("tokens", []) if not t["text"].startswith(("[_", "<|"))]
        s = offset + seg["offsets"]["from"] / 1000
        e = offset + seg["offsets"]["to"] / 1000
        words.append({"w": text, "s": round(s, 3), "e": round(max(e, s + 0.02), 3),
                      "p": round(sum(probs) / len(probs), 3) if probs else None})
    return words


def group_segments(words: list[dict]) -> list[dict]:
    segs, cur = [], []
    for w in words:
        if cur and (w["s"] - cur[-1]["e"] > GAP_S or cur[-1]["w"].endswith((".", "?", "!"))
                    or w["e"] - cur[0]["s"] > MAX_SEG_S):
            segs.append(cur)
            cur = []
        cur.append(w)
    if cur:
        segs.append(cur)
    return [{"start": g[0]["s"], "end": g[-1]["e"], "text": " ".join(x["w"] for x in g), "words": g} for g in segs]


def transcribe(video: Path) -> Path:
    out = video.with_suffix(".transcript.json")
    dur = duration(video)
    words: list[dict] = []
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        wav, base = Path(td) / "c.wav", Path(td) / "c"
        k = 0.0
        while k < dur - 0.5:  # whisper.cpp exits with an error on a near-empty final chunk
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{k:.2f}", "-t", f"{CHUNK_S:.2f}",
                            "-i", str(video), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)], check=True)
            words += chunk_words(wav, base, k)
            k += CHUNK_S
            print(f"{video.name}: {min(k, dur):7.0f}/{dur:.0f}s  {len(words)} words  {time.time() - t0:5.0f}s",
                  flush=True)
    result = {"engine": f"whisper.cpp:{MODEL.removesuffix('.bin')}", "language": LANG,
              "segments": group_segments(words), "wall_s": round(time.time() - t0, 1)}
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print("wrote", transcribe(Path(arg)), flush=True)
