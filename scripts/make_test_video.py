"""Generate a synthetic, fully original long-form test video for HighlightHQ.

- Narration: espeak-ng text-to-speech of an original script (written for this repo).
- Picture: 1920x1080 animated "speaker" whose mouth follows the audio and whose
  position changes per scene, with hard scene cuts at paragraph boundaries.
- Sidecar transcript (<name>.transcript.json): exact sentence timings with
  per-word timings spread by word length. Ingest uses it only when
  faster-whisper cannot load a model.

All content is generated here, so the output is free to use (CC0).
Usage: python scripts/make_test_video.py [out.mp4]
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np

SCRIPT = [
    ["Welcome back to the show.", "Today we are talking about how people actually learn new skills.",
     "I have spent ten years teaching, and I want to share what I got wrong."],
    ["Here's the truth nobody tells you about practice.", "More hours do not make you better.",
     "Better feedback makes you better.", "If you practice the wrong thing for a thousand hours, you just get very good at the wrong thing."],
    ["So what does good feedback look like?", "It is fast, it is specific, and it hurts a little.",
     "When I learned to cook, my grandmother would taste one spoon and tell me exactly what was missing.",
     "One spoon. That was the whole lesson.", "I improved more in one month with her than in a year of cooking videos."],
    ["Let me give you some numbers.", "In one study, students who tested themselves remembered about fifty percent more a week later.",
     "Re-reading the same notes felt productive, but it barely moved the needle.",
     "Feeling productive and being productive are not the same thing."],
    ["Why do we keep choosing the comfortable method?", "Because struggle feels like failure.",
     "But struggle is the signal that your brain is building something new.",
     "If it feels easy, you are probably reviewing, not learning."],
    ["Now, a quick story.", "A friend of mine wanted to learn the guitar in her forties.",
     "Everyone told her she was too old.", "She practiced fifteen minutes a day, every day, and recorded herself every Sunday.",
     "Two years later she played at her daughter's wedding.", "The recording was the secret. It was her feedback loop."],
    ["What if you only have ten minutes a day?", "Ten minutes is plenty if you spend it on the hardest part.",
     "Skip the warm up you already know.", "Go straight to the thing you keep avoiding."],
    ["The biggest mistake I see is waiting to feel ready.", "Nobody feels ready.",
     "Readiness is something you earn by starting badly.", "Start badly, get feedback, and start again tomorrow."],
    ["To wrap up, here are the three rules.", "First, shorten the gap between doing and knowing how you did.",
     "Second, practice the part that hurts.", "Third, record yourself, because memory flatters you.",
     "Thanks for listening, and I will see you next week."],
]

W, H, FPS = 1920, 1080, 30
SR = 22050
BGS = [(60, 40, 30), (40, 60, 90), (70, 30, 70), (30, 70, 50), (90, 60, 30), (50, 50, 80), (30, 50, 70), (80, 40, 40), (40, 40, 40)]
POS = [0.5, 0.33, 0.67, 0.4, 0.62, 0.3, 0.55, 0.7, 0.45]


def tts(text: str, path: Path) -> np.ndarray:
    subprocess.run(["espeak-ng", "-v", "en-us", "-s", "158", "-p", "45", "-w", str(path), text], check=True)
    with wave.open(str(path)) as wf:
        sr = wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    if sr != SR:
        idx = np.linspace(0, len(pcm) - 1, int(len(pcm) * SR / sr)).astype(int)
        pcm = pcm[idx]
    # trim leading/trailing near-silence so sentence timings are tight
    nz = np.where(np.abs(pcm) > 300)[0]
    return pcm[nz[0]: nz[-1] + 1] if nz.size else pcm


def words_for(text: str, start: float, end: float) -> list[dict]:
    toks = text.split()
    weights = np.array([len(t) + 1 for t in toks], dtype=float)
    edges = start + (end - start) * np.concatenate([[0], np.cumsum(weights) / weights.sum()])
    return [{"w": t, "s": round(float(edges[i]), 3), "e": round(float(edges[i + 1]), 3), "p": None} for i, t in enumerate(toks)]


def main(out: Path, paragraphs: int | None = None) -> Path:
    script = SCRIPT[:paragraphs] if paragraphs else SCRIPT
    out.parent.mkdir(parents=True, exist_ok=True)
    audio, segments, scene_starts = [np.zeros(int(SR * 0.6), np.int16)], [], []
    t = 0.6
    with tempfile.TemporaryDirectory() as td:
        for pi, para in enumerate(script):
            scene_starts.append(max(0.0, t - 0.4))
            for si, sent in enumerate(para):
                pcm = tts(sent, Path(td) / f"{pi}_{si}.wav")
                dur = len(pcm) / SR
                segments.append({"start": round(t, 3), "end": round(t + dur, 3), "text": sent, "words": words_for(sent, t, t + dur)})
                audio.append(pcm)
                gap = 0.35 if si < len(para) - 1 else 0.9
                audio.append(np.zeros(int(SR * gap), np.int16))
                t += dur + gap
    pcm = np.concatenate(audio)
    total = len(pcm) / SR
    wav_path = out.with_suffix(".wav")
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())

    n_frames = int(total * FPS)
    hop = SR // FPS
    env = np.array([np.sqrt(np.mean((pcm[i * hop:(i + 1) * hop].astype(np.float32) / 32768) ** 2) + 1e-9)
                    for i in range(n_frames)])
    env = np.clip(env / (env.max() + 1e-9) * 1.6, 0, 1)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-i", str(wav_path), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k", "-shortest", str(out)], stdin=subprocess.PIPE)
    scene_idx = 0
    for f in range(n_frames):
        ts = f / FPS
        while scene_idx + 1 < len(scene_starts) and ts >= scene_starts[scene_idx + 1]:
            scene_idx += 1
        frame = np.zeros((H, W, 3), np.uint8)
        frame[:] = BGS[scene_idx % len(BGS)]
        cv2.rectangle(frame, (0, int(H * 0.78)), (W, H), tuple(int(c * 0.6) for c in BGS[scene_idx % len(BGS)]), -1)
        cx = int(W * POS[scene_idx % len(POS)] + 40 * np.sin(ts * 0.7))
        cy = int(H * 0.45 + 8 * np.sin(ts * 1.3))
        cv2.ellipse(frame, (cx, cy + 330), (230, 160), 0, 180, 360, (90, 110, 160), -1)  # shoulders
        cv2.circle(frame, (cx, cy), 150, (150, 185, 225), -1)                            # head
        cv2.circle(frame, (cx - 52, cy - 30), 16, (40, 40, 40), -1)
        cv2.circle(frame, (cx + 52, cy - 30), 16, (40, 40, 40), -1)
        mouth = int(6 + 46 * env[f])
        cv2.ellipse(frame, (cx, cy + 60), (55, mouth), 0, 0, 360, (60, 40, 120), -1)
        cv2.putText(frame, "The Learning Hour (synthetic test video)", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (230, 230, 230), 2, cv2.LINE_AA)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    wav_path.unlink()
    sidecar = out.with_suffix(".transcript.json")
    sidecar.write_text(json.dumps({"engine": "sidecar:json (script timings)", "language": "en", "segments": segments}, indent=1))
    print(f"{out} ({total:.1f}s, {len(segments)} sentences, {len(script)} scenes)\n{sidecar}")
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/test/learning_hour.mp4").resolve())
