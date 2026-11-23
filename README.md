# HighlightHQ

A multi-agent clipping pipeline that turns long-form videos into short vertical clips. Agents never call each
other: each one polls SQLite for job cards in its input status, does its work, and moves the card on.

```
scouted -> ingested -> candidates -> rendered -> packaged -> approved -> posted -> submitted -> tracked
                                                     (+ failed, dead_letter, killed)
```

**Phase 1 (this release):** agents 1-5 (Ingest, Moment Finder, Clip Editor, Packager, QA Gate) working end to end.
Publisher, Submitter, Tracker (Phase 2) and Learner, Orchestrator, real platform integrations (Phase 3) come next.

## Setup

Requires Python 3.11, ffmpeg built with libass (for subtitles), and optionally `espeak-ng` (only for the test video
generator).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config/.env.example .env        # add ANTHROPIC_API_KEY; never commit .env
python scripts/make_watermark.py   # or drop your own assets/watermark.png
```

The first transcription downloads the faster-whisper model (`HQ_WHISPER_MODEL`, default `base.en`) from Hugging Face.

### Windows notes

- If Windows Smart App Control blocks faster-whisper's DLLs (`ctranslate2`, PyAV: "An Application Control policy has
  blocked this file"), transcribe with a local whisper.cpp build instead:
  `python scripts/whispercpp_sidecar.py data/vods/<id>.mp4`. It writes `<id>.transcript.json` next to the video with
  real word timestamps, and Ingest picks it up through its sidecar fallback.
- Shells started by the Claude desktop app set `ANTHROPIC_BASE_URL`; clear it (`Remove-Item Env:ANTHROPIC_BASE_URL`)
  so pipeline calls go to api.anthropic.com.
- The playbook is versioned in the DB: an existing DB keeps using its stored version. Start a fresh DB (or bump
  `version` and clear the active row) after editing `playbook.yaml`.

## Run

```bash
python run.py init-db
python run.py add "https://www.youtube.com/watch?v=..."   # or a local file path
python run.py add data/vods/x.mp4 --peaks peaks.json --source-date 2026-11-20
#   --peaks: kai_scan.py output (or a list of seconds); the Moment Finder then only builds windows within
#   playbook clip.peak_radius_s of a peak. A local <video>.peaks.json is picked up automatically.
python run.py run-pipeline        # drains agents 1-5 until there is no work left, then prints the job table
python run.py status              # job table at any time
python run.py digest              # writes alerts/digest-YYYY-MM-DD.md
```

Each agent can also run as its own long-lived worker (one process per agent, any number in parallel):

```bash
python run.py run-agent ingest
python run.py run-agent moment_finder
python run.py run-agent clip_editor
python run.py run-agent packager
python run.py run-agent qa_gate          # add --once to drain and exit
```

### Test video

```bash
python scripts/make_test_video.py data/test/learning_hour.mp4
python run.py add data/test/learning_hour.mp4 && python run.py run-pipeline
```

The generator makes an original 2.3-minute talk (espeak-ng narration, animated speaker, hard scene cuts) plus a
`.transcript.json` sidecar with sentence timings. Everything in it is generated, so it is free to use.

### Tests

```bash
python -m pytest -q          # 44 tests; the end-to-end test renders real clips and takes ~3-4 minutes
```

## Layout

```
agents/   base.py (worker loop) · ingest.py · moment_finder.py · clip_editor.py · packager.py · qa_gate.py
core/     db.py (schema) · jobs.py (state machine) · llm.py · retry.py · breaker.py · alerts.py · logging.py · config.py
config/   playbook.yaml (style/strategy, versioned) · rules.yaml (hard rules, human-owned) · .env.example
assets/   watermark.png
data/     highlighthq.db · media/source_<id>/ (source, transcript.json, scenes.json, clip_<id>/) · logs/highlighthq.jsonl
alerts/   alert-*.json (escalations only) · digest-YYYY-MM-DD.md
scripts/  make_watermark.py · make_test_video.py
tests/    test_core.py · test_agents_unit.py · test_e2e.py
```

## Agents

| # | Agent | In -> out | Primary path | Fallback on persistent failure |
|---|---|---|---|---|
| 1 | Ingest | scouted -> ingested | yt-dlp, faster-whisper word timestamps, ffmpeg `scdet` scene cuts, loudness curve | direct HTTP download; smaller whisper model, then a sidecar `.transcript.json`/`.srt`; fixed-interval cuts |
| 2 | Moment Finder | ingested -> candidates (+ one clip job per moment) | sentence-aligned 20-60s windows scored by `claude-sonnet-5-5` on hook (first 3s), self-contained arc, emotional peak, quotability, weighted per playbook | deterministic heuristic scorer (hook patterns, dangling openers, loudness peaks, quotable lines) |
| 3 | Clip Editor | candidates -> rendered | 1080x1920 reframe following the persistent face track (OpenCV; outliers rejected, zero-lag smoothing, whole face kept in the crop), real-time frame mapping for any source fps, word-pop ASS subtitles, optional watermark (`watermark.enabled`; never while `campaign.no_logos`), checkpoint frame | motion/saliency tracking, then a static center-crop ffmpeg render |
| 4 | Packager | rendered -> packaged | per-platform title/caption/hashtags for the playbook's platforms (YouTube Shorts, Instagram Reels, TikTok, X) by the LLM, naming `packaging.creator_name` with `required_hashtags` first | template packaging; platform limits are enforced on either path |
| 5 | QA Gate | packaged -> approved / back / killed | format, audio-caption sync (captions vs detected silence), cut-off sentences, black frames, rules.yaml compliance, duplicates, plus an independent LLM review | heuristic review score |

QA rejections go back to the responsible agent with a machine-readable fix: editor problems return the card to
`candidates` (`snap_boundaries`, `subtitle_shift_s`), packaging problems to `rendered` (reviewer notes). After
`rules.qa.max_qa_loops` (2) rejections the clip is killed. Duplicates, banned spoken content and weak moments are
killed immediately.

## Error handling

- **Retries:** transient errors (timeouts, rate limits, 5xx, network) retry 3 times with exponential backoff and
  jitter (`core/retry.py`). Persistent errors are not retried.
- **Fallbacks:** every external tool has a fallback path (table above); each use is logged as a `fallback` event.
- **Failures:** an agent exception moves the card to `failed`, then back to the agent's input status; on the third
  failure it goes to `dead_letter` with the error, payload and the job's full log trail.
- **Circuit breakers:** persisted per key in the `breakers` table (`llm:anthropic` today; `platform:<name>:<account>`
  for the Publisher in Phase 2). Tripping one escalates an alert.
- **Structured logging:** JSON lines in `data/logs/highlighthq.jsonl`; every transition also lands in `transitions`.
- **Escalation:** only account strikes, payout anomalies, tripped breakers, or a daily failure rate above 20% write
  to `alerts/`. Everything else goes to the daily digest.

## Config

- `playbook.yaml` holds style and strategy (lengths, hook rules, scoring weights, subtitle style, packaging). It is
  versioned in the `playbook_versions` table; the Learner (Phase 3) will only ever edit this file.
- `rules.yaml` holds hard limits (durations, QA thresholds, banned terms, posting caps). No agent writes to it.
  Its `campaign` section (Kai Cenat GTA 6 Marathon) is enforced by the QA Gate: required mention and hashtag on
  every platform, allowed platforms, no logos, English only, and content rules (Kai the primary focus, no negative
  portrayal, no hate farming) judged by the LLM review, where any violation kills the clip.
- Secrets come only from the environment or `.env`. `HQ_LLM_ENABLED=0` forces every agent onto its fallback path.
