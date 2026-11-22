"""Structured (JSON lines) logging. Every job transition is logged here and in
the `transitions` table."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_LOG_FILE: Path | None = None
_QUIET = False


def configure(logs_dir: Path, quiet: bool = False) -> None:
    global _LOG_FILE, _QUIET
    logs_dir.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = logs_dir / "highlighthq.jsonl"
    _QUIET = quiet


def log(event: str, level: str = "info", **fields) -> dict:
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": level, "event": event}
    record.update(fields)
    line = json.dumps(record, default=str)
    if _LOG_FILE is not None:
        with open(_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    if not _QUIET:
        print(line, file=sys.stderr)
    return record


def tail_for_job(job_id: int, limit: int = 200) -> list[dict]:
    """Return recent log records for one job (used for dead-letter logs)."""
    if _LOG_FILE is None or not _LOG_FILE.exists():
        return []
    out = []
    with open(_LOG_FILE, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("job_id") == job_id:
                out.append(rec)
    return out[-limit:]
