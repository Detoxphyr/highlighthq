#!/usr/bin/env python3
"""HighlightHQ CLI.

  python run.py init-db
  python run.py add <url-or-path> [--title T] [--peaks FILE] [--source-date YYYY-MM-DD]
  python run.py run-agent <ingest|moment_finder|clip_editor|packager|qa_gate> [--once]
  python run.py run-pipeline            # drain every Phase 1 agent until no work is left
  python run.py status                  # job table
  python run.py digest                  # write alerts/digest-YYYY-MM-DD.md
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from core import jobs as J
from core.alerts import write_digest
from core.config import Settings, load_playbook
from core.db import connect, init_db
from core.logging import configure

PHASE1 = ["ingest", "moment_finder", "clip_editor", "packager", "qa_gate"]


def agent_class(name: str):
    if name == "ingest":
        from agents.ingest import IngestAgent as A
    elif name == "moment_finder":
        from agents.moment_finder import MomentFinderAgent as A
    elif name == "clip_editor":
        from agents.clip_editor import ClipEditorAgent as A
    elif name == "packager":
        from agents.packager import PackagerAgent as A
    elif name == "qa_gate":
        from agents.qa_gate import QAGateAgent as A
    else:
        raise SystemExit(f"unknown agent {name}")
    return A


def run_pipeline(settings: Settings, conn, max_rounds: int = 50) -> int:
    """Phase 1 stand-in for the Orchestrator: round-robin every agent until idle."""
    agents = [agent_class(n)(settings, conn) for n in PHASE1]
    total = 0
    for _ in range(max_rounds):
        n = sum(a.run(once=True) for a in agents)
        total += n
        if n == 0:
            break
    return total


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("(no jobs)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="HighlightHQ clipping pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    a = sub.add_parser("add")
    a.add_argument("source")
    a.add_argument("--title")
    a.add_argument("--peaks", help="JSON peaks file (kai_scan.py output or a list of seconds); default: "
                                   "<video>.peaks.json next to a local file, if present")
    a.add_argument("--source-date", help="date the footage was streamed (YYYY-MM-DD), checked against "
                                         "rules.yaml campaign.min_source_date")
    r = sub.add_parser("run-agent")
    r.add_argument("agent", choices=PHASE1)
    r.add_argument("--once", action="store_true")
    r.add_argument("--poll", type=float, default=5.0)
    sub.add_parser("run-pipeline")
    sub.add_parser("status")
    sub.add_parser("digest")
    args = ap.parse_args(argv)

    settings = Settings.from_env()
    settings.ensure_dirs()
    configure(settings.logs_dir, quiet=True)
    conn = connect(settings.db_path)
    init_db(conn)

    if args.cmd == "init-db":
        pb = load_playbook(settings, conn)
        print(f"initialized {settings.db_path} (playbook v{pb['version']})")
    elif args.cmd == "add":
        from agents.moment_finder import load_peaks
        peaks_file = args.peaks
        if not peaks_file and not re.match(r"^https?://", args.source):
            sidecar = Path(args.source).expanduser().with_suffix(".peaks.json")
            peaks_file = str(sidecar) if sidecar.exists() else None
        peaks = load_peaks(peaks_file) if peaks_file else None
        job_id = J.add_source(conn, args.source, args.title, peaks=peaks, source_date=args.source_date)
        print(f"added source job {job_id} (scouted)" + (f" with {len(peaks)} peaks from {peaks_file}" if peaks else ""))
    elif args.cmd == "run-agent":
        n = agent_class(args.agent)(settings, conn).run(once=args.once, poll_interval=args.poll)
        print(f"{args.agent}: processed {n} job(s)")
    elif args.cmd == "run-pipeline":
        n = run_pipeline(settings, conn)
        print(f"pipeline processed {n} job step(s)\n")
        print_table(J.job_table(conn))
    elif args.cmd == "status":
        print_table(J.job_table(conn))
    elif args.cmd == "digest":
        print(write_digest(conn, settings.alerts_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
