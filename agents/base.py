"""Simple worker loop shared by every agent.

An agent declares `name`, `input_status`, `kind` and implements `process(job)`,
which returns an AgentResult. The loop claims a job, processes it, and writes
the resulting transition. Exceptions become failures (requeue, then dead-letter
after 3 attempts). Agents never call other agents."""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field

from core import jobs as J
from core.alerts import raise_alert
from core.config import Settings, load_playbook, load_rules
from core.db import connect, init_db
from core.llm import LLM
from core.logging import configure, log


@dataclass
class AgentResult:
    to_status: str
    reason: str = ""
    payload: dict = field(default_factory=dict)
    decisions: dict = field(default_factory=dict)
    qa_loops: int | None = None
    reset_attempts: bool = True


class BaseAgent:
    name = "base"
    input_status = ""
    kind = "source"

    def __init__(self, settings: Settings | None = None, conn=None):
        self.settings = settings or Settings.from_env()
        self.settings.ensure_dirs()
        configure(self.settings.logs_dir, quiet=True)
        self.conn = conn or connect(self.settings.db_path)
        init_db(self.conn)
        self.playbook = load_playbook(self.settings, self.conn)
        self.rules = load_rules(self.settings)
        self.llm = LLM(self.settings, self.conn, on_trip=self._on_breaker_trip)

    def _on_breaker_trip(self, key: str, failures: int) -> None:
        raise_alert(self.conn, self.settings.alerts_dir, "breaker_tripped",
                    f"circuit breaker {key} opened after {failures} failures", breaker=key)

    def job_dir(self, job: J.Job):
        d = self.settings.media_dir / f"source_{job.source_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def process(self, job: J.Job) -> AgentResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def run_once(self) -> J.Job | None:
        job = J.claim(self.conn, self.input_status, self.kind, self.name)
        if job is None:
            return None
        log("job_claimed", job_id=job.id, agent=self.name, status=job.status)
        started = time.time()
        try:
            result = self.process(job)
        except Exception as exc:  # noqa: BLE001 - every failure must be recorded
            err = f"{type(exc).__name__}: {exc}"
            log("job_error", level="error", job_id=job.id, agent=self.name, error=err, trace=traceback.format_exc())
            return J.fail(self.conn, J.get(self.conn, job.id), self.name, err, self.input_status)
        job = J.get(self.conn, job.id)
        if result.to_status == "killed":
            out = J.transition(self.conn, job, "killed", self.name, result.reason, payload=result.payload,
                               decisions=result.decisions, qa_loops=result.qa_loops)
        else:
            out = J.transition(self.conn, job, result.to_status, self.name, result.reason, payload=result.payload,
                               decisions=result.decisions, qa_loops=result.qa_loops,
                               reset_attempts=result.reset_attempts)
        log("job_done", job_id=job.id, agent=self.name, to_status=out.status, seconds=round(time.time() - started, 2))
        return out

    def run(self, once: bool = False, poll_interval: float = 5.0, max_idle_polls: int | None = None) -> int:
        """Poll forever (or drain the queue when once=True). Returns jobs processed."""
        processed, idle = 0, 0
        while True:
            job = self.run_once()
            if job is not None:
                processed, idle = processed + 1, 0
                continue
            if once:
                return processed
            idle += 1
            if max_idle_polls is not None and idle >= max_idle_polls:
                return processed
            time.sleep(poll_interval)
