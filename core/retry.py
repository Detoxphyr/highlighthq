"""Retry with exponential backoff for transient errors, plus a fallback helper."""
from __future__ import annotations

import functools
import random
import subprocess
import time
from typing import Callable, TypeVar

from core.logging import log

T = TypeVar("T")


class TransientError(Exception):
    """Errors worth retrying (network blips, rate limits, flaky subprocesses)."""


class PersistentError(Exception):
    """Errors that retrying will not fix (bad input, missing model, policy)."""


def retry(tries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0,
          retry_on: tuple[type[BaseException], ...] = (TransientError,),
          sleep: Callable[[float], None] = time.sleep):
    """Retry `tries` times total with exponential backoff + jitter."""

    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            for attempt in range(1, tries + 1):
                try:
                    return fn(*args, **kwargs)
                except retry_on as exc:
                    if attempt == tries:
                        log("retry_exhausted", level="warning", fn=fn.__name__, attempt=attempt, error=str(exc))
                        raise
                    delay = min(max_delay, base_delay * 2 ** (attempt - 1)) * (0.8 + 0.4 * random.random())
                    log("retry", level="warning", fn=fn.__name__, attempt=attempt, delay_s=round(delay, 2), error=str(exc))
                    sleep(delay)
            raise AssertionError("unreachable")

        return wrapper

    return deco


def with_fallback(primary: Callable[[], T], fallback: Callable[[], T], what: str, **ctx) -> tuple[T, str]:
    """Run primary; on any failure run fallback. Returns (result, path_used)."""
    try:
        return primary(), "primary"
    except Exception as exc:  # noqa: BLE001 - fallback is deliberately broad
        log("fallback", level="warning", what=what, error=f"{type(exc).__name__}: {exc}", **ctx)
        return fallback(), "fallback"


def run_cmd(cmd: list[str], timeout: float = 600, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    """Run a subprocess. Timeouts/signals are transient; non-zero exits persistent."""
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, input=input_bytes)
    except subprocess.TimeoutExpired as exc:
        raise TransientError(f"timeout after {timeout}s: {cmd[0]}") from exc
    if proc.returncode < 0:
        raise TransientError(f"{cmd[0]} killed by signal {-proc.returncode}")
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace")[-2000:]
        raise PersistentError(f"{cmd[0]} exited {proc.returncode}: {err}")
    return proc
