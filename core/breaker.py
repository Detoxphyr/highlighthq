"""Circuit breaker per account/platform/tool, persisted in SQLite so every
worker process sees the same state."""
from __future__ import annotations

import time

from core.logging import log


class BreakerOpen(Exception):
    pass


class CircuitBreaker:
    def __init__(self, conn, key: str, threshold: int = 5, cooldown_s: float = 600,
                 on_trip=None, clock=time.time):
        self.conn, self.key, self.threshold, self.cooldown_s = conn, key, threshold, cooldown_s
        self.on_trip = on_trip
        self.clock = clock
        conn.execute("INSERT OR IGNORE INTO breakers(key) VALUES (?)", (key,))

    def _row(self):
        return self.conn.execute("SELECT * FROM breakers WHERE key=?", (self.key,)).fetchone()

    @property
    def state(self) -> str:
        row = self._row()
        if row["state"] == "open" and self.clock() - (row["opened_at"] or 0) >= self.cooldown_s:
            self.conn.execute("UPDATE breakers SET state='half_open' WHERE key=?", (self.key,))
            log("breaker_half_open", key=self.key)
            return "half_open"
        return row["state"]

    def allow(self) -> bool:
        return self.state != "open"

    def check(self) -> None:
        if not self.allow():
            raise BreakerOpen(f"circuit open for {self.key}")

    def success(self) -> None:
        self.conn.execute("UPDATE breakers SET state='closed', failures=0, opened_at=NULL WHERE key=?", (self.key,))

    def failure(self) -> None:
        state = self.state
        self.conn.execute("UPDATE breakers SET failures=failures+1 WHERE key=?", (self.key,))
        failures = self._row()["failures"]
        if state == "half_open" or failures >= self.threshold:
            self.conn.execute("UPDATE breakers SET state='open', opened_at=? WHERE key=?", (self.clock(), self.key))
            log("breaker_tripped", level="error", key=self.key, failures=failures)
            if self.on_trip:
                self.on_trip(self.key, failures)
