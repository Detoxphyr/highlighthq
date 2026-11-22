"""Thin Anthropic client for all LLM agents: JSON-schema structured output,
transient-error retries, a circuit breaker, and a clear LLMUnavailable signal
so each agent can take its heuristic fallback path."""
from __future__ import annotations

import json
import os

from core.breaker import BreakerOpen, CircuitBreaker
from core.logging import log
from core.retry import TransientError, retry


class LLMUnavailable(Exception):
    """No credential, breaker open, refusal, or persistent API failure."""


def _has_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


class LLM:
    def __init__(self, settings, conn=None, on_trip=None):
        self.model = settings.llm_model
        self.enabled = settings.llm_enabled and _has_credentials()
        self.breaker = CircuitBreaker(conn, "llm:anthropic", threshold=3, cooldown_s=900, on_trip=on_trip) if conn else None
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(max_retries=0)  # retries handled by core.retry
        return self._client

    def complete_json(self, system: str, prompt: str, schema: dict, max_tokens: int = 8000, effort: str = "medium") -> dict:
        if not self.enabled:
            raise LLMUnavailable("LLM disabled or no ANTHROPIC_API_KEY set")
        if self.breaker:
            try:
                self.breaker.check()
            except BreakerOpen as exc:
                raise LLMUnavailable(str(exc)) from exc
        try:
            result = self._call(system, prompt, schema, max_tokens, effort)
        except (TransientError, LLMUnavailable) as exc:
            if self.breaker:
                self.breaker.failure()
            raise LLMUnavailable(str(exc)) from exc
        if self.breaker:
            self.breaker.success()
        return result

    @retry(tries=3, base_delay=2.0)
    def _call(self, system: str, prompt: str, schema: dict, max_tokens: int, effort: str) -> dict:
        import anthropic
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
                # Server-side fallback on a policy refusal, routed by refusal category.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as exc:
            raise TransientError(f"{type(exc).__name__}: {exc}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise TransientError(f"HTTP {exc.status_code}") from exc
            raise LLMUnavailable(f"HTTP {exc.status_code}: {exc.message}") from exc
        if response.stop_reason == "refusal":
            raise LLMUnavailable(f"refusal: {getattr(response, 'stop_details', None)}")
        if response.stop_reason == "max_tokens":
            raise LLMUnavailable("response truncated at max_tokens")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMUnavailable("no text block in response")
        log("llm_call", model=response.model, input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens)
        return json.loads(text)
