import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import Settings  # noqa: E402
from core.db import connect, init_db  # noqa: E402
from core.logging import configure  # noqa: E402


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("HQ_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HQ_ALERTS_DIR", str(tmp_path / "alerts"))
    monkeypatch.setenv("HQ_DB_PATH", str(tmp_path / "data" / "test.db"))
    monkeypatch.setenv("HQ_LLM_ENABLED", "0")
    s = Settings.from_env()
    s.ensure_dirs()
    configure(s.logs_dir, quiet=True)
    return s


@pytest.fixture
def conn(settings):
    c = connect(settings.db_path)
    init_db(c)
    return c


@pytest.fixture
def playbook(settings):
    from core.config import read_yaml
    pb = read_yaml(settings.config_dir / "playbook.yaml")
    pb["version"] = "1"
    return pb


@pytest.fixture
def rules(settings):
    from core.config import load_rules
    return load_rules(settings)


def make_words(sentences, start=0.0, wps=2.5, gap=0.3):
    """Build a word-level transcript from a list of sentences."""
    words, t = [], start
    for sent in sentences:
        for tok in sent.split():
            d = 1 / wps
            words.append({"w": tok, "s": round(t, 3), "e": round(t + d * 0.9, 3), "p": None})
            t += d
        t += gap
    return words


def transcript_from(sentences, **kw):
    words = make_words(sentences, **kw)
    return {"engine": "test", "language": "en",
            "segments": [{"start": words[0]["s"], "end": words[-1]["e"], "text": " ".join(sentences), "words": words}]}


class FakeLLM:
    """Stands in for core.llm.LLM. `responses` is a list of dicts or exceptions returned in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, system, prompt, schema, **kw):
        self.calls.append({"system": system, "prompt": prompt, "schema": schema})
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r(prompt) if callable(r) else r
