"""Configuration loading: .env, playbook.yaml (versioned), rules.yaml.

Secrets come only from the environment (optionally loaded from a .env file);
nothing secret is ever hardcoded or written to the DB.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader. Existing environment variables always win."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    db_path: Path
    media_dir: Path
    alerts_dir: Path
    logs_dir: Path
    config_dir: Path
    assets_dir: Path
    llm_model: str
    llm_enabled: bool
    whisper_model: str
    whisper_fallback_model: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        data = Path(os.environ.get("HQ_DATA_DIR", ROOT / "data"))
        return cls(
            db_path=Path(os.environ.get("HQ_DB_PATH", data / "highlighthq.db")),
            media_dir=data / "media",
            alerts_dir=Path(os.environ.get("HQ_ALERTS_DIR", ROOT / "alerts")),
            logs_dir=data / "logs",
            config_dir=Path(os.environ.get("HQ_CONFIG_DIR", ROOT / "config")),
            assets_dir=ROOT / "assets",
            llm_model=os.environ.get("HQ_LLM_MODEL", "claude-sonnet-5-5"),
            # LLM is used only when explicitly enabled AND a credential exists.
            llm_enabled=os.environ.get("HQ_LLM_ENABLED", "1") == "1",
            whisper_model=os.environ.get("HQ_WHISPER_MODEL", "base.en"),
            whisper_fallback_model=os.environ.get("HQ_WHISPER_FALLBACK_MODEL", "tiny.en"),
        )

    def ensure_dirs(self) -> None:
        for d in (self.db_path.parent, self.media_dir, self.alerts_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)


def read_yaml(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def load_rules(settings: Settings) -> dict:
    """rules.yaml is human-owned. Agents (including the Learner) only read it."""
    return read_yaml(settings.config_dir / "rules.yaml")


def load_playbook(settings: Settings, conn=None) -> dict:
    """Return the active playbook. If the DB holds an active version it wins,
    otherwise the file is used and registered as the first version."""
    if conn is not None:
        row = conn.execute(
            "SELECT version, yaml FROM playbook_versions WHERE status='active' "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row:
            pb = yaml.safe_load(row["yaml"]) or {}
            pb["version"] = row["version"]
            return pb
    path = settings.config_dir / "playbook.yaml"
    pb = read_yaml(path)
    if conn is not None:
        conn.execute(
            "INSERT OR IGNORE INTO playbook_versions(version, yaml, status) VALUES (?,?, 'active')",
            (str(pb.get("version", "1")), path.read_text()),
        )
        conn.commit()
    pb["version"] = str(pb.get("version", "1"))
    return pb
