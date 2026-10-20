"""Runtime configuration, read from environment variables only."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    provider: str
    api_key: str | None
    model: str
    base_url: str
    db_path: str
    log_level: str
    agent_bridge_dir: str = "data/agent-bridge"
    agent_timeout: float = 120.0
    retrieval_k: int = 4
    max_candidates: int = 3


def default_provider(api_key: str | None) -> str:
    """OpenAI only when a key is present; otherwise the Cursor project agent answers through the file bridge."""
    return "openai" if api_key else "agent"


def load_settings() -> Settings:
    api_key = os.environ.get("OPENAI_API_KEY") or None
    provider = os.environ.get("AVATAR_LLM_PROVIDER") or default_provider(api_key)
    return Settings(
        provider=provider,
        api_key=api_key,
        model=os.environ.get("AVATAR_LLM_MODEL", "gpt-4o-mini"),
        base_url=os.environ.get("AVATAR_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        db_path=os.environ.get("AVATAR_DB_PATH", "data/avatar.db"),
        log_level=os.environ.get("AVATAR_LOG_LEVEL", "INFO"),
        agent_bridge_dir=os.environ.get("AVATAR_AGENT_BRIDGE_DIR", "data/agent-bridge"),
        agent_timeout=float(os.environ.get("AVATAR_AGENT_TIMEOUT", "120")),
    )
