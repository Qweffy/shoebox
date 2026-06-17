"""Runtime configuration loaded from .env.

Secrets never live in code. `load_settings()` reads the environment (populated from
`.env` via python-dotenv) and exposes provider keys + model ids. M1 needs none of these;
M2's cloud adapters do.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    groq_api_key: str | None
    ollama_host: str
    ollama_model: str
    gemini_model: str
    groq_model: str
    groq_vision_model: str


def load_settings() -> Settings:
    return Settings(
        gemini_api_key=os.environ.get("GEMINI_API_KEY") or None,
        groq_api_key=os.environ.get("GROQ_API_KEY") or None,
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-instruct-q4_K_M"),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        groq_model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
        groq_vision_model=os.environ.get(
            "GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
        ),
    )
