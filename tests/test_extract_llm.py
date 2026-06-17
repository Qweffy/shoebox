from __future__ import annotations

import json
import os
from datetime import date
from decimal import Decimal

import pytest

from shoebox.config import Settings
from shoebox.extract import (
    LLMExtractor,
    LocalExtractor,
    OpenAICompatExtractor,
    RegexExtractor,
    get_extractor,
)
from shoebox.models import OcrLine, OcrResult

_BBOX = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def _ocr(lines: list[str]) -> OcrResult:
    return OcrResult(
        source="r.png",
        lines=[OcrLine(text=t, confidence=0.99, bbox=_BBOX) for t in lines],
    )


class _ScriptedLLM(LLMExtractor):
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = 0

    def _complete(self, system: str, user: str) -> str:
        response = self._responses[self.calls]
        self.calls += 1
        return response


def test_parses_valid_json_in_one_call() -> None:
    payload = '{"vendor": "CAFE MARGARITA", "date": "2026-06-12", "total": 7.0, "currency": "USD"}'
    llm = _ScriptedLLM([payload])
    fields = llm.extract(_ocr(["CAFE MARGARITA", "TOTAL 7.00"]))
    assert llm.calls == 1
    assert fields.vendor == "CAFE MARGARITA"
    assert fields.date == date(2026, 6, 12)
    assert fields.total == Decimal("7.0")
    assert fields.currency == "USD"


def test_repairs_once_then_succeeds() -> None:
    llm = _ScriptedLLM(["not json at all", '{"vendor": "SHOP", "total": 9.99}'])
    fields = llm.extract(_ocr(["SHOP"]))
    assert llm.calls == 2
    assert fields.vendor == "SHOP"
    assert fields.total == Decimal("9.99")


def test_raises_after_failed_repair() -> None:
    llm = _ScriptedLLM(["garbage", "still garbage"])
    with pytest.raises(json.JSONDecodeError):
        llm.extract(_ocr(["X"]))
    assert llm.calls == 2


def _settings(*, gemini_api_key: str | None = None, groq_api_key: str | None = None) -> Settings:
    return Settings(
        gemini_api_key=gemini_api_key,
        groq_api_key=groq_api_key,
        ollama_host="http://localhost:11434",
        ollama_model="qwen2.5:3b-instruct-q4_K_M",
        gemini_model="gemini-2.5-flash",
        groq_model="openai/gpt-oss-120b",
        groq_vision_model="meta-llama/llama-4-scout-17b-16e-instruct",
    )


def test_factory_returns_expected_types() -> None:
    assert isinstance(get_extractor("regex", _settings()), RegexExtractor)
    assert isinstance(get_extractor("local", _settings()), LocalExtractor)
    assert isinstance(get_extractor("gemini", _settings(gemini_api_key="k")), OpenAICompatExtractor)
    assert isinstance(get_extractor("groq", _settings(groq_api_key="k")), OpenAICompatExtractor)


def test_factory_requires_keys_and_known_name() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        get_extractor("gemini", _settings())
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        get_extractor("groq", _settings())
    with pytest.raises(ValueError, match="unknown extractor"):
        get_extractor("bogus", _settings())


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("SHOEBOX_RUN_OLLAMA") != "1",
    reason="set SHOEBOX_RUN_OLLAMA=1 to run the real local model",
)
def test_local_extractor_against_real_ollama() -> None:
    ocr = _ocr(["CAFE MARGARITA", "Date: 2026-06-12", "Espresso 3.50", "TOTAL 7.00"])
    fields = LocalExtractor(
        model="qwen2.5:3b-instruct-q4_K_M", host="http://localhost:11434"
    ).extract(ocr)
    assert fields.vendor is not None
    assert fields.total == Decimal("7.00")
