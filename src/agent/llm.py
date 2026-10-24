"""Local LLM access through Ollama.

No API keys, no cloud calls. If Ollama is not running, callers fall back to
deterministic interventions so a nightly briefing can still be produced.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL = "qwen3"
DEFAULT_HOST = "http://localhost:11434"

_THINK_BLOCK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


class LLMUnavailable(RuntimeError):
    """Ollama is not running, the model is missing, or generation failed."""


@dataclass(frozen=True)
class LLMStatus:
    available: bool
    model: str
    host: str
    models: tuple[str, ...] = ()
    detail: str = ""


class OllamaLLM:
    """Small wrapper with a low temperature and strict JSON parsing."""

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        *,
        temperature: float = 0.2,
        timeout: float = 120.0,
        think: bool | None = None,
    ) -> None:
        self.model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
        self.host = host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST)
        self.temperature = temperature
        self.timeout = timeout
        # qwen3 is a reasoning model. Left on, it spends the whole token budget
        # inside <think> and returns an empty answer for short prompts, so the
        # agent silently falls back to heuristics. Think off unless asked for.
        self.think = think if think is not None else _env_flag("OLLAMA_THINK", False)
        self._client: Any | None = None
        self._supports_think = True

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import ollama
            except ImportError as exc:  # pragma: no cover - depends on install
                raise LLMUnavailable("ollama package is not installed") from exc
            try:
                self._client = ollama.Client(host=self.host, timeout=self.timeout)
            except TypeError:  # older ollama-python without timeout kwarg
                self._client = ollama.Client(host=self.host)
        return self._client

    def list_models(self) -> tuple[str, ...]:
        try:
            response = self._get_client().list()
        except Exception as exc:
            raise LLMUnavailable(f"could not reach Ollama at {self.host}: {exc}") from exc
        models = getattr(response, "models", None) or []
        names: list[str] = []
        for item in models:
            name = getattr(item, "model", None) or getattr(item, "name", None)
            if name:
                names.append(str(name))
        return tuple(names)

    def status(self) -> LLMStatus:
        try:
            models = self.list_models()
        except LLMUnavailable as exc:
            return LLMStatus(False, self.model, self.host, detail=str(exc))
        found = any(m == self.model or m.startswith(f"{self.model}:") for m in models)
        detail = "" if found else f"{self.model} not pulled; run: ollama pull {self.model}"
        return LLMStatus(found, self.model, self.host, models, detail)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
    ) -> str:
        client = self._get_client()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "format": "json" if json_mode else None,
            "options": {"temperature": self.temperature, "num_predict": 600},
        }
        if self._supports_think:
            kwargs["think"] = self.think
        try:
            try:
                response = client.chat(**kwargs)
            except TypeError:
                # older ollama-python without the `think` kwarg
                self._supports_think = False
                kwargs.pop("think", None)
                response = client.chat(**kwargs)
        except Exception as exc:
            raise LLMUnavailable(f"Ollama generation failed: {exc}") from exc

        message = getattr(response, "message", None)
        if message is None and isinstance(response, dict):
            message = response.get("message", {})
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        text = str(content or "").strip()
        if not text:
            raise LLMUnavailable("Ollama returned an empty response")
        cleaned = _THINK_BLOCK.sub("", text).strip()
        if not cleaned:
            raise LLMUnavailable(
                "Ollama spent the whole response on reasoning and returned no "
                "answer; set OLLAMA_THINK=0 or raise num_predict"
            )
        return cleaned

    def generate_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
    ) -> dict[str, Any]:
        text = self.generate(prompt, system=system, json_mode=True)
        return _parse_json_object(text)


def _parse_json_object(text: str) -> dict[str, Any]:
    """Tolerate fenced JSON and a little chatter around the object."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise LLMUnavailable(f"model did not return JSON: {text[:200]!r}")
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"invalid JSON from model: {text[:200]!r}") from exc
    if not isinstance(value, dict):
        raise LLMUnavailable("model returned JSON that is not an object")
    return value
