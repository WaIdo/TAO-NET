"""LLM client abstractions used during Stage-2 OOD classification."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openai = None


class BaseLLMClient:
    """Minimal interface every LLM backend must implement."""

    def generate(self, prompt: str, **kwargs) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class LocalEchoClient(BaseLLMClient):
    """Deterministic offline stub useful for testing the pipeline."""

    def __init__(self, template_hint: str = "This is a stub response."):
        self.template_hint = template_hint

    def generate(self, prompt: str, **kwargs) -> str:
        candidate_labels = kwargs.get("candidate_labels") or []
        label = candidate_labels[0] if candidate_labels else "UNKNOWN"
        payload = {
            "label": label,
            "rationale": f"{self.template_hint} Selected '{label}' heuristically.",
        }
        return json.dumps(payload)


class OpenAIClient(BaseLLMClient):
    """Lightweight wrapper around OpenAI's Chat Completions API."""

    def __init__(self,
                 model: str,
                 api_key: Optional[str] = None,
                 organization: Optional[str] = None,
                 max_retries: int = 3):
        if openai is None:
            raise ImportError("openai package is required for OpenAIClient")
        self.model = model
        self.max_retries = max_retries
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if organization:
            openai.organization = organization
        if not openai.api_key:
            raise ValueError("OpenAI API key not provided")

    def generate(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.get("temperature", 0.2)
        top_p = kwargs.get("top_p", 0.95)
        messages = kwargs.get("messages") or [
            {"role": "system", "content": "You label encrypted traffic."},
            {"role": "user", "content": prompt},
        ]
        for _ in range(self.max_retries):
            try:
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                )
                return resp.choices[0].message["content"].strip()
            except openai.error.OpenAIError as exc:  # type: ignore
                last_err = exc
        raise RuntimeError(f"OpenAI API failed after retries: {last_err}")
