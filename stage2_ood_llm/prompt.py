"""Utilities for building prompts and parsing LLM responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .spk import SPKTemplate


@dataclass
class LLMGeneration:
    prompt: str
    raw_response: str
    label: str
    rationale: str


class PromptBuilder:
    """Format traffic samples into semantic-enhanced prompts for LLMs."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def build(self, template: SPKTemplate, sample_text: str, metadata: Dict[str, Any]) -> str:
        stage1_score = metadata.get("score")
        score_line = (
            f"Stage-1 hybrid OOD score: {stage1_score:.4f} (higher means further from ID baseline)\n"
            if stage1_score is not None else ""
        )
        additional = metadata.get("notes", "")
        prompt = (
            "You are a senior encrypted-traffic analyst.\n"
            f"Dataset: {self.dataset_name}\n"
            f"Sample identifier: {metadata.get('sample_id', 'unknown')}\n"
            f"Source stage: {metadata.get('stage', 'TAO-Net Stage-1')}\n"
            f"{score_line}"
            "Packet sequence (truncated to fit context):\n"
            f"{sample_text}\n\n"
            f"{template.instruction_block()}\n"
            f"Additional context: {additional}\n"
        )
        return prompt

    def parse_response(self, response: str, template: SPKTemplate) -> Tuple[str, str]:
        response = response.strip()
        # Prefer JSON responses whenever possible.
        try:
            payload = json.loads(response)
            label = payload.get("label", "").strip()
            rationale = payload.get("rationale", "").strip()
            return self._sanitize_label(label, rationale, template)
        except json.JSONDecodeError:
            pass

        # Fallback: look for a ``label`` key in plain text.
        match = re.search(r"label\s*[:=]\s*['\"]?([A-Za-z0-9_\- ]+)", response, re.IGNORECASE)
        if match:
            label = match.group(1).strip()
            rationale = response
            return self._sanitize_label(label, rationale, template)

        # Last resort: pick the first candidate label mentioned in the response.
        for candidate in template.candidate_labels:
            pattern = re.compile(re.escape(candidate), re.IGNORECASE)
            if pattern.search(response):
                return candidate, response

        return "UNKNOWN", response

    @staticmethod
    def _sanitize_label(label: str, rationale: str, template: SPKTemplate) -> Tuple[str, str]:
        normalized = label.strip().lower()
        for candidate in template.candidate_labels:
            if normalized == candidate.lower():
                return candidate, rationale
        return "UNKNOWN", rationale
