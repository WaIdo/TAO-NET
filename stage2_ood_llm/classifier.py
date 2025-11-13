"""Stage-2 controller that drives SPK prompts through an LLM backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .llm_client import BaseLLMClient
from .prompt import LLMGeneration, PromptBuilder
from .spk import SemanticEnhancedPromptKnowledge


@dataclass
class SampleRecord:
    text: str
    score: float
    sample_id: str
    metadata: Dict[str, str]


class OODLLMClassifier:
    def __init__(self,
                 dataset_name: str,
                 llm_client: BaseLLMClient,
                 spk: Optional[SemanticEnhancedPromptKnowledge] = None,
                 prompt_builder: Optional[PromptBuilder] = None):
        self.dataset_name = dataset_name
        self.llm = llm_client
        self.spk = spk or SemanticEnhancedPromptKnowledge(dataset_name)
        self.prompt_builder = prompt_builder or PromptBuilder(dataset_name)

    def classify(self,
                 samples: Iterable[SampleRecord],
                 mode: str = "strict",
                 llm_kwargs: Optional[Dict] = None) -> List[LLMGeneration]:
        template = self.spk.get(mode)
        generations: List[LLMGeneration] = []
        for sample in samples:
            metadata = {
                "sample_id": sample.sample_id,
                "score": sample.score,
                "stage": sample.metadata.get("stage", "TAO-Net Stage-2"),
                "notes": sample.metadata.get("notes", ""),
            }
            prompt = self.prompt_builder.build(template, sample.text, metadata)
            raw = self.llm.generate(
                prompt,
                candidate_labels=template.candidate_labels,
                **(llm_kwargs or {}),
            )
            label, rationale = self.prompt_builder.parse_response(raw, template)
            generations.append(LLMGeneration(prompt=prompt, raw_response=raw, label=label, rationale=rationale))
        return generations
