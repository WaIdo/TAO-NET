"""Command-line entry point for the Stage-2 LLM/SPK pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Type

from stage1_ood_detection import my_datasets as md

try:  # Support both ``python -m`` and direct execution
    from .classifier import OODLLMClassifier, SampleRecord
    from .llm_client import LocalEchoClient, OpenAIClient
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from stage2_ood_llm.classifier import OODLLMClassifier, SampleRecord
    from stage2_ood_llm.llm_client import LocalEchoClient, OpenAIClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label Stage-1 OOD traffic via SPK+LLM")
    parser.add_argument("--dataset", type=str, default="Tinghuaall",
                        help="Dataset name (e.g., Tinghuaall, VPN, Nontor)")
    parser.add_argument("--scores-file", type=str, required=True,
                        help="Hybrid score JSON produced by stage1 run")
    parser.add_argument("--mode", type=str, default="strict",
                        choices=["strict", "complete", "extended"],
                        help="SPK template mode")
    parser.add_argument("--max-samples", type=int, default=32,
                        help="Limit the number of OOD samples sent to the LLM")
    parser.add_argument("--client", choices=["local", "openai"], default="local",
                        help="LLM backend to use")
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini",
                        help="OpenAI model name (when --client=openai)")
    parser.add_argument("--output", type=str, default="stage2_llm_outputs.jsonl",
                        help="Where to store generated labels")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="LLM sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="LLM nucleus sampling parameter")
    return parser.parse_args()


def _resolve_dataset_class(dataset_name: str) -> Type:
    attr = f"{dataset_name}Data"
    dataset_cls = getattr(md, attr, None)
    if dataset_cls is None:
        raise ValueError(f"Dataset loader '{attr}' not defined in my_datasets.py")
    return dataset_cls


def _load_stage1_scores(scores_path: Path) -> dict:
    with open(scores_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _prepare_samples(dataset_name: str,
                     scores_payload: dict,
                     max_samples: int) -> List[SampleRecord]:
    dataset_cls = _resolve_dataset_class(dataset_name)
    dataset = dataset_cls()
    _, X_test, _, y_test, _ = dataset.load()
    test_records = list(getattr(dataset, "iter_test_records", lambda: [])())
    if not test_records:
        test_records = [{"text": text, "label": None} for text in X_test]

    scores_ood = scores_payload["scores_ood"]
    threshold = scores_payload["threshold"]
    test_indices = scores_payload.get("test_indices", {})
    ood_indices = test_indices.get("ood")
    if ood_indices is None:
        ood_indices = [idx for idx, label in enumerate(y_test) if label == 1]

    if len(ood_indices) != len(scores_ood):
        raise ValueError("Mismatch between OOD indices and score array length")

    samples = []
    for local_idx, (sample_idx, score) in enumerate(zip(ood_indices, scores_ood)):
        record = test_records[sample_idx]
        text = record.get("text", X_test[sample_idx])
        label = record.get("label")
        if score < threshold:
            continue
        sample_id = f"{dataset_name}-OOD-{sample_idx}"
        samples.append(SampleRecord(
            text=text,
            score=score,
            sample_id=sample_id,
            metadata={
                "stage": "Stage-1 hybrid",
                "notes": f"gt_label={label}" if label is not None else "auto-selected",
            },
        ))
    return samples[:max_samples]


def build_client(args: argparse.Namespace):
    if args.client == "local":
        return LocalEchoClient()
    return OpenAIClient(model=args.openai_model)


def main() -> None:
    args = parse_args()
    scores_path = Path(args.scores_file)
    payload = _load_stage1_scores(scores_path)
    samples = _prepare_samples(args.dataset, payload, args.max_samples)
    if not samples:
        print("No OOD samples exceeded the hybrid threshold; nothing to classify.")
        return

    client = build_client(args)
    classifier = OODLLMClassifier(dataset_name=args.dataset, llm_client=client)
    generations = classifier.classify(
        samples,
        mode=args.mode,
        llm_kwargs={"temperature": args.temperature, "top_p": args.top_p},
    )

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        for record, generation in zip(samples, generations):
            f.write(json.dumps({
                "sample_id": record.sample_id,
                "score": record.score,
                "label": generation.label,
                "rationale": generation.rationale,
                "raw_response": generation.raw_response,
            }, ensure_ascii=False) + "\n")
    print(f"Saved {len(generations)} labeled OOD samples to {output_path}")

if __name__ == "__main__":
    main()
