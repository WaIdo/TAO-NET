"""Helper CLI to run PacRep ID-branch experiments across all datasets."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List

DATASETS = {
    "tinghua": Path("stage2_pacrep_tsinghua"),
    "vpn": Path("stage2_pacrep_vpn"),
    "nontor": Path("stage2_pacrep_nontor"),
}

DEFAULT_MODELS = {
    "tinghua": "att_tinghua_addvocab",
    "vpn": "att_VPN_addvocab",
    "nontor": "att_Nontor_addvocab",
}


def _fmt_score(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PacRep/BERT ID classifiers for TAO-Net Stage-2 experiments"
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS.keys(),
                        help="Dataset to train on (tinghua/vpn/nontor)")
    parser.add_argument("--name-model", type=str, default=None,
                        help="Name used for checkpoints; defaults to att_<dataset>_addvocab")
    parser.add_argument("train_args", nargs=argparse.REMAINDER,
                        help="Arguments forwarded to run_train.py (prefix with -- before the first one)")
    return parser.parse_args()


def run_training(dataset_key: str, model_name: str, extra_args: List[str]) -> None:
    workdir = DATASETS[dataset_key]
    cmd = ["python", "run_train.py", "--name_model", model_name]
    if extra_args:
        # argparse.REMAINDER keeps the leading '--' if provided, so strip it
        if extra_args[0] == "--":
            extra_args = extra_args[1:]
        cmd.extend(extra_args)
    print(f"[PacRep] Running {' '.join(cmd)} (cwd={workdir})")
    subprocess.run(cmd, cwd=workdir, check=True)

    summary_file = workdir / "saved_model" / model_name / "best_metrics.json"
    if summary_file.exists():
        with summary_file.open("r", encoding="utf-8") as fh:
            metrics = json.load(fh)
        print(f"[PacRep] Best metrics stored in {summary_file}")
        for metric, payload in metrics.items():
            best_scores = payload.get("best_scores", {})
            best_iter = payload.get("best_iter")
            print(f"  - {metric}: "
                  f"train={_fmt_score(best_scores.get('train'))} "
                  f"valid={_fmt_score(best_scores.get('valid'))} "
                  f"test={_fmt_score(best_scores.get('test'))} "
                  f"(iter={best_iter})")
    else:
        print(f"[PacRep] Summary file not found at {summary_file}")


def main() -> None:
    args = parse_args()
    dataset_key = args.dataset.lower()
    model_name = args.name_model or DEFAULT_MODELS[dataset_key]
    run_training(dataset_key, model_name, args.train_args)


if __name__ == "__main__":  # pragma: no cover
    main()
