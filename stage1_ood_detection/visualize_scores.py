"""Quick histogram visualizer for Stage-1 hybrid scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ID/OOD score histograms with the hybrid threshold")
    parser.add_argument("--scores-file", type=str, required=True,
                        help="JSON file produced by run.py --save-scores")
    parser.add_argument("--output", type=str, default="score_hist.png",
                        help="Destination image path")
    parser.add_argument("--title", type=str, default=None,
                        help="Optional plot title override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores_path = Path(args.scores_file)
    with scores_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - visualization helper
        raise SystemExit("matplotlib is required for visualization") from exc

    threshold = payload["threshold"]
    scores_id = payload["scores_id"]
    scores_ood = payload["scores_ood"]
    title = args.title or f"{payload.get('dataset', '')} Stage-1 Hybrid Scores"

    plt.figure(figsize=(8, 5))
    plt.hist(scores_id, bins=80, alpha=0.6, label="ID (Stage-1 bucket)", color="#4c72b0")
    plt.hist(scores_ood, bins=80, alpha=0.6, label="OOD (Stage-1 bucket)", color="#dd8452")
    plt.axvline(threshold, color="#55a868", linestyle="--", label=f"δ={threshold:.4f}")
    plt.xlabel("Hybrid score")
    plt.ylabel("Count")
    plt.title(title.strip())
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved histogram to {args.output}")


if __name__ == "__main__":  # pragma: no cover - CLI tool
    main()
