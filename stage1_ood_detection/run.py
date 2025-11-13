# -*- coding: utf-8 -*-
"""Stage-1 hybrid OOD scoring runner with calibrated thresholds."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch import nn

import my_datasets as md
import my_models as mm
import my_uncertainty as mu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage-1 OOD detection with hybrid PCA+smoothness scoring"
    )
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Weight for PCA residual score when fusing with smoothness")
    parser.add_argument("--threshold-method", choices=["youden", "percentile", "fixed"],
                        default="youden", help="Strategy used to derive the detection threshold δ")
    parser.add_argument("--fixed-threshold", type=float, default=None,
                        help="Explicit δ when --threshold-method=fixed")
    parser.add_argument("--percentile", type=float, default=95.0,
                        help="Percentile (ID distribution) used when threshold-method=percentile")
    parser.add_argument("--output-dir", type=str, default="stage1_outputs",
                        help="Where to persist hybrid scores for downstream LLM classification")
    parser.add_argument("--results-file", type=str, default="results_tinghuaall.pkl",
                        help="Pickle file storing aggregated metrics")
    parser.add_argument("--save-scores", action="store_true",
                        help="Persist per-sample hybrid scores + thresholds to JSON")
    parser.add_argument("--gpu", type=str, default="cuda:0",
                        help="CUDA device id used for model fine-tuning")
    parser.add_argument("--num-seeds", type=int, default=5,
                        help="Number of random seeds for repeated experiments")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_by_id(scores_id: np.ndarray, scores_ood: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize scores using ID statistics (z-score)."""
    scores_id = np.asarray(scores_id, dtype=np.float64)
    scores_ood = np.asarray(scores_ood, dtype=np.float64)
    mean = np.mean(scores_id)
    std = np.std(scores_id) + 1e-8
    return (scores_id - mean) / std, (scores_ood - mean) / std


def calibrate_threshold(scores_id: np.ndarray,
                        scores_ood: np.ndarray,
                        *,
                        method: str,
                        fixed_threshold: float | None,
                        percentile: float) -> float:
    scores_id = np.asarray(scores_id, dtype=np.float64)
    scores_ood = np.asarray(scores_ood, dtype=np.float64)
    if method == "fixed":
        if fixed_threshold is None:
            raise ValueError("fixed threshold method selected but --fixed-threshold missing")
        return float(fixed_threshold)
    if method == "percentile":
        return float(np.percentile(scores_id, percentile))

    # Use Youden's J index to search for the optimal threshold
    combined = np.unique(np.concatenate([scores_id, scores_ood]))
    best_delta = combined[0]
    best_score = -np.inf
    for delta in combined:
        tpr = np.mean(scores_ood >= delta)
        fpr = np.mean(scores_id >= delta)
        score = tpr - fpr
        if score > best_score:
            best_score = score
            best_delta = delta
    return float(best_delta)


def evaluate_scores(scores_id: np.ndarray,
                    scores_ood: np.ndarray,
                    threshold: float) -> Dict[str, float | List[List[int]]]:
    scores_id = np.asarray(scores_id, dtype=np.float64)
    scores_ood = np.asarray(scores_ood, dtype=np.float64)
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([
        np.zeros_like(scores_id, dtype=np.int64),
        np.ones_like(scores_ood, dtype=np.int64)
    ])
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    metrics = {
        "confusion_matrix": cm.tolist(),
        "f1_micro": float(f1_score(labels, preds, average="micro")),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "auroc": float(roc_auc_score(labels, scores)),
        "tpr": float(np.mean(scores_ood >= threshold)),
        "fpr": float(np.mean(scores_id >= threshold)),
        "threshold": float(threshold),
    }
    return metrics


def save_hybrid_scores(output_dir: Path,
                       dataset_name: str,
                       model_name: str,
                       seed: int,
                       scores_id: Iterable[float],
                       scores_ood: Iterable[float],
                       metrics: Dict[str, float | List[List[int]]],
                       id_indices: Iterable[int],
                       ood_indices: Iterable[int]) -> None:
    out_dir = output_dir / dataset_name / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "dataset": dataset_name,
        "model_name": model_name,
        "threshold": metrics["threshold"],
        "scores_id": list(map(float, scores_id)),
        "scores_ood": list(map(float, scores_ood)),
        "metrics": metrics,
        "test_indices": {
            "id": list(map(int, id_indices)),
            "ood": list(map(int, ood_indices)),
        },
    }
    with open(out_dir / f"seed_{seed}_hybrid.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_experiment(args: argparse.Namespace) -> None:
    start_time = time.time()
    DATASETS = [md.TinghuaallData()]
    MODEL_NAMES = ["RoBERTa"]
    LONG_DATA = {"Tinghuaall"}
    BATCH_SIZE = 32
    BATCH_SIZE_INFERENCE = 32

    UNCERNS = [
        mu.CombinedBLOODPCAQuant(),
        mu.PCAResidualQuant(),
        mu.BLOODQuant(),
        mu.LeastConfidentQuant(),
        mu.EntropyQuant(),
    ]

    results: Dict[str, Dict[str, Dict[str, List[Dict[str, Dict]]]]] = {}
    output_dir = Path(args.output_dir)

    for data in DATASETS:
        set_seed(42 + len(data.name))
        print(f"{data.name} - {time.time() - start_time}s")
        X_train_id, X_test_all, y_train_id, y_test_all, _ = data.load()

        id_indices = [idx for idx, y in enumerate(y_test_all) if y == 0]
        ood_indices = [idx for idx, y in enumerate(y_test_all) if y == 1]
        X_test_id = [X_test_all[idx] for idx in id_indices]
        X_test_ood = [X_test_all[idx] for idx in ood_indices]

        results[data.name] = {}
        for model_name in MODEL_NAMES:
            print(f"\t{model_name} - {time.time() - start_time}s")
            results[data.name][model_name] = {"fine-tuned": []}

            model = mm.TransformerClassifier(model_name, data.num_out, device=torch.device(args.gpu))
            criterion = nn.BCEWithLogitsLoss().to(model.device)

            for seed in range(args.num_seeds):
                set_seed(seed)
                print(f"\t\tSeed: {seed + 1} - {time.time() - start_time}s")
                rez_seed: Dict[str, Dict] = {}
                train_batch_size = BATCH_SIZE // 2 if data.name in LONG_DATA else BATCH_SIZE
                inference_batch_size = 16 if data.name in LONG_DATA else BATCH_SIZE_INFERENCE
                model.train_loop(
                    X_train_id,
                    y_train_id,
                    criterion=criterion,
                    batch_size=train_batch_size,
                    cartography=False,
                )

                for uncertainty in UNCERNS:
                    print(f"\t\t\t{uncertainty.name} - {time.time() - start_time}s")
                    rez_seed[uncertainty.name] = {}
                    score_cache = {}
                    for X, split_name in zip([X_test_id, X_test_ood], ["id", "ood"]):
                        print(f"\t\t\t\t{split_name} - {time.time() - start_time}s")
                        kwargs = {
                            "X_eval": X,
                            "X_anchor": X_train_id,
                            "y_anchor": y_train_id,
                            "model": model,
                            "criterion": criterion,
                            "batch_size": inference_batch_size,
                        }
                        score_cache[split_name] = np.array(uncertainty.quantify(**kwargs))
                        rez_seed[uncertainty.name][split_name] = score_cache[split_name].tolist()

                    threshold = calibrate_threshold(
                        score_cache['id'],
                        score_cache['ood'],
                        method=args.threshold_method,
                        fixed_threshold=args.fixed_threshold,
                        percentile=args.percentile,
                    )
                    metrics = evaluate_scores(score_cache['id'], score_cache['ood'], threshold)
                    rez_seed[uncertainty.name]['metrics'] = metrics

                    print(f"\t\t\tConfusion Matrix for {uncertainty.name}: {metrics['confusion_matrix']}")
                    print(f"\t\t\tMicro F1 Score: {metrics['f1_micro']}")
                    print(f"\t\t\tMacro F1 Score: {metrics['f1_macro']}")
                    print(f"\t\t\tAUROC: {metrics['auroc']}")

                if {'PCA_Residual', 'BLOOD'}.issubset(rez_seed.keys()):
                    res_id = np.array(rez_seed['PCA_Residual']['id'], dtype=np.float64)
                    res_ood = np.array(rez_seed['PCA_Residual']['ood'], dtype=np.float64)
                    blood_id = np.array(rez_seed['BLOOD']['id'], dtype=np.float64)
                    blood_ood = np.array(rez_seed['BLOOD']['ood'], dtype=np.float64)
                    res_id_norm, res_ood_norm = normalize_by_id(res_id, res_ood)
                    blood_id_norm, blood_ood_norm = normalize_by_id(blood_id, blood_ood)
                    hybrid_id = args.alpha * res_id_norm + (1 - args.alpha) * blood_id_norm
                    hybrid_ood = args.alpha * res_ood_norm + (1 - args.alpha) * blood_ood_norm
                    hybrid_threshold = calibrate_threshold(
                        hybrid_id,
                        hybrid_ood,
                        method=args.threshold_method,
                        fixed_threshold=args.fixed_threshold,
                        percentile=args.percentile,
                    )
                    hybrid_metrics = evaluate_scores(hybrid_id, hybrid_ood, hybrid_threshold)
                    rez_seed['Hybrid'] = {
                        'id': hybrid_id.tolist(),
                        'ood': hybrid_ood.tolist(),
                        'metrics': hybrid_metrics,
                        'test_indices': {
                            'id': id_indices,
                            'ood': ood_indices,
                        },
                    }

                    if args.save_scores:
                        save_hybrid_scores(
                            output_dir,
                            data.name,
                            model_name,
                            seed,
                            hybrid_id,
                            hybrid_ood,
                            hybrid_metrics,
                            id_indices,
                            ood_indices,
                        )

                results[data.name][model_name]["fine-tuned"].append(rez_seed)
                with open(args.results_file, "wb") as f:
                    pickle.dump(results, f)

    print("DONE")


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
