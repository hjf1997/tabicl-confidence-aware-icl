"""Controlled label-contamination experiment: does method lift scale with noise?

Injects synthetic label noise into the TRAIN split only (random bogus rows
flipped to fraud), runs the full reliability pipeline on the contaminated
data, and evaluates on the untouched val/test splits. Because the injected
rows are known, this yields two things natural data cannot:

1. Detection quality: ROC-AUC of the reliability score for identifying the
   injected rows among fraud-labeled cases — the scorer's first validation
   against known ground truth. (Random flips are the EASY case: typical bogus
   rows, far from the boundary. Natural mislabeling is ambiguity-correlated,
   so detection results are an upper bound.)
2. The lift-vs-pi curve: for each contamination rate pi, method performance
   vs two reference arms sharing the random-positive policy:
   - baseline_fully_random: random negatives from the contaminated fraud pile
     (degrades with pi)
   - oracle: random negatives from the fraud pile EXCLUDING injected rows
     (ceiling of perfect noise removal; "method captures X% of the oracle gap")

pi is the INJECTED noise fraction of the resulting fraud-labeled pile:
n_flip = pi * n_fraud / (1 - pi). It stacks on top of the natural noise
(estimated rho ~ 0.1-0.2), which is present in all arms including pi=0.

Pre-registered predictions: detection AUC high; baseline degrades ~linearly
in pi; methods degrade slower; oracle slowest; method-vs-baseline gap grows
with pi. If the gap does not grow, the noise-dependence story is wrong.

Usage:
    python run_contamination.py --setting setting5 --top-features 150
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "supporting_set_constr"))

import torch.multiprocessing as mp

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, precision_recall_curve,
)

from config import (
    DataConfig, TabICLConfig, MultiGPUConfig, ReliabilityConfig,
    SupportSetConfig, PROJECT_ROOT, DEFAULT_MODEL_PATH,
)
from data_loader import DataLoader
from multi_gpu_inference import MultiGPUInference
from reliability_scorer import ReliabilityScorer, variance_decomposition
from support_set_selector import SupportSetSelector

logger = logging.getLogger(__name__)

NEG_SAMPLING_METHODS = ["random", "reliable", "informed", "kmeans", "boundary", "hybrid"]


def compute_metrics(y_true, y_proba, threshold=None):
    """Positive class = bogus (label 0). threshold=None finds the F1-optimal
    threshold (use on val); pass val's threshold when scoring test."""
    pos_proba = y_proba[:, 0]
    y_binary = (y_true == 0).astype(int)

    roc_auc = roc_auc_score(y_binary, pos_proba)
    pr_auc = average_precision_score(y_binary, pos_proba)

    if threshold is None:
        prec_c, rec_c, thresholds = precision_recall_curve(y_binary, pos_proba)
        f1_scores = 2 * (prec_c * rec_c) / (prec_c + rec_c + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        threshold = float(thresholds[optimal_idx]) if optimal_idx < len(thresholds) else 0.5

    y_pred = (pos_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1_score(y_binary, y_pred)),
        "precision": float(precision_score(y_binary, y_pred, zero_division=0)),
        "recall": float(recall_score(y_binary, y_pred, zero_division=0)),
        "optimal_threshold": float(threshold),
    }


def contaminate(y_train: np.ndarray, pi: float, positive_class: int,
                negative_class: int, rng: np.random.Generator):
    """Flip random bogus rows to fraud so injected rows are a pi-fraction of
    the resulting fraud pile. Returns (y_contaminated, flipped_row_indices)."""
    if pi <= 0:
        return y_train.copy(), np.array([], dtype=int)
    pos_idx = np.where(y_train == positive_class)[0]
    n_fraud = int((y_train == negative_class).sum())
    n_flip = int(round(pi * n_fraud / (1.0 - pi)))
    if n_flip > len(pos_idx):
        raise ValueError(f"pi={pi} needs {n_flip} flips but only {len(pos_idx)} bogus rows exist")
    flipped = rng.choice(pos_idx, size=n_flip, replace=False)
    y_cont = y_train.copy()
    y_cont[flipped] = negative_class
    return y_cont, np.sort(flipped)


def eval_support_set(multi_gpu, X_support, y_support, X_val_np, y_val,
                     X_test_np, y_test, desc):
    val_proba = multi_gpu.predict_proba_parallel(X_support, y_support, X_val_np,
                                                 desc=f"{desc}: val")
    val_metrics = compute_metrics(y_val, val_proba)
    test_proba = multi_gpu.predict_proba_parallel(X_support, y_support, X_test_np,
                                                  desc=f"{desc}: test")
    test_metrics = compute_metrics(y_test, test_proba,
                                   threshold=val_metrics["optimal_threshold"])
    return val_metrics, test_metrics


def result_row(pi, method, target_size, run, val_metrics, test_metrics, n_pos, n_neg):
    return {
        "pi": pi, "method": method, "target_size": target_size, "run": run,
        "n_pos_support": n_pos, "n_neg_support": n_neg,
        "val_roc_auc": val_metrics["roc_auc"], "val_pr_auc": val_metrics["pr_auc"],
        "val_f1": val_metrics["f1"], "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "test_roc_auc": test_metrics["roc_auc"], "test_pr_auc": test_metrics["pr_auc"],
        "test_f1": test_metrics["f1"], "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "threshold": val_metrics["optimal_threshold"],
    }


def main():
    parser = argparse.ArgumentParser(description="Label-contamination experiment (train-only injection)")
    parser.add_argument("--setting", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--top-features", type=int, default=150)
    parser.add_argument("--pi", type=str, default="0.1,0.2,0.3",
                        help="Comma-separated injected-noise fractions of the fraud pile")
    parser.add_argument("--target-sizes", type=str, default="500,1000")
    parser.add_argument("--K", type=int, default=40, help="Probe count (anchored: M x K/M)")
    parser.add_argument("--support-size", type=int, default=500)
    parser.add_argument("--probe-design", type=str, default="anchored", choices=["random", "anchored"])
    parser.add_argument("--n-anchors", type=int, default=4)
    parser.add_argument("--threshold-method", type=str, default="percentile", choices=["fixed", "percentile"])
    parser.add_argument("--reliable-percentile", type=float, default=30.0)
    parser.add_argument("--suspect-percentile", type=float, default=30.0)
    parser.add_argument("--reliable-threshold", type=float, default=0.75)
    parser.add_argument("--uncertain-threshold", type=float, default=0.45)
    parser.add_argument("--baseline-runs", type=int, default=3,
                        help="Runs for baseline_fully_random AND oracle (paired negative seeds)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pis = [float(s.strip()) for s in args.pi.split(",")]
    target_sizes = [int(s.strip()) for s in args.target_sizes.split(",")]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = PROJECT_ROOT / "exp" / f"{timestamp}_contamination_{args.setting}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    data_config = DataConfig(setting=args.setting, top_features=args.top_features)
    tabicl_config = TabICLConfig(model_path=args.model_path)
    gpu_config = MultiGPUConfig(num_gpus=args.num_gpus,
                                devices=[f"cuda:{i}" for i in range(args.num_gpus)])
    reliability_config = ReliabilityConfig(
        K=args.K, support_set_size=args.support_size,
        probe_design=args.probe_design, n_anchors=args.n_anchors,
        threshold_method=args.threshold_method,
        reliable_threshold=args.reliable_threshold,
        uncertain_threshold=args.uncertain_threshold,
        reliable_percentile=args.reliable_percentile,
        suspect_percentile=args.suspect_percentile,
    )

    loader = DataLoader(data_config)
    data = loader.load_all()
    X_train, y_train, ids_train = data["train"]
    X_val, y_val, _ = data["val"]
    X_test, y_test, _ = data["test"]
    X_train_np = X_train.values
    X_val_np = X_val.values
    X_test_np = X_test.values
    pos_c, neg_c = data_config.positive_class, data_config.negative_class

    logger.info("Data: train=%d (bogus=%d, fraud=%d), val=%d, test=%d",
                len(y_train), int((y_train == pos_c).sum()),
                int((y_train == neg_c).sum()), len(y_val), len(y_test))
    logger.info("pi grid: %s, target sizes: %s", pis, target_sizes)

    multi_gpu = MultiGPUInference(gpu_config, tabicl_config)
    selector_probes = SupportSetSelector(SupportSetConfig(), reliability_config)

    all_results, detection_rows = [], []

    for pi in pis:
        logger.info("=== pi = %.2f ===", pi)
        rng_flip = np.random.default_rng(args.seed + int(round(pi * 1000)))
        y_cont, flipped_idx = contaminate(y_train, pi, pos_c, neg_c, rng_flip)

        pd.DataFrame({data_config.id_col: ids_train.iloc[flipped_idx].values}) \
            .to_csv(output_dir / f"flipped_ids_pi{pi:.2f}.csv", index=False)

        pos_mask = y_cont == pos_c
        neg_mask = y_cont == neg_c
        X_pos_c, y_pos_c = X_train_np[pos_mask], y_cont[pos_mask]
        X_neg_c, y_neg_c = X_train_np[neg_mask], y_cont[neg_mask]
        neg_row_idx = np.where(neg_mask)[0]
        injected = np.isin(neg_row_idx, flipped_idx)  # per fraud-pile row
        ids_neg_c = ids_train[neg_mask].reset_index(drop=True)
        logger.info("Contaminated: flipped %d bogus -> fraud pile %d (injected fraction %.3f), bogus left %d",
                    len(flipped_idx), len(y_neg_c), injected.mean(), len(y_pos_c))

        # Stage A: probes from the CONTAMINATED train (what a real pipeline would see)
        if args.probe_design == "anchored":
            if args.K % args.n_anchors != 0:
                raise ValueError("K must be divisible by n_anchors")
            probe_sets, anchor_ids = selector_probes.build_anchored_support_sets(
                X_train_np, y_cont, M=args.n_anchors,
                draws_per_anchor=args.K // args.n_anchors,
                size=args.support_size, positive_class=pos_c, random_state=args.seed)
        else:
            anchor_ids = None
            probe_sets = selector_probes.build_initial_support_sets(
                X_train_np, y_cont, K=args.K, size=args.support_size,
                positive_class=pos_c, random_state=args.seed)

        # Stage B/C
        predictions = multi_gpu.predict_proba_multi_support(
            support_sets=probe_sets, X_query=X_neg_c,
            desc=f"pi={pi:.2f} Stage B")
        scorer = ReliabilityScorer(reliability_config)
        scores = scorer.compute_scores(predictions)
        classification = scorer.classify_samples(scores)
        suspect = classification["suspect"]

        # Detection vs ground truth (low score = more suspect => invert)
        det_auc = float(roc_auc_score(injected, -scores)) if 0 < injected.sum() < len(injected) else float("nan")
        det = {
            "pi": pi,
            "n_injected": int(injected.sum()),
            "n_fraud_pile": int(len(y_neg_c)),
            "detection_auc": det_auc,
            "suspect_recall_of_injected": float(suspect[injected].mean()) if injected.any() else float("nan"),
            "suspect_precision_for_injected": float(injected[suspect].mean()) if suspect.any() else float("nan"),
            "injected_base_rate": float(injected.mean()),
            "resolved_reliable_threshold": scorer.resolved_thresholds_["reliable_threshold"],
            "resolved_uncertain_threshold": scorer.resolved_thresholds_["uncertain_threshold"],
        }
        detection_rows.append(det)
        logger.info("Detection: AUC=%.3f, suspect recall of injected=%.3f, "
                    "suspect precision for injected=%.3f (base rate %.3f)",
                    det["detection_auc"], det["suspect_recall_of_injected"],
                    det["suspect_precision_for_injected"], det["injected_base_rate"])

        df_sc = pd.DataFrame({
            data_config.id_col: ids_neg_c.values,
            "reliability_score": scores,
            "injected": injected,
            "classification": np.where(classification["reliable"], "reliable",
                                       np.where(classification["uncertain"], "uncertain", "suspect")),
        })
        if anchor_ids is not None:
            vw, vb = variance_decomposition(predictions, anchor_ids)
            df_sc["var_within_anchor"] = vw
            df_sc["var_between_anchor"] = vb
        df_sc.to_csv(output_dir / f"reliability_scores_pi{pi:.2f}.csv", index=False)

        clean_neg_pool = np.where(~injected)[0]  # oracle pool (ground truth)

        for target_size in target_sizes:
            n_half = target_size // 2
            logger.info("-- pi=%.2f target=%d --", pi, target_size)

            # Reference arms: random positives; negative seeds paired between
            # baseline and oracle so their difference isolates pool cleanliness.
            for run in range(args.baseline_runs):
                rng_pos = np.random.default_rng(args.seed + 7919 * (run + 1))
                pos_b = rng_pos.choice(len(X_pos_c), size=min(n_half, len(X_pos_c)), replace=False)
                rng_neg = np.random.default_rng(args.seed + run)
                for arm, pool in (("baseline_fully_random", np.arange(len(X_neg_c))),
                                  ("oracle_clean_negatives", clean_neg_pool)):
                    neg_b = rng_neg.choice(pool, size=min(n_half, len(pool)), replace=False) \
                        if len(pool) > n_half else pool
                    X_sup = np.vstack([X_pos_c[pos_b], X_neg_c[neg_b]])
                    y_sup = np.concatenate([y_pos_c[pos_b], y_neg_c[neg_b]])
                    vm, tm = eval_support_set(multi_gpu, X_sup, y_sup, X_val_np, y_val,
                                              X_test_np, y_test,
                                              desc=f"pi={pi:.2f} t={target_size} {arm} r{run+1}")
                    all_results.append(result_row(pi, arm, target_size, run + 1, vm, tm,
                                                  len(pos_b), len(neg_b)))
                    logger.info("  %s r%d: val_pr_auc=%.4f test_pr_auc=%.4f",
                                arm, run + 1, vm["pr_auc"], tm["pr_auc"])

            # Reliability methods (diversity positives kept — asymmetric design)
            for method in NEG_SAMPLING_METHODS:
                selector = SupportSetSelector(SupportSetConfig(neg_sampling_strategy=method),
                                              reliability_config, neg_sampling_strategy=method)
                (X_sup, y_sup), (p_idx, n_idx) = selector.build_optimized_support_set(
                    X_positives=X_pos_c, y_positives=y_pos_c,
                    X_negatives=X_neg_c, y_negatives=y_neg_c,
                    neg_reliability_scores=scores,
                    neg_prediction_vectors=scorer.prediction_vectors_,
                    neg_classification=classification,
                    n_neg_override=n_half, random_state=args.seed)
                vm, tm = eval_support_set(multi_gpu, X_sup, y_sup, X_val_np, y_val,
                                          X_test_np, y_test,
                                          desc=f"pi={pi:.2f} t={target_size} {method}")
                row = result_row(pi, method, target_size, 1, vm, tm, len(p_idx), len(n_idx))
                row["injected_in_support"] = int(injected[n_idx].sum())
                all_results.append(row)
                logger.info("  %-9s: val_pr_auc=%.4f test_pr_auc=%.4f (injected in support: %d/%d)",
                            method, vm["pr_auc"], tm["pr_auc"],
                            int(injected[n_idx].sum()), len(n_idx))

    df_res = pd.DataFrame(all_results)
    df_res.to_csv(output_dir / "contamination_results.csv", index=False)
    pd.DataFrame(detection_rows).to_csv(output_dir / "detection_metrics.csv", index=False)

    manifest = {
        "artifact_type": "contamination_experiment",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"data_setting": args.setting, "top_features": args.top_features,
                    "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test)},
        "experiment": {
            "pi_grid": pis,
            "pi_definition": "injected fraction of resulting fraud pile: n_flip = pi*n_fraud/(1-pi)",
            "contamination": "train only; random bogus->fraud flips; val/test untouched",
            "natural_noise_note": "natural rho (~0.1-0.2 est.) present in all arms on top of pi",
            "target_sizes": target_sizes,
            "methods": NEG_SAMPLING_METHODS,
            "reference_arms": ["baseline_fully_random", "oracle_clean_negatives"],
            "positive_policy": "methods: diversity-selected; reference arms: random",
            "probe_design": args.probe_design, "K": args.K,
            "n_anchors": args.n_anchors if args.probe_design == "anchored" else None,
            "suspect_percentile": args.suspect_percentile,
            "baseline_runs": args.baseline_runs,
            "seed": args.seed,
        },
    }
    with open(output_dir / "artifact_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved contamination_results.csv, detection_metrics.csv, manifest to %s", output_dir)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
