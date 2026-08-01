"""Ablation study: scale target-size and compare all neg-sampling strategies.

For each target-size:
1. Select positives (shared across all methods for this target-size)
2. Baseline: random negatives from ALL negatives (n_runs averaged)
3. Run reliability scoring pipeline (Stage A + B + C, shared)
4. Apply each neg-sampling strategy using shared scores
5. Evaluate all variants on validation + test

Output: single CSV with columns [target_size, method, split, pr_auc, roc_auc, f1, ...]
"""
import argparse
import logging
import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from sklearn.cluster import KMeans

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
from reliability_scorer import ReliabilityScorer
from support_set_selector import SupportSetSelector

logger = logging.getLogger(__name__)

NEG_SAMPLING_METHODS = ["random", "reliable", "informed", "kmeans", "boundary", "hybrid"]


def compute_metrics(y_true, y_proba, threshold=None):
    """Compute metrics. Positive class = bogus (label 0).

    If threshold is None, finds optimal threshold from the data (use for val).
    If threshold is provided, applies it directly (use for test with val-determined threshold).

    Returns dict with roc_auc, pr_auc, f1, precision, recall, optimal_threshold, n_predicted_bogus.
    """
    pos_proba = y_proba[:, 0]  # P(bogus)
    y_binary = (y_true == 0).astype(int)  # 1 where bogus

    roc_auc = roc_auc_score(y_binary, pos_proba)
    pr_auc = average_precision_score(y_binary, pos_proba)

    if threshold is None:
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_binary, pos_proba)
        f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        threshold = float(thresholds[optimal_idx]) if optimal_idx < len(thresholds) else 0.5

    y_pred = (pos_proba >= threshold).astype(int)
    f1 = float(f1_score(y_binary, y_pred))
    prec = float(precision_score(y_binary, y_pred, zero_division=0))
    rec = float(recall_score(y_binary, y_pred, zero_division=0))

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "optimal_threshold": float(threshold),
        "n_predicted_bogus": int(y_pred.sum()),
    }


def select_positives_diversity(X_pos, n_pos, random_state=42):
    """Select n_pos diverse positives via KMeans. Returns indices."""
    if len(X_pos) <= n_pos:
        return np.arange(len(X_pos))
    X_float = np.asarray(X_pos, dtype=np.float64)
    km = KMeans(n_clusters=n_pos, random_state=random_state, n_init=3)
    km.fit(X_float)
    selected = []
    for c in range(n_pos):
        cluster_indices = np.where(km.labels_ == c)[0]
        if len(cluster_indices) == 0:
            continue
        dists = np.linalg.norm(X_float[cluster_indices] - km.cluster_centers_[c], axis=1)
        selected.append(cluster_indices[np.argmin(dists)])
    return np.array(selected)


def main():
    parser = argparse.ArgumentParser(description="Ablation: target-size scaling with all neg-sampling methods")
    parser.add_argument("--setting", type=str, required=True, help="Data setting folder name")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--top-features", type=int, default=150)
    parser.add_argument("--K", type=int, default=20, help="Number of scoring support sets")
    parser.add_argument("--support-size", type=int, default=500, help="Size of each scoring support set")
    parser.add_argument("--target-sizes", type=str, required=True,
                        help="Comma-separated target sizes (e.g., 200,500,1000,2000,5000)")
    parser.add_argument("--threshold-method", type=str, default="percentile", choices=["fixed", "percentile"])
    parser.add_argument("--reliable-threshold", type=float, default=0.75)
    parser.add_argument("--uncertain-threshold", type=float, default=0.45)
    parser.add_argument("--reliable-percentile", type=float, default=30.0)
    parser.add_argument("--suspect-percentile", type=float, default=30.0)
    parser.add_argument("--baseline-runs", type=int, default=5, help="Number of baseline random runs per target-size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    target_sizes = [int(s.strip()) for s in args.target_sizes.split(",")]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = f"{timestamp}_ablation_target_size_{args.setting}"
    output_dir = PROJECT_ROOT / "exp" / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load data
    data_config = DataConfig(setting=args.setting, top_features=args.top_features)
    tabicl_config = TabICLConfig(model_path=args.model_path)
    gpu_config = MultiGPUConfig(
        num_gpus=args.num_gpus,
        devices=[f"cuda:{i}" for i in range(args.num_gpus)],
    )

    loader = DataLoader(data_config)
    data = loader.load_all()
    X_train, y_train, ids_train = data["train"]
    X_val, y_val, _ = data["val"]
    X_test, y_test, _ = data["test"]

    feature_columns = X_train.columns.tolist()
    X_train_np = X_train.values
    X_val_np = X_val.values
    X_test_np = X_test.values

    pos_mask = y_train == data_config.positive_class
    neg_mask = y_train == data_config.negative_class
    X_pos = X_train_np[pos_mask]
    y_pos = y_train[pos_mask]
    ids_pos = ids_train[pos_mask].reset_index(drop=True)
    X_neg = X_train_np[neg_mask]
    y_neg = y_train[neg_mask]
    ids_neg = ids_train[neg_mask].reset_index(drop=True)

    logger.info("Data: train=%d (pos=%d, neg=%d), val=%d, test=%d, features=%d",
                len(y_train), len(X_pos), len(X_neg), len(y_val), len(y_test), len(feature_columns))
    logger.info("Target sizes: %s", target_sizes)

    multi_gpu = MultiGPUInference(gpu_config, tabicl_config)

    # Stage A + B: compute reliability scores (shared across all target sizes)
    logger.info("=== Stage A: Building %d scoring support sets (size=%d) ===", args.K, args.support_size)
    reliability_config = ReliabilityConfig(
        K=args.K,
        support_set_size=args.support_size,
        threshold_method=args.threshold_method,
        reliable_threshold=args.reliable_threshold,
        uncertain_threshold=args.uncertain_threshold,
        reliable_percentile=args.reliable_percentile,
        suspect_percentile=args.suspect_percentile,
    )

    selector_for_initial = SupportSetSelector(SupportSetConfig(), reliability_config)
    support_sets = selector_for_initial.build_initial_support_sets(
        X_train_np, y_train,
        K=args.K,
        size=args.support_size,
        positive_class=data_config.positive_class,
    )

    logger.info("=== Stage B: Scoring %d negatives with %d support sets ===", len(X_neg), args.K)
    predictions_matrix = multi_gpu.predict_proba_multi_support(
        support_sets=support_sets,
        X_query=X_neg,
        desc="Stage B: scoring negatives",
    )

    scorer = ReliabilityScorer(reliability_config)
    reliability_scores = scorer.compute_scores(predictions_matrix)
    classification = scorer.classify_samples(reliability_scores)
    prediction_vectors = scorer.prediction_vectors_

    n_reliable = int(classification["reliable"].sum())
    n_uncertain = int(classification["uncertain"].sum())
    n_suspect = int(classification["suspect"].sum())
    resolved = scorer.resolved_thresholds_
    logger.info("Stage C: Thresholds (reliable=%.4f, suspect=%.4f) → Reliable=%d, Uncertain=%d, Suspect=%d",
                resolved["reliable_threshold"], resolved["uncertain_threshold"],
                n_reliable, n_uncertain, n_suspect)

    # Save reliability scores
    df_scores = pd.DataFrame({
        data_config.id_col: ids_neg.values,
        "reliability_score": reliability_scores,
        "mu": scorer.components_["mu"],
        "sigma": scorer.components_["sigma"],
        "entropy": scorer.components_["H"],
        "agreement": scorer.components_["A"],
        "density": scorer.components_["D"],
        "classification": np.where(
            classification["reliable"], "reliable",
            np.where(classification["uncertain"], "uncertain", "suspect")
        ),
    })
    df_scores.to_csv(output_dir / "reliability_scores.csv", index=False)

    # Run ablation across target sizes
    all_results = []

    for target_size in target_sizes:
        n_half = target_size // 2
        logger.info("=== Target size: %d (%d pos + %d neg) ===", target_size, n_half, n_half)

        # Select positives (shared for this target-size)
        pos_idx = select_positives_diversity(X_pos, n_half, random_state=args.seed)
        X_pos_sel = X_pos[pos_idx]
        y_pos_sel = y_pos[pos_idx]
        logger.info("  Selected %d positives", len(pos_idx))

        # Save selected positive ids
        pos_ids_sel = ids_pos.iloc[pos_idx].reset_index(drop=True)

        # --- Baseline: random from ALL negatives ---
        logger.info("  Evaluating baseline (random from all negatives, %d runs)", args.baseline_runs)
        for run in range(args.baseline_runs):
            rng = np.random.default_rng(args.seed + run)
            neg_idx_baseline = rng.choice(len(X_neg), size=min(n_half, len(X_neg)), replace=False)
            X_support = np.vstack([X_pos_sel, X_neg[neg_idx_baseline]])
            y_support = np.concatenate([y_pos_sel, y_neg[neg_idx_baseline]])

            val_proba = multi_gpu.predict_proba_parallel(
                X_support, y_support, X_val_np,
                desc=f"target={target_size} baseline run {run+1}: val",
            )
            val_metrics = compute_metrics(y_val, val_proba)

            test_proba = multi_gpu.predict_proba_parallel(
                X_support, y_support, X_test_np,
                desc=f"target={target_size} baseline run {run+1}: test",
            )
            test_metrics = compute_metrics(y_test, test_proba, threshold=val_metrics["optimal_threshold"])

            all_results.append({
                "target_size": target_size,
                "method": "baseline_random",
                "run": run + 1,
                "val_roc_auc": val_metrics["roc_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_f1": test_metrics["f1"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "threshold": val_metrics["optimal_threshold"],
            })
            logger.info("    Baseline run %d: val_pr_auc=%.4f, test_pr_auc=%.4f",
                        run + 1, val_metrics["pr_auc"], test_metrics["pr_auc"])

        # --- Neg-sampling strategies (using shared reliability scores) ---
        for method in NEG_SAMPLING_METHODS:
            logger.info("  Evaluating method: %s", method)
            support_config = SupportSetConfig(neg_sampling_strategy=method)
            selector = SupportSetSelector(support_config, reliability_config, neg_sampling_strategy=method)

            (X_support, y_support), (sel_pos_idx, sel_neg_idx) = selector.build_optimized_support_set(
                X_positives=X_pos,
                y_positives=y_pos,
                X_negatives=X_neg,
                y_negatives=y_neg,
                neg_reliability_scores=reliability_scores,
                neg_prediction_vectors=prediction_vectors,
                neg_classification=classification,
                n_neg_override=n_half,
                random_state=args.seed,
            )

            val_proba = multi_gpu.predict_proba_parallel(
                X_support, y_support, X_val_np,
                desc=f"target={target_size} {method}: val",
            )
            val_metrics = compute_metrics(y_val, val_proba)

            test_proba = multi_gpu.predict_proba_parallel(
                X_support, y_support, X_test_np,
                desc=f"target={target_size} {method}: test",
            )
            test_metrics = compute_metrics(y_test, test_proba, threshold=val_metrics["optimal_threshold"])

            all_results.append({
                "target_size": target_size,
                "method": method,
                "run": 1,
                "val_roc_auc": val_metrics["roc_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_f1": test_metrics["f1"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "threshold": val_metrics["optimal_threshold"],
            })
            logger.info("    %s: val_pr_auc=%.4f, test_pr_auc=%.4f",
                        method, val_metrics["pr_auc"], test_metrics["pr_auc"])

            # Save support set for this method + target-size
            method_dir = output_dir / f"target_{target_size}"
            method_dir.mkdir(parents=True, exist_ok=True)
            neg_ids_sel = ids_neg.iloc[sel_neg_idx].reset_index(drop=True)
            support_ids = pd.concat([
                ids_pos.iloc[sel_pos_idx].reset_index(drop=True),
                neg_ids_sel,
            ], ignore_index=True)
            df_support = pd.DataFrame(X_support, columns=feature_columns)
            df_support.insert(0, data_config.id_col, support_ids.values)
            df_support[data_config.label_col] = y_support
            df_support.to_csv(method_dir / f"support_set_{method}.csv", index=False)

    # Save all results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_dir / "ablation_results.csv", index=False)
    logger.info("All results saved to %s", output_dir / "ablation_results.csv")

    # Print summary table
    logger.info("=== Summary ===")
    summary = df_results.groupby(["target_size", "method"]).agg(
        val_roc_auc_mean=("val_roc_auc", "mean"),
        val_pr_auc_mean=("val_pr_auc", "mean"),
        val_pr_auc_std=("val_pr_auc", "std"),
        val_f1_mean=("val_f1", "mean"),
        val_precision_mean=("val_precision", "mean"),
        val_recall_mean=("val_recall", "mean"),
        test_roc_auc_mean=("test_roc_auc", "mean"),
        test_pr_auc_mean=("test_pr_auc", "mean"),
        test_pr_auc_std=("test_pr_auc", "std"),
        test_f1_mean=("test_f1", "mean"),
        test_precision_mean=("test_precision", "mean"),
        test_recall_mean=("test_recall", "mean"),
    ).reset_index()
    summary.to_csv(output_dir / "ablation_summary.csv", index=False)
    for _, row in summary.iterrows():
        std_val = row["val_pr_auc_std"] if not np.isnan(row["val_pr_auc_std"]) else 0
        std_test = row["test_pr_auc_std"] if not np.isnan(row["test_pr_auc_std"]) else 0
        logger.info("  target=%d  method=%-16s  val_pr_auc=%.4f±%.4f  test[pr_auc=%.4f±%.4f  f1=%.4f  prec=%.4f  rec=%.4f]",
                    row["target_size"], row["method"],
                    row["val_pr_auc_mean"], std_val,
                    row["test_pr_auc_mean"], std_test,
                    row["test_f1_mean"], row["test_precision_mean"], row["test_recall_mean"])

    # Save manifest
    manifest = {
        "artifact_type": "ablation_target_size",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "data_setting": args.setting,
            "n_train": len(y_train),
            "n_val": len(y_val),
            "n_test": len(y_test),
        },
        "model": {
            "model_type": "TabICLClassifier",
            "model_path": tabicl_config.model_path,
            "n_estimators": tabicl_config.n_estimators,
            "kv_cache": tabicl_config.kv_cache,
        },
        "experiment": {
            "target_sizes": target_sizes,
            "neg_sampling_methods": NEG_SAMPLING_METHODS,
            "baseline_runs": args.baseline_runs,
            "K": args.K,
            "support_size": args.support_size,
            "threshold_method": args.threshold_method,
            "reliable_percentile": args.reliable_percentile,
            "suspect_percentile": args.suspect_percentile,
            "resolved_thresholds": scorer.resolved_thresholds_,
            "n_reliable": n_reliable,
            "n_uncertain": n_uncertain,
            "n_suspect": n_suspect,
            "seed": args.seed,
            "top_features": args.top_features,
        },
    }
    with open(output_dir / "artifact_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Experiment complete. Output: %s", output_dir)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
