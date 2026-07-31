import argparse
import logging
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "supporting_set_constr"))

import torch.multiprocessing as mp

from config import DataConfig, TabICLConfig, MultiGPUConfig, PROJECT_ROOT, DEFAULT_MODEL_PATH, FEATURE_IMPORTANCE_PATH
from data_loader import DataLoader
from support_set_selector import SupportSetSelector
from multi_gpu_inference import MultiGPUInference, estimate_max_query_chunk
from evaluate import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Baseline: random negative sampling without reliability filtering")
    parser.add_argument("--setting", type=str, required=True, help="Data setting folder name (e.g., setting7)")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Local TabICL checkpoint path")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--top-features", type=int, default=150, help="Number of top SHAP features to use")
    parser.add_argument("--target-size", type=int, default=None, help="Final support set size (n/2 pos + n/2 neg). If not set, uses all positives + equal negatives")
    parser.add_argument("--n-runs", type=int, default=5, help="Number of random runs to average over")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = f"{timestamp}_baseline_random_{args.setting}"
    output_dir = PROJECT_ROOT / "exp" / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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
    ids_pos = ids_train[pos_mask]
    X_neg = X_train_np[neg_mask]
    y_neg = y_train[neg_mask]
    ids_neg = ids_train[neg_mask]

    # Determine support set sizes
    if args.target_size is not None:
        n_pos_target = args.target_size // 2
        n_neg_sample = args.target_size // 2
    else:
        n_pos_target = len(X_pos)
        n_neg_sample = len(X_pos)

    # Select positives (diversity sampling if needed)
    if len(X_pos) <= n_pos_target:
        X_pos_sel = X_pos
        y_pos_sel = y_pos
    else:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_pos_target, random_state=42, n_init=3)
        X_pos_float = np.asarray(X_pos, dtype=np.float64)
        km.fit(X_pos_float)
        pos_selected = []
        for c in range(n_pos_target):
            cluster_indices = np.where(km.labels_ == c)[0]
            if len(cluster_indices) == 0:
                continue
            dists = np.linalg.norm(X_pos_float[cluster_indices] - km.cluster_centers_[c], axis=1)
            pos_selected.append(cluster_indices[np.argmin(dists)])
        pos_selected = np.array(pos_selected)
        X_pos_sel = X_pos[pos_selected]
        y_pos_sel = y_pos[pos_selected]

    logging.info("Data: train=%d (pos=%d, neg=%d), val=%d, test=%d, features=%d",
                 len(y_train), len(X_pos), len(X_neg), len(y_val), len(y_test), len(feature_columns))
    logging.info("Support set: %d pos + %d random neg = %d total",
                 len(X_pos_sel), n_neg_sample, len(X_pos_sel) + n_neg_sample)

    multi_gpu = MultiGPUInference(gpu_config, tabicl_config)
    evaluator = Evaluator()

    all_val_metrics = []
    all_test_metrics = []

    for run in range(args.n_runs):
        rng = np.random.default_rng(args.seed + run)
        neg_idx = rng.choice(len(X_neg), size=min(n_neg_sample, len(X_neg)), replace=False)

        X_support = np.vstack([X_pos_sel, X_neg[neg_idx]])
        y_support = np.concatenate([y_pos_sel, y_neg[neg_idx]])

        logging.info("Run %d/%d: evaluating on validation set", run + 1, args.n_runs)
        val_proba = multi_gpu.predict_proba_parallel(
            X_support, y_support, X_val_np,
            desc=f"Run {run + 1}/{args.n_runs}: validation",
        )
        val_metrics = evaluator.compute_all(y_val, val_proba)
        val_metrics["run"] = run + 1
        all_val_metrics.append(val_metrics)
        logging.info("  Val: %s", val_metrics)

        logging.info("Run %d/%d: evaluating on test set", run + 1, args.n_runs)
        test_proba = multi_gpu.predict_proba_parallel(
            X_support, y_support, X_test_np,
            desc=f"Run {run + 1}/{args.n_runs}: test",
        )
        test_metrics = evaluator.compute_all(y_test, test_proba)
        test_metrics["run"] = run + 1
        all_test_metrics.append(test_metrics)
        logging.info("  Test: %s", test_metrics)

    # Save results
    df_val = pd.DataFrame(all_val_metrics)
    df_test = pd.DataFrame(all_test_metrics)
    df_val.to_csv(output_dir / "val_metrics.csv", index=False)
    df_test.to_csv(output_dir / "test_metrics.csv", index=False)

    # Summary stats
    summary = {
        "val_pr_auc_mean": float(df_val["pr_auc"].mean()),
        "val_pr_auc_std": float(df_val["pr_auc"].std()),
        "val_roc_auc_mean": float(df_val["roc_auc"].mean()),
        "val_f1_mean": float(df_val["f1"].mean()),
        "test_pr_auc_mean": float(df_test["pr_auc"].mean()),
        "test_pr_auc_std": float(df_test["pr_auc"].std()),
        "test_roc_auc_mean": float(df_test["roc_auc"].mean()),
        "test_f1_mean": float(df_test["f1"].mean()),
    }

    logging.info("=== Summary (n_runs=%d) ===", args.n_runs)
    logging.info("Val  PR-AUC: %.4f ± %.4f", summary["val_pr_auc_mean"], summary["val_pr_auc_std"])
    logging.info("Test PR-AUC: %.4f ± %.4f", summary["test_pr_auc_mean"], summary["test_pr_auc_std"])

    # Save manifest
    manifest = {
        "artifact_type": "baseline_random_sampling",
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
            "method": "baseline_random_neg_from_all",
            "description": "All positives + random negatives from ALL negatives (no reliability filtering)",
            "target_size": args.target_size,
            "n_positives": int(len(X_pos_sel)),
            "n_negatives": int(n_neg_sample),
            "n_runs": args.n_runs,
            "base_seed": args.seed,
            "top_features": args.top_features,
        },
        "summary": summary,
    }
    with open(output_dir / "artifact_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logging.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
