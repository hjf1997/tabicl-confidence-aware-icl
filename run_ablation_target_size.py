"""Ablation study: scale target-size and compare all neg-sampling strategies.

For each target-size:
1. Select positives (shared across all methods for this target-size)
2. Baseline: random negatives from ALL negatives (n_runs averaged)
3. Run reliability scoring pipeline (Stage A + B + C, shared)
4. Apply each neg-sampling strategy using shared scores
5. Evaluate all variants on validation + test

Cross-model comparison: --model picks the frozen ICL family (tabicl, tabpfn v2
line, tabpfn_v25/v3, tabdpt); --support-sets <prior exp dir> reuses that run's
materialized support sets verbatim (skipping the probe/scoring stages), so all
families are evaluated on byte-identical context — verified by the
support_sha256 column in ablation_results.csv.

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

sys.path.insert(0, str(Path(__file__).resolve().parent / "supporting_set_constr"))

import torch.multiprocessing as mp

from eval_common import compute_metrics, array_sha256, load_support_set_csv
from config import (
    DataConfig, TabICLConfig, MultiGPUConfig, ReliabilityConfig,
    SupportSetConfig, PROJECT_ROOT, DEFAULT_MODEL_PATHS,
)
from data_loader import DataLoader
from multi_gpu_inference import MultiGPUInference, MODEL_FAMILIES
from reliability_scorer import ReliabilityScorer, variance_decomposition
from support_set_selector import SupportSetSelector, diversity_sample_features

logger = logging.getLogger(__name__)

NEG_SAMPLING_METHODS = ["random", "reliable", "informed", "kmeans", "boundary", "hybrid"]


def main():
    parser = argparse.ArgumentParser(description="Ablation: target-size scaling with all neg-sampling methods")
    parser.add_argument("--setting", type=str, required=True, help="Data setting folder name")
    parser.add_argument("--model", type=str, default="tabicl", choices=list(MODEL_FAMILIES),
                        help="In-context model family. tabpfn = v2 line (pip tabpfn==2.2.1; "
                             "vanilla vs Real-TabPFN via --model-path); tabpfn_v25/tabpfn_v3 "
                             "need pip tabpfn>=8.0.0; tabdpt needs pip tabdpt.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Local checkpoint path; defaults to the family's path in config.py "
                             "(families with no default resolve weights via their package/HF cache)")
    parser.add_argument("--tabdpt-context-size", type=int, default=2048,
                        help="(tabdpt) rows retrieved per query; clamped to the support-set size")
    parser.add_argument("--support-sets", type=str, default=None,
                        help="Path to a previous ablation exp dir; reuse its materialized "
                             "target_<size>/support_set_<method>.csv files instead of running "
                             "the probe/scoring stages (cross-model comparison on identical sets)")
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
    parser.add_argument("--score-weights", type=str, default=None,
                        help="learned_weights.json from run_optimize_score.py; replaces the fixed component mixture")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip the two baseline variants (use when only the reliability methods matter, e.g. suspect-percentile sweeps)")
    parser.add_argument("--probe-design", type=str, default="random", choices=["random", "anchored"],
                        help="Stage A probe design: random (both halves resampled) or anchored (M fixed bogus anchors x K/M fraud draws)")
    parser.add_argument("--n-anchors", type=int, default=4, help="(anchored) Number of fixed bogus anchors M; K must be divisible by M")
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
    if args.model_path is None:
        default_path = DEFAULT_MODEL_PATHS.get(args.model)
        args.model_path = str(default_path) if default_path is not None else None
    if args.model_path is not None and not Path(args.model_path).exists():
        raise SystemExit(f"Checkpoint not found: {args.model_path}\n"
                         "Pass --model-path or place the checkpoint at the default location "
                         "(see DEFAULT_MODEL_PATHS in supporting_set_constr/config.py).")
    if args.model_path is None:
        logging.getLogger(__name__).warning(
            "No checkpoint path for family '%s'; the package will resolve its own weights "
            "(requires internet or a pre-populated HF cache).", args.model)
    tabicl_config = TabICLConfig(model_family=args.model, model_path=args.model_path,
                                 tabdpt_context_size=args.tabdpt_context_size)
    support_sets_src = Path(args.support_sets) if args.support_sets else None
    if support_sets_src is not None and not support_sets_src.is_dir():
        raise SystemExit(f"--support-sets dir not found: {support_sets_src}")
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
        score_weights_file=args.score_weights,
        K=args.K,
        support_set_size=args.support_size,
        threshold_method=args.threshold_method,
        reliable_threshold=args.reliable_threshold,
        uncertain_threshold=args.uncertain_threshold,
        reliable_percentile=args.reliable_percentile,
        suspect_percentile=args.suspect_percentile,
    )

    if support_sets_src is not None:
        logger.info("=== Reusing materialized support sets from %s (probe/scoring stages skipped) ===",
                    support_sets_src)
        scorer = None
        reliability_scores = classification = prediction_vectors = None
        n_reliable = n_uncertain = n_suspect = None
    else:
        selector_for_initial = SupportSetSelector(SupportSetConfig(), reliability_config)
        if args.probe_design == "anchored":
            if args.K % args.n_anchors != 0:
                raise ValueError(f"--K ({args.K}) must be divisible by --n-anchors ({args.n_anchors})")
            draws_per_anchor = args.K // args.n_anchors
            support_sets, probe_anchor_ids = selector_for_initial.build_anchored_support_sets(
                X_train_np, y_train,
                M=args.n_anchors,
                draws_per_anchor=draws_per_anchor,
                size=args.support_size,
                positive_class=data_config.positive_class,
                random_state=args.seed,
            )
            logger.info("Built %d anchored probes (M=%d anchors x %d fraud draws)",
                        args.K, args.n_anchors, draws_per_anchor)
        else:
            probe_anchor_ids = None
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
        if probe_anchor_ids is not None:
            var_within, var_between = variance_decomposition(predictions_matrix, probe_anchor_ids)
            df_scores["var_within_anchor"] = var_within
            df_scores["var_between_anchor"] = var_between
        df_scores.to_csv(output_dir / "reliability_scores.csv", index=False)

    # Run ablation across target sizes
    all_results = []

    for target_size in target_sizes:
        n_half = target_size // 2
        logger.info("=== Target size: %d (%d pos + %d neg) ===", target_size, n_half, n_half)

        # Select positives (shared for this target-size)
        pos_idx = diversity_sample_features(X_pos, n_half, random_state=args.seed)
        X_pos_sel = X_pos[pos_idx]
        y_pos_sel = y_pos[pos_idx]
        logger.info("  Selected %d positives", len(pos_idx))

        # Save selected positive ids
        pos_ids_sel = ids_pos.iloc[pos_idx].reset_index(drop=True)

        # --- Baselines: random negatives from ALL negatives, two positive policies ---
        # baseline_fully_random: random positives + random negatives (paper main comparison)
        # baseline_diverse_pos:  shared diversity-selected positives + random negatives
        #                        (ablation; isolates the value of positive selection).
        # Negative draws share the same per-run seed across both variants, so within
        # a run the two baselines differ ONLY in the positive half.
        baseline_variants = () if args.skip_baselines else ("baseline_fully_random", "baseline_diverse_pos")
        for variant in baseline_variants:
            logger.info("  Evaluating %s (%d runs)", variant, args.baseline_runs)
            for run in range(args.baseline_runs):
                rng_neg = np.random.default_rng(args.seed + run)
                neg_idx_baseline = rng_neg.choice(len(X_neg), size=min(n_half, len(X_neg)), replace=False)

                if variant == "baseline_fully_random":
                    rng_pos = np.random.default_rng(args.seed + 7919 * (run + 1))
                    pos_idx_b = rng_pos.choice(len(X_pos), size=min(n_half, len(X_pos)), replace=False)
                    X_pos_b, y_pos_b = X_pos[pos_idx_b], y_pos[pos_idx_b]
                else:
                    X_pos_b, y_pos_b = X_pos_sel, y_pos_sel

                X_support = np.vstack([X_pos_b, X_neg[neg_idx_baseline]])
                y_support = np.concatenate([y_pos_b, y_neg[neg_idx_baseline]])

                val_proba = multi_gpu.predict_proba_parallel(
                    X_support, y_support, X_val_np,
                    desc=f"target={target_size} {variant} run {run+1}: val",
                )
                val_metrics = compute_metrics(y_val, val_proba)

                test_proba = multi_gpu.predict_proba_parallel(
                    X_support, y_support, X_test_np,
                    desc=f"target={target_size} {variant} run {run+1}: test",
                )
                test_metrics = compute_metrics(y_test, test_proba, threshold=val_metrics["optimal_threshold"])

                all_results.append({
                    "target_size": target_size,
                    "method": variant,
                    "run": run + 1,
                    "support_sha256": array_sha256(X_support, y_support),
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
                logger.info("    %s run %d: val_pr_auc=%.4f, test_pr_auc=%.4f",
                            variant, run + 1, val_metrics["pr_auc"], test_metrics["pr_auc"])

        # --- Neg-sampling strategies (using shared reliability scores) ---
        for method in NEG_SAMPLING_METHODS:
            logger.info("  Evaluating method: %s", method)
            if support_sets_src is not None:
                csv_path = support_sets_src / f"target_{target_size}" / f"support_set_{method}.csv"
                if not csv_path.exists():
                    logger.warning("  %s not found in source dir; skipping method", csv_path)
                    continue
                X_support, y_support = load_support_set_csv(
                    csv_path, feature_columns, data_config.label_col)
                sel_pos_idx = sel_neg_idx = None
            else:
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
                "support_sha256": array_sha256(X_support, y_support),
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

            # Save support set for this method + target-size (fresh-selection
            # mode only; in reuse mode the source dir already holds the files
            # and the support_sha256 column ties results to them).
            if support_sets_src is None:
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
    model_type_labels = {
        "tabicl": "TabICLClassifier",
        "tabpfn": "TabPFNClassifier (v2 line, pip tabpfn==2.2.1)",
        "tabpfn_v25": "TabPFNClassifier (v2.5, pip tabpfn>=8.0.0)",
        "tabpfn_v3": "TabPFNClassifier (v3, pip tabpfn>=8.0.0)",
        "tabdpt": "TabDPTClassifier (via _TabDPTAdapter)",
    }
    manifest = {
        "artifact_type": "ablation_target_size",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "data_setting": args.setting,
            "n_train": len(y_train),
            "n_val": len(y_val),
            "n_test": len(y_test),
        },
        "input_hashes": {
            "X_train": array_sha256(X_train_np, y_train),
            "X_val": array_sha256(X_val_np, y_val),
            "X_test": array_sha256(X_test_np, y_test),
        },
        "model": {
            "model_type": model_type_labels[args.model],
            "model_family": args.model,
            "model_path": tabicl_config.model_path,
            "n_estimators": tabicl_config.n_estimators,
            "kv_cache": tabicl_config.kv_cache,
            "tabdpt_context_size": args.tabdpt_context_size if args.model == "tabdpt" else None,
        },
        "experiment": {
            "target_sizes": target_sizes,
            "neg_sampling_methods": NEG_SAMPLING_METHODS,
            "baseline_runs": args.baseline_runs,
            "baselines_skipped": args.skip_baselines,
            "baseline_variants": None if args.skip_baselines else {
                "baseline_fully_random": "random positives + random negatives (paper main comparison)",
                "baseline_diverse_pos": "diversity-selected positives + random negatives (ablation; was 'baseline_random' in runs before this change)",
            },
            "support_sets_source": str(support_sets_src) if support_sets_src is not None else None,
            "K": args.K,
            "probe_design": args.probe_design,
            "n_anchors": args.n_anchors if args.probe_design == "anchored" else None,
            "score_weights_file": args.score_weights,
            "support_size": args.support_size,
            "threshold_method": args.threshold_method,
            "reliable_percentile": args.reliable_percentile,
            "suspect_percentile": args.suspect_percentile,
            "resolved_thresholds": scorer.resolved_thresholds_ if scorer is not None else None,
            "n_reliable": n_reliable,
            "n_uncertain": n_uncertain,
            "n_suspect": n_suspect,
            "seed": args.seed,
            "top_features": args.top_features,
            "ordinal_encoded_columns": {
                col: len(mapping) for col, mapping in loader.categorical_encodings_.items()
            },
        },
    }
    with open(output_dir / "artifact_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Experiment complete. Output: %s", output_dir)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
