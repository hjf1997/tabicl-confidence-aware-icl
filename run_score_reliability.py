"""Score reliability for ALL cases in train + val + test with one joint run.

Design (agreed in discussion):
- Probe support sets are built from TRAIN only (a probe needs labeled context;
  test labels must not enter the context).
- Every row of train, val, and test is scored through the same K probes in a
  single joint run, so neighborhood agreement (A) and density (D) are computed
  over the union population with one normalization — no cross-cohort scale
  mismatch.
- Reliable/uncertain/suspect thresholds are resolved on the FRAUD-LABELED
  subset only (that is the population the semantics are defined on; trusted
  bogus rows would shift the percentiles mechanically), then applied to all
  rows.
- For fraud-labeled rows the score reads as "reliability of the fraud label";
  for bogus rows (trusted labels) it reads as "how fraud-like the case
  behaves" — a model sanity check, not a label check.

Output:
  exp/YYYYMMDD_HHMM_reliability_scores_settingX/
    reliability_scores_all.csv   ar_case_no, split, label, components,
                                 reliability_score, classification
    prediction_vectors.csv       ar_case_no + the K raw P(fraud) columns
    artifact_manifest.json
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

from config import (
    DataConfig, TabICLConfig, MultiGPUConfig, ReliabilityConfig,
    SupportSetConfig, PROJECT_ROOT, DEFAULT_MODEL_PATH,
)
from data_loader import DataLoader
from multi_gpu_inference import MultiGPUInference
from reliability_scorer import ReliabilityScorer
from support_set_selector import SupportSetSelector

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Joint reliability scoring of train + val + test")
    parser.add_argument("--setting", type=str, required=True, help="Data setting folder name (e.g., setting7)")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Local TabICL checkpoint path")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--top-features", type=int, default=150, help="Number of top SHAP features to use")
    parser.add_argument("--K", type=int, default=20, help="Number of probe support sets")
    parser.add_argument("--support-size", type=int, default=500, help="Rows per probe support set")
    parser.add_argument("--threshold-method", type=str, default="percentile", choices=["fixed", "percentile"])
    parser.add_argument("--reliable-threshold", type=float, default=0.75, help="(fixed) reliable cutoff")
    parser.add_argument("--uncertain-threshold", type=float, default=0.45, help="(fixed) suspect cutoff")
    parser.add_argument("--reliable-percentile", type=float, default=30.0, help="(percentile) top X%% of fraud-labeled rows = reliable")
    parser.add_argument("--suspect-percentile", type=float, default=30.0, help="(percentile) bottom X%% of fraud-labeled rows = suspect")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = PROJECT_ROOT / "exp" / f"{timestamp}_reliability_scores_{args.setting}"
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
    reliability_config = ReliabilityConfig(
        K=args.K,
        support_set_size=args.support_size,
        threshold_method=args.threshold_method,
        reliable_threshold=args.reliable_threshold,
        uncertain_threshold=args.uncertain_threshold,
        reliable_percentile=args.reliable_percentile,
        suspect_percentile=args.suspect_percentile,
    )

    loader = DataLoader(data_config)
    data = loader.load_all()
    X_train, y_train, ids_train = data["train"]

    # Union query matrix over all splits, with split provenance.
    parts, split_names = [], []
    y_parts, id_parts = [], []
    for name in ("train", "val", "test"):
        X, y, ids = data[name]
        parts.append(X.values)
        y_parts.append(y)
        id_parts.append(ids)
        split_names.extend([name] * len(y))
    X_all = np.vstack(parts)
    y_all = np.concatenate(y_parts)
    ids_all = pd.concat(id_parts, ignore_index=True)
    split_all = np.array(split_names)

    logger.info("Data: train=%d, val=%d, test=%d → scoring %d rows total, features=%d",
                len(y_parts[0]), len(y_parts[1]), len(y_parts[2]), len(y_all), X_all.shape[1])

    # Probes from train only.
    selector = SupportSetSelector(SupportSetConfig(), reliability_config)
    probe_sets = selector.build_initial_support_sets(
        X_train.values, y_train,
        K=args.K,
        size=args.support_size,
        positive_class=data_config.positive_class,
        random_state=args.seed,
    )
    logger.info("Built %d probe support sets (size=%d) from train", args.K, args.support_size)

    multi_gpu = MultiGPUInference(gpu_config, tabicl_config)
    predictions = multi_gpu.predict_proba_multi_support(
        support_sets=probe_sets,
        X_query=X_all,
        desc="Scoring all splits",
    )

    # Scores and components over the union population (one kNN pool, one normalization).
    scorer = ReliabilityScorer(reliability_config)
    scores = scorer.compute_scores(predictions)

    # Resolve thresholds on the fraud-labeled subset only, apply to all rows.
    fraud_mask = y_all == data_config.negative_class
    scorer.classify_samples(scores[fraud_mask])
    resolved = scorer.resolved_thresholds_
    reliable_t = resolved["reliable_threshold"]
    suspect_t = resolved["uncertain_threshold"]
    logger.info("Thresholds resolved on %d fraud-labeled rows: reliable=%.4f, suspect=%.4f",
                int(fraud_mask.sum()), reliable_t, suspect_t)

    classification = np.where(
        scores >= reliable_t, "reliable",
        np.where(scores < suspect_t, "suspect", "uncertain"),
    )

    for name in ("train", "val", "test"):
        m = (split_all == name) & fraud_mask
        n_rel = int((classification[m] == "reliable").sum())
        n_unc = int((classification[m] == "uncertain").sum())
        n_sus = int((classification[m] == "suspect").sum())
        logger.info("%s fraud-labeled: reliable=%d, uncertain=%d, suspect=%d", name, n_rel, n_unc, n_sus)

    df = pd.DataFrame({
        data_config.id_col: ids_all.values,
        "split": split_all,
        data_config.label_col: y_all,
        "reliability_score": scores,
        "mu": scorer.components_["mu"],
        "sigma": scorer.components_["sigma"],
        "entropy": scorer.components_["H"],
        "agreement": scorer.components_["A"],
        "density": scorer.components_["D"],
        "classification": classification,
    })
    df.to_csv(output_dir / "reliability_scores_all.csv", index=False)
    logger.info("Scores saved to %s", output_dir / "reliability_scores_all.csv")

    df_pred = pd.DataFrame(predictions, columns=[f"p_fraud_probe_{k}" for k in range(args.K)])
    df_pred.insert(0, data_config.id_col, ids_all.values)
    df_pred.insert(1, "split", split_all)
    df_pred.to_csv(output_dir / "prediction_vectors.csv", index=False)
    logger.info("Prediction vectors saved to %s", output_dir / "prediction_vectors.csv")

    manifest = {
        "artifact_type": "reliability_scores_all_splits",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "data_setting": args.setting,
            "n_train": int(len(y_parts[0])),
            "n_val": int(len(y_parts[1])),
            "n_test": int(len(y_parts[2])),
            "top_features": args.top_features,
            "ordinal_encoded_columns": {
                col: len(mapping) for col, mapping in loader.categorical_encodings_.items()
            },
        },
        "model": {
            "model_type": "TabICLClassifier",
            "model_path": tabicl_config.model_path,
            "n_estimators": tabicl_config.n_estimators,
            "kv_cache": tabicl_config.kv_cache,
        },
        "scoring": {
            "probe_source": "train_only",
            "K": args.K,
            "support_set_size": args.support_size,
            "seed": args.seed,
            "knn_population": "union_of_all_splits",
            "threshold_method": args.threshold_method,
            "threshold_population": "fraud_labeled_rows_all_splits",
            "reliable_percentile": args.reliable_percentile,
            "suspect_percentile": args.suspect_percentile,
            "resolved_reliable_threshold": reliable_t,
            "resolved_uncertain_threshold": suspect_t,
            "weights": {
                "w_mean_prob": reliability_config.w_mean_prob,
                "w_stability": reliability_config.w_stability,
                "w_entropy": reliability_config.w_entropy,
                "w_agreement": reliability_config.w_agreement,
                "w_density": reliability_config.w_density,
            },
            "n_neighbors": reliability_config.n_neighbors,
        },
    }
    with open(output_dir / "artifact_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved to %s", output_dir / "artifact_manifest.json")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
