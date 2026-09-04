"""Trained ML baselines (XGBoost / LightGBM / CatBoost) under the support-set protocol.

Mirrors run_ablation_target_size.py: --target-sizes fixes how many training rows
each model gets (n/2 bogus + n/2 fraud, the support-set analog), and --sampling
picks how those rows are drawn:

  baseline_random  random positives + random negatives from ALL negatives
                   (n runs averaged; the "just train a GBM" reference)
  score_filtered   diversity-selected positives + random negatives drawn from
                   the reliable+uncertain pool (suspects excluded) using the
                   reliability scores of a previous ICL run (--scores-dir)
  score_top        diversity-selected positives + top-n negatives by
                   reliability score (--scores-dir)
  support_csv      exact reuse of a previous run's materialized
                   target_<size>/support_set_<method>.csv (--support-sets +
                   --support-method); byte-identical rows to the ICL arms,
                   including prediction-space selections (e.g. kmeans) that
                   cannot be reproduced from the scores CSV alone

Preprocessing: the shared DataLoader output (top-N features, train-fit ordinal
encoding, float64, numeric NaN preserved). GBDTs need no normalization; the
ordinally-encoded categorical columns are declared as native categoricals per
library (LightGBM categorical_feature; CatBoost cat_features with string cast,
where the -1 unseen/NaN code becomes its own category; XGBoost pandas
category dtype + enable_categorical). Numeric NaN is handled natively by all
three. Support sets are balanced by construction, so no class weighting.

Evaluation matches the ICL runs: threshold calibrated on val (max F1), applied
unchanged to test; positive class = bogus (label 0). Early stopping on val
logloss (also used for threshold calibration — standard, recorded in manifest).
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

from config import DataConfig, PROJECT_ROOT
from data_loader import DataLoader
from support_set_selector import diversity_sample_features
from eval_common import compute_metrics, array_sha256, load_support_set_csv

logger = logging.getLogger(__name__)

ML_MODELS = ("xgboost", "lightgbm", "catboost")
SAMPLING_METHODS = ("baseline_random", "score_filtered", "score_top", "support_csv")

# Fixed hyperparameters (shared shape across libraries): shallow-ish trees,
# small learning rate, generous round budget bounded by early stopping on val.
N_ESTIMATORS = 2000
LEARNING_RATE = 0.05
MAX_DEPTH = 6
EARLY_STOPPING_ROUNDS = 50


def _catboost_frame(df: pd.DataFrame, cat_cols) -> pd.DataFrame:
    # CatBoost wants int/str categoricals without NaN; our ordinal codes are
    # float64 with -1 for NaN/unseen -> cast to int then str ("-1" becomes its
    # own category, which is the correct semantics for unseen/missing).
    df = df.copy()
    for c in cat_cols:
        df[c] = df[c].astype(np.int64).astype(str)
    return df


def _make_xgboost_frame_fn(cat_dtypes: dict):
    # XGBoost's enable_categorical encodes by pandas category codes, so every
    # frame (train/val/test) must share one fixed category set per column —
    # per-frame astype("category") would silently misalign codes.
    def _frame(df: pd.DataFrame, cat_cols) -> pd.DataFrame:
        df = df.copy()
        for c, dtype in cat_dtypes.items():
            df[c] = df[c].astype(np.int64).astype(dtype)
        return df
    return _frame


def make_model(model_name: str, cat_cols, cat_dtypes, seed: int):
    """Return (estimator, fit_fn) where fit_fn(model, Xtr, ytr, Xval, yval)
    trains with early stopping on val and returns nothing. All models expose
    predict_proba over a DataFrame prepared by the matching frame fn."""
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH, tree_method="hist", enable_categorical=True,
            eval_metric="logloss", early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            random_state=seed, n_jobs=-1,
        )
        xgb_frame = _make_xgboost_frame_fn(cat_dtypes)

        def fit(m, Xtr, ytr, Xval, yval):
            m.fit(xgb_frame(Xtr, cat_cols), ytr,
                  eval_set=[(xgb_frame(Xval, cat_cols), yval)], verbose=False)

        return model, fit, xgb_frame

    if model_name == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH, random_state=seed, n_jobs=-1, verbose=-1,
        )

        def fit(m, Xtr, ytr, Xval, yval):
            import inspect
            kwargs = dict(
                eval_metric="binary_logloss",
                categorical_feature=list(cat_cols) if cat_cols else "auto",
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            if "eval_X" in inspect.signature(m.fit).parameters:  # lightgbm >= 4.7
                m.fit(Xtr, ytr, eval_X=Xval, eval_y=yval, **kwargs)
            else:
                m.fit(Xtr, ytr, eval_set=[(Xval, yval)], **kwargs)

        return model, fit, lambda df, cc: df

    if model_name == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=N_ESTIMATORS, learning_rate=LEARNING_RATE,
            depth=MAX_DEPTH, loss_function="Logloss", random_seed=seed,
            cat_features=list(cat_cols) if cat_cols else None, verbose=False,
        )

        def fit(m, Xtr, ytr, Xval, yval):
            m.fit(_catboost_frame(Xtr, cat_cols), ytr,
                  eval_set=(_catboost_frame(Xval, cat_cols), yval),
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)

        return model, fit, _catboost_frame

    raise ValueError(f"Unknown model: {model_name}")


def load_scores(scores_dir: Path, id_col: str) -> pd.DataFrame:
    csv = scores_dir / "reliability_scores.csv"
    if not csv.exists():
        raise SystemExit(f"reliability_scores.csv not found in {scores_dir}")
    df = pd.read_csv(csv)
    needed = {id_col, "reliability_score", "classification"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"{csv} lacks columns: {sorted(missing)}")
    return df


def align_scores_to_negatives(df_scores: pd.DataFrame, ids_neg: pd.Series,
                              id_col: str):
    """Map the scores CSV onto the training negatives by case id.

    Returns (scores: float array, candidate_mask: bool array) aligned to
    ids_neg order. Every negative must be scored (same split, same setting).
    """
    m = df_scores.set_index(id_col)
    missing = ~ids_neg.isin(m.index)
    if missing.any():
        raise SystemExit(
            f"{int(missing.sum())} training negatives have no reliability score "
            "— is --scores-dir from the same --setting?")
    aligned = m.loc[ids_neg]
    scores = aligned["reliability_score"].to_numpy(dtype=float)
    candidate_mask = aligned["classification"].isin(["reliable", "uncertain"]).to_numpy()
    return scores, candidate_mask


def main():
    parser = argparse.ArgumentParser(description="Trained GBDT baselines under the support-set protocol")
    parser.add_argument("--setting", type=str, required=True)
    parser.add_argument("--top-features", type=int, default=150)
    parser.add_argument("--target-sizes", type=str, required=True,
                        help="Comma-separated training-set sizes (support-set analog)")
    parser.add_argument("--models", type=str, default="xgboost,lightgbm,catboost",
                        help=f"Comma-separated subset of {ML_MODELS}")
    parser.add_argument("--sampling", type=str, default="baseline_random,score_filtered,score_top",
                        help=f"Comma-separated subset of {SAMPLING_METHODS}")
    parser.add_argument("--scores-dir", type=str, default=None,
                        help="Exp dir of a previous ICL run holding reliability_scores.csv "
                             "(required for score_filtered / score_top)")
    parser.add_argument("--support-sets", type=str, default=None,
                        help="Exp dir with materialized target_<size>/support_set_<method>.csv "
                             "(required for sampling=support_csv)")
    parser.add_argument("--support-method", type=str, default="kmeans",
                        help="Which support_set_<method>.csv to reuse for sampling=support_csv")
    parser.add_argument("--baseline-runs", type=int, default=5,
                        help="Random-draw repeats for baseline_random and score_filtered")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    target_sizes = [int(s.strip()) for s in args.target_sizes.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    samplings = [s.strip() for s in args.sampling.split(",")]
    for m in models:
        if m not in ML_MODELS:
            raise SystemExit(f"Unknown model '{m}' (choose from {ML_MODELS})")
    for s in samplings:
        if s not in SAMPLING_METHODS:
            raise SystemExit(f"Unknown sampling '{s}' (choose from {SAMPLING_METHODS})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = PROJECT_ROOT / "exp" / f"{timestamp}_ablation_ml_baselines_{args.setting}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    data_config = DataConfig(setting=args.setting, top_features=args.top_features)
    loader = DataLoader(data_config)
    data = loader.load_all()
    X_train, y_train, ids_train = data["train"]
    X_val, y_val, _ = data["val"]
    X_test, y_test, _ = data["test"]

    feature_columns = X_train.columns.tolist()
    cat_cols = [c for c in feature_columns if c in loader.categorical_encodings_]
    # Fixed category universe per column: every train-fit code plus -1
    # (unseen/NaN), shared by all frames so codes align across splits.
    cat_dtypes = {
        c: pd.CategoricalDtype(categories=[-1] + sorted(loader.categorical_encodings_[c].values()))
        for c in cat_cols
    }
    logger.info("Features: %d total, %d categorical (native GBDT handling): %s",
                len(feature_columns), len(cat_cols), cat_cols or "-")

    X_val_df = pd.DataFrame(X_val.values, columns=feature_columns)
    X_test_df = pd.DataFrame(X_test.values, columns=feature_columns)

    pos_mask = y_train == data_config.positive_class
    neg_mask = y_train == data_config.negative_class
    X_pos, y_pos = X_train.values[pos_mask], y_train[pos_mask]
    X_neg, y_neg = X_train.values[neg_mask], y_train[neg_mask]
    ids_neg = ids_train[neg_mask].reset_index(drop=True)

    logger.info("Data: train=%d (pos=%d, neg=%d), val=%d, test=%d",
                len(y_train), len(X_pos), len(X_neg), len(y_val), len(y_test))

    scores = candidate_mask = None
    needs_scores = {"score_filtered", "score_top"} & set(samplings)
    if needs_scores:
        if not args.scores_dir:
            raise SystemExit(f"--scores-dir is required for sampling: {sorted(needs_scores)}")
        df_scores = load_scores(Path(args.scores_dir), data_config.id_col)
        scores, candidate_mask = align_scores_to_negatives(df_scores, ids_neg, data_config.id_col)
        logger.info("Scores: %d negatives aligned, candidate pool (reliable+uncertain)=%d, suspects excluded=%d",
                    len(scores), int(candidate_mask.sum()), int((~candidate_mask).sum()))
    if "support_csv" in samplings and not args.support_sets:
        raise SystemExit("--support-sets is required for sampling=support_csv")

    all_results = []

    def evaluate(model_name, sampling, target_size, run, X_sup, y_sup):
        X_sup_df = pd.DataFrame(X_sup, columns=feature_columns)
        model, fit, frame = make_model(model_name, cat_cols, cat_dtypes,
                                       seed=args.seed + run)
        fit(model, X_sup_df, y_sup, X_val_df, y_val)

        val_proba = model.predict_proba(frame(X_val_df, cat_cols))
        val_metrics = compute_metrics(y_val, val_proba)
        test_proba = model.predict_proba(frame(X_test_df, cat_cols))
        test_metrics = compute_metrics(y_test, test_proba,
                                       threshold=val_metrics["optimal_threshold"])
        all_results.append({
            "target_size": target_size, "model": model_name, "method": sampling,
            "run": run,
            "support_sha256": array_sha256(X_sup, y_sup),
            "n_support": len(y_sup),
            **{f"val_{k}": v for k, v in val_metrics.items()
               if k in ("roc_auc", "pr_auc", "f1", "precision", "recall")},
            **{f"test_{k}": v for k, v in test_metrics.items()
               if k in ("roc_auc", "pr_auc", "f1", "precision", "recall")},
            "threshold": val_metrics["optimal_threshold"],
        })
        logger.info("    %s/%s run %d: val_pr_auc=%.4f test_pr_auc=%.4f",
                    model_name, sampling, run, val_metrics["pr_auc"], test_metrics["pr_auc"])

    for target_size in target_sizes:
        n_half = target_size // 2
        logger.info("=== Target size: %d (%d pos + %d neg) ===", target_size, n_half, n_half)

        # Shared diversity positives (same policy + seed as the ICL ablation).
        pos_idx_shared = diversity_sample_features(X_pos, min(n_half, len(X_pos)),
                                                   random_state=args.seed)
        X_pos_sel, y_pos_sel = X_pos[pos_idx_shared], y_pos[pos_idx_shared]

        for sampling in samplings:
            logger.info("  Sampling: %s", sampling)

            if sampling == "support_csv":
                csv_path = (Path(args.support_sets) / f"target_{target_size}"
                            / f"support_set_{args.support_method}.csv")
                if not csv_path.exists():
                    logger.warning("  %s not found; skipping", csv_path)
                    continue
                X_sup, y_sup = load_support_set_csv(csv_path, feature_columns,
                                                    data_config.label_col)
                for model_name in models:
                    evaluate(model_name, f"support_csv_{args.support_method}",
                             target_size, 1, X_sup, y_sup)
                continue

            if sampling == "score_top":
                cand = np.where(candidate_mask)[0]
                order = np.argsort(scores[cand])[::-1]
                neg_idx = cand[order[:min(n_half, len(cand))]]
                X_sup = np.vstack([X_pos_sel, X_neg[neg_idx]])
                y_sup = np.concatenate([y_pos_sel, y_neg[neg_idx]])
                for model_name in models:
                    evaluate(model_name, sampling, target_size, 1, X_sup, y_sup)
                continue

            # Random-draw arms, n runs. Negative draws share per-run seeds with
            # the ICL ablation's baselines so rows match across scripts.
            for run in range(args.baseline_runs):
                rng_neg = np.random.default_rng(args.seed + run)
                if sampling == "baseline_random":
                    neg_idx = rng_neg.choice(len(X_neg), size=min(n_half, len(X_neg)),
                                             replace=False)
                    rng_pos = np.random.default_rng(args.seed + 7919 * (run + 1))
                    pos_idx = rng_pos.choice(len(X_pos), size=min(n_half, len(X_pos)),
                                             replace=False)
                    X_pos_b, y_pos_b = X_pos[pos_idx], y_pos[pos_idx]
                else:  # score_filtered
                    cand = np.where(candidate_mask)[0]
                    neg_idx = rng_neg.choice(cand, size=min(n_half, len(cand)),
                                             replace=False)
                    X_pos_b, y_pos_b = X_pos_sel, y_pos_sel
                X_sup = np.vstack([X_pos_b, X_neg[neg_idx]])
                y_sup = np.concatenate([y_pos_b, y_neg[neg_idx]])
                for model_name in models:
                    evaluate(model_name, sampling, target_size, run + 1, X_sup, y_sup)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_dir / "ml_baseline_results.csv", index=False)

    summary = df_results.groupby(["target_size", "model", "method"]).agg(
        val_pr_auc_mean=("val_pr_auc", "mean"), val_pr_auc_std=("val_pr_auc", "std"),
        val_roc_auc_mean=("val_roc_auc", "mean"),
        test_pr_auc_mean=("test_pr_auc", "mean"), test_pr_auc_std=("test_pr_auc", "std"),
        test_roc_auc_mean=("test_roc_auc", "mean"),
        test_f1_mean=("test_f1", "mean"),
        test_precision_mean=("test_precision", "mean"),
        test_recall_mean=("test_recall", "mean"),
    ).reset_index()
    summary.to_csv(output_dir / "ml_baseline_summary.csv", index=False)
    logger.info("=== Summary ===")
    for _, r in summary.iterrows():
        logger.info("  target=%d %-9s %-24s val_pr_auc=%.4f test[pr_auc=%.4f roc=%.4f prec=%.4f rec=%.4f]",
                    r["target_size"], r["model"], r["method"],
                    r["val_pr_auc_mean"], r["test_pr_auc_mean"], r["test_roc_auc_mean"],
                    r["test_precision_mean"], r["test_recall_mean"])

    lib_versions = {}
    for lib in ("xgboost", "lightgbm", "catboost"):
        try:
            lib_versions[lib] = __import__(lib).__version__
        except ImportError:
            lib_versions[lib] = None

    manifest = {
        "artifact_type": "ablation_ml_baselines",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"data_setting": args.setting, "n_train": len(y_train),
                    "n_val": len(y_val), "n_test": len(y_test)},
        "input_hashes": {
            "X_train": array_sha256(X_train.values, y_train),
            "X_val": array_sha256(X_val.values, y_val),
            "X_test": array_sha256(X_test.values, y_test),
        },
        "models": {m: lib_versions.get(m) for m in models},
        "hyperparameters": {
            "n_estimators": N_ESTIMATORS, "learning_rate": LEARNING_RATE,
            "max_depth": MAX_DEPTH, "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "early_stopping_split": "val (also used for threshold calibration)",
        },
        "preprocessing": {
            "top_features": args.top_features,
            "categorical_columns_native": cat_cols,
            "categorical_handling": {
                "lightgbm": "categorical_feature indices (ordinal codes; -1 treated as missing)",
                "catboost": "cat_features, codes cast int->str ('-1' = own category)",
                "xgboost": "pandas category dtype + enable_categorical",
            },
            "numeric_nan": "passed through (native handling in all three)",
            "normalization": "none (trees are scale-invariant)",
        },
        "experiment": {
            "target_sizes": target_sizes,
            "sampling_methods": samplings,
            "support_method": args.support_method if "support_csv" in samplings else None,
            "scores_dir": args.scores_dir,
            "support_sets_source": args.support_sets,
            "baseline_runs": args.baseline_runs,
            "seed": args.seed,
        },
    }
    with open(output_dir / "artifact_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Experiment complete. Output: %s", output_dir)


if __name__ == "__main__":
    main()
