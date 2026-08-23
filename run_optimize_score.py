"""Optimize reliability-score hyperparameters on contamination ground truth.

Consumes two run_score_reliability.py output dirs produced on contaminated
settings (see make_contaminated_setting.py) plus their ground-truth flip
lists. Everything here is CPU-side column arithmetic on the saved prediction
vectors — no GPU.

Optimized:
- component weights (mu, sigma, entropy, agreement, density): class-balanced
  logistic regression predicting the injected flag from z-scored components
- n_neighbors for agreement/density: grid search, components recomputed from
  the prediction vectors at each value

Validation is cross-pi in both directions (fit on run A, detect on run B, and
the reverse) against fixed baselines: the current composite (default weights,
k=15), mu alone, and each component solo. A "with_var" variant (adding the
anchored within/between-anchor variances) is reported for information but NOT
exported — the exported weights use only the five portable components so they
work under any probe design.

Outputs (exp/YYYYMMDD_HHMM_score_optimization/):
- optimization_results.csv    grid x direction AUC table + baselines
- learned_weights.json        weights for z-scored components, higher = more
                              reliable; consumed via --score-weights /
                              ReliabilityConfig.score_weights_file
- rescored_reliability_scores_all.csv   (if --natural-scores-dir) the natural
  run re-scored with the learned weights, standard format, for Level-2 checks
- artifact_manifest.json

Fitting cohort: train-split fraud-labeled rows of each contaminated setting,
with agreement/density recomputed over that cohort (mirroring how the pipeline
scores its selection pool). Standardization is within-cohort at both fit and
apply time; percentile thresholds downstream consume ranks only.
"""
import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent / "supporting_set_constr"))

from config import ReliabilityConfig, PROJECT_ROOT
from reliability_scorer import ReliabilityScorer, variance_decomposition

logger = logging.getLogger(__name__)

BASE_FEATURES = ["mu", "sigma", "entropy", "agreement", "density"]
VAR_FEATURES = ["var_within", "var_between"]


def parse_anchor_ids(columns):
    """Anchor index per probe column, or None for the random design."""
    ids = []
    for c in columns:
        m = re.match(r"p_fraud_anchor(\d+)_draw\d+$", c)
        if not m:
            return None
        ids.append(int(m.group(1)))
    return ids


def load_run(scores_dir: Path, flip_file: Path, id_col="ar_case_no", label_col="label"):
    """Return (preds, injected, anchor_ids) for train-split fraud-labeled rows."""
    scores = pd.read_csv(scores_dir / "reliability_scores_all.csv")
    vectors = pd.read_csv(scores_dir / "prediction_vectors.csv")
    probe_cols = [c for c in vectors.columns if c.startswith("p_fraud_")]

    df = scores[[id_col, "split", label_col]].join(vectors[probe_cols])
    if len(scores) != len(vectors) or (scores[id_col].values != vectors[id_col].values).any():
        raise ValueError("scores and prediction_vectors rows are not aligned")

    mask = (df["split"] == "train") & (df[label_col] == 1)
    sub = df[mask]
    flipped = set(pd.read_csv(flip_file)[id_col])
    injected = sub[id_col].isin(flipped).values
    preds = sub[probe_cols].values.astype(float)
    return preds, injected, parse_anchor_ids(probe_cols)


def compute_features(preds: np.ndarray, n_neighbors: int, anchor_ids):
    """Feature matrix (cohort = the given rows) and the composite score at the
    same n_neighbors with default weights (the baseline when n_neighbors=15)."""
    scorer = ReliabilityScorer(ReliabilityConfig(n_neighbors=n_neighbors))
    composite = scorer.compute_scores(preds)
    c = scorer.components_
    feats = {"mu": c["mu"], "sigma": c["sigma"], "entropy": c["H"],
             "agreement": c["A"], "density": c["D"]}
    if anchor_ids is not None:
        vw, vb = variance_decomposition(preds, anchor_ids)
        feats["var_within"], feats["var_between"] = vw, vb
    return feats, composite


def zscore_matrix(feats: dict, names) -> np.ndarray:
    cols = []
    for n in names:
        x = feats[n]
        std = x.std()
        cols.append((x - x.mean()) / (std if std > 1e-8 else 1.0))
    return np.column_stack(cols)


def fit_lr(X, y, seed):
    return LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                              random_state=seed).fit(X, y)


def detection_auc_from_reliable_score(injected, reliable_score):
    """Reliable-oriented score (higher = reliable): invert for detection AUC."""
    return float(roc_auc_score(injected, -reliable_score))


def main():
    parser = argparse.ArgumentParser(description="Optimize reliability-score weights and n_neighbors")
    parser.add_argument("--scores-dirs", type=str, required=True,
                        help="Two run_score_reliability output dirs (contaminated settings), comma-separated")
    parser.add_argument("--flip-files", type=str, required=True,
                        help="Matching flipped_ids.csv paths, comma-separated, same order")
    parser.add_argument("--natural-scores-dir", type=str, default=None,
                        help="Natural (clean) run_score_reliability dir to re-score with the learned weights")
    parser.add_argument("--n-neighbors-grid", type=str, default="5,10,15,25,50")
    parser.add_argument("--reliable-percentile", type=float, default=30.0)
    parser.add_argument("--suspect-percentile", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    dirs = [Path(s.strip()) for s in args.scores_dirs.split(",")]
    flips = [Path(s.strip()) for s in args.flip_files.split(",")]
    if len(dirs) != 2 or len(flips) != 2:
        raise ValueError("Exactly two scores dirs and two flip files are required (cross-pi validation)")
    grid = [int(s.strip()) for s in args.n_neighbors_grid.split(",")]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = PROJECT_ROOT / "exp" / f"{timestamp}_score_optimization"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for d, f in zip(dirs, flips):
        preds, injected, anchor_ids = load_run(d, f)
        runs.append({"dir": str(d), "preds": preds, "injected": injected, "anchor_ids": anchor_ids})
        logger.info("Loaded %s: %d train fraud rows, %d injected (%.1f%%), design=%s",
                    d.name, len(injected), int(injected.sum()), 100 * injected.mean(),
                    "anchored" if anchor_ids is not None else "random")

    results = []

    # Baselines at the current configuration (k=15, default mixture)
    for i, run in enumerate(runs):
        feats, composite = compute_features(run["preds"], 15, run["anchor_ids"])
        run["feats15"] = feats
        auc_comp = detection_auc_from_reliable_score(run["injected"], composite)
        auc_mu = detection_auc_from_reliable_score(run["injected"], feats["mu"])
        results.append({"config": "baseline_composite_k15", "eval_run": i, "auc": auc_comp})
        results.append({"config": "baseline_mu_alone_k15", "eval_run": i, "auc": auc_mu})
        for name in BASE_FEATURES:
            a = roc_auc_score(run["injected"], run["feats15"][name])
            results.append({"config": f"solo_{name}_k15", "eval_run": i, "auc": float(max(a, 1 - a))})
        logger.info("run%d baselines: composite=%.3f, mu=%.3f", i, auc_comp, auc_mu)

    # Grid over n_neighbors x feature variant, cross-pi both directions
    both_anchored = all(r["anchor_ids"] is not None for r in runs)
    variants = [("base5", BASE_FEATURES)]
    if both_anchored:
        variants.append(("with_var", BASE_FEATURES + VAR_FEATURES))

    grid_summary = []
    for k in grid:
        featsets = []
        for run in runs:
            f, _ = compute_features(run["preds"], k, run["anchor_ids"])
            featsets.append(f)
        for vname, names in variants:
            aucs = []
            for fit_i, eval_i in ((0, 1), (1, 0)):
                X_fit = zscore_matrix(featsets[fit_i], names)
                X_eval = zscore_matrix(featsets[eval_i], names)
                lr = fit_lr(X_fit, runs[fit_i]["injected"], args.seed)
                p_injected = lr.predict_proba(X_eval)[:, 1]
                auc = float(roc_auc_score(runs[eval_i]["injected"], p_injected))
                aucs.append(auc)
                results.append({"config": f"learned_{vname}_k{k}", "fit_run": fit_i,
                                "eval_run": eval_i, "auc": auc})
            mean_auc = float(np.mean(aucs))
            grid_summary.append({"variant": vname, "n_neighbors": k, "mean_heldout_auc": mean_auc})
            logger.info("k=%-3d %-9s cross-pi AUCs: %.3f / %.3f (mean %.3f)",
                        k, vname, aucs[0], aucs[1], mean_auc)

    df_grid = pd.DataFrame(grid_summary)
    best = df_grid[df_grid.variant == "base5"].sort_values("mean_heldout_auc").iloc[-1]
    best_k = int(best["n_neighbors"])
    logger.info("Selected n_neighbors=%d (base5 mean held-out AUC %.3f)", best_k, best["mean_heldout_auc"])

    # Final weights: refit on both runs pooled (each z-scored within its cohort)
    featsets = []
    for run in runs:
        f, _ = compute_features(run["preds"], best_k, run["anchor_ids"])
        featsets.append(f)
    X_pool = np.vstack([zscore_matrix(f, BASE_FEATURES) for f in featsets])
    y_pool = np.concatenate([r["injected"] for r in runs])
    lr = fit_lr(X_pool, y_pool, args.seed)

    # LR predicts "injected" (unreliable): negate so higher = more reliable.
    weights = (-lr.coef_[0]).tolist()
    intercept = float(-lr.intercept_[0])
    learned = {
        "features": BASE_FEATURES,
        "weights": weights,
        "intercept": intercept,
        "n_neighbors": best_k,
        "orientation": "higher = more reliable",
        "standardization": "z-score each component within the scoring cohort at apply time",
        "trained_on": [r["dir"] for r in runs],
        "target": "injected flag (random bogus->fraud flips)",
        "validation": {
            "scheme": "cross-pi, both directions",
            "mean_heldout_auc": float(best["mean_heldout_auc"]),
            "baseline_composite_k15": float(np.mean([r["auc"] for r in results
                                                     if r["config"] == "baseline_composite_k15"])),
            "baseline_mu_alone_k15": float(np.mean([r["auc"] for r in results
                                                    if r["config"] == "baseline_mu_alone_k15"])),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "learned_weights.json", "w") as f:
        json.dump(learned, f, indent=2)
    logger.info("Learned weights (z-scored features, higher=reliable): %s",
                dict(zip(BASE_FEATURES, [round(w, 4) for w in weights])))

    pd.DataFrame(results).to_csv(output_dir / "optimization_results.csv", index=False)
    df_grid.to_csv(output_dir / "optimization_grid_summary.csv", index=False)

    # Re-score the natural run with the learned weights (zero GPU)
    if args.natural_scores_dir:
        nat_dir = Path(args.natural_scores_dir)
        nat_scores = pd.read_csv(nat_dir / "reliability_scores_all.csv")
        nat_vectors = pd.read_csv(nat_dir / "prediction_vectors.csv")
        probe_cols = [c for c in nat_vectors.columns if c.startswith("p_fraud_")]
        preds_nat = nat_vectors[probe_cols].values.astype(float)
        anchor_ids_nat = parse_anchor_ids(probe_cols)

        # Union cohort, mirroring run_score_reliability's design.
        feats_nat, _ = compute_features(preds_nat, best_k, anchor_ids_nat)
        Z = zscore_matrix(feats_nat, BASE_FEATURES)
        new_score = intercept + Z @ np.array(weights)

        fraud_mask = nat_scores["label"].values == 1
        rel_t = np.percentile(new_score[fraud_mask], 100 - args.reliable_percentile)
        sus_t = np.percentile(new_score[fraud_mask], args.suspect_percentile)
        classification = np.where(new_score >= rel_t, "reliable",
                                  np.where(new_score < sus_t, "suspect", "uncertain"))

        out = nat_scores.copy()
        old = out["reliability_score"].values
        out["reliability_score"] = new_score
        out["mu"] = feats_nat["mu"]
        out["sigma"] = feats_nat["sigma"]
        out["entropy"] = feats_nat["entropy"]
        out["agreement"] = feats_nat["agreement"]
        out["density"] = feats_nat["density"]
        out["classification"] = classification
        out.to_csv(output_dir / "rescored_reliability_scores_all.csv", index=False)

        from scipy.stats import spearmanr
        # scipy < 1.9 names the field .correlation; >= 1.9 adds .statistic.
        # Indexing works across all versions.
        rho = float(spearmanr(old[fraud_mask], new_score[fraud_mask])[0])
        logger.info("Natural re-score: Spearman(old, new) on fraud rows = %.3f; "
                    "thresholds reliable=%.4f suspect=%.4f; saved rescored CSV for Level-2 checks",
                    rho, rel_t, sus_t)

    manifest = {
        "artifact_type": "score_optimization",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"scores_dirs": [str(d) for d in dirs],
                   "flip_files": [str(f) for f in flips],
                   "natural_scores_dir": args.natural_scores_dir},
        "grid": {"n_neighbors": grid, "variants": [v for v, _ in variants]},
        "selected": {"n_neighbors": best_k, "exported_features": BASE_FEATURES,
                     "mean_heldout_auc": float(best["mean_heldout_auc"])},
        "notes": [
            "with_var variant reported for information only; exported weights use the five portable components",
            "weights learned on random-flip noise: validate Level-2 on natural data before adoption",
        ],
        "seed": args.seed,
    }
    with open(output_dir / "artifact_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved learned_weights.json, optimization_results.csv, manifest to %s", output_dir)


if __name__ == "__main__":
    main()
