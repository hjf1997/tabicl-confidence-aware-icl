"""Estimate the fraud-label mislabeling rate rho from reliability-scoring artifacts.

Standalone follow-up to run_score_reliability.py. Answers: "what fraction of
fraud-labeled cases are actually bogus?" — the quantity the percentile
thresholds currently hard-code at 30% regardless of data quality.

Method (PU-learning mixture proportion estimation):
The fraud-labeled score distribution is a two-component mixture

    f_fraud(s) = (1 - rho) * f_F(s) + rho * f_B(s)

where f_B is the score distribution of truly-bogus cases. Because bogus labels
are trusted (the anchor assumption) and run_score_reliability.py scores ALL
rows, the bogus-labeled rows provide f_B directly. Two estimators:

- cdf_ratio: rho_hat = min_t CDF_fraud(t) / CDF_bogus(t). Since
  CDF_fraud >= rho * CDF_bogus everywhere, this is an UPPER bound on rho,
  tight when some score region contains (almost) no true frauds
  (the standard irreducibility condition).
- density_ratio: same idea bin-wise on histograms; noisier, cross-check.

Both are bootstrapped for CIs, computed pooled and per split, and run in up to
three score spaces:
- reliability_score and mu (TabICL probe view; shares the anchor assumption)
- an external model score, e.g. the production GBM's score_val (--gbm-csv).
  NOTE: a model trained on the noisy labels is biased toward reproducing them,
  pushing mislabeled cases toward the fraud side — its rho_hat is therefore
  biased LOW. Read reliability-space and GBM-space estimates as a bracket.

Interpretation guide (logged): if rho_hat is well below the suspect fraction
(default 30%), most of the suspect bucket is hard-but-correctly-labeled fraud,
and filtering it removes informative boundary examples; set
--suspect-percentile near 100*rho_hat in downstream runs instead.

Usage:
    python run_estimate_noise_rate.py \
        --scores exp/20260821_XXXX_reliability_scores_setting5 \
        [--gbm-csv data/tabular_dataset_full_single_transaction_2025_outcomes.csv]
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

from config import PROJECT_ROOT

logger = logging.getLogger(__name__)


def _orient(s_fraud: np.ndarray, s_bogus: np.ndarray):
    """Ensure higher score = more fraud-like (bogus concentrates low)."""
    if np.nanmean(s_fraud) < np.nanmean(s_bogus):
        return -s_fraud, -s_bogus, True
    return s_fraud, s_bogus, False


def cdf_ratio_estimator(s_fraud: np.ndarray, s_bogus: np.ndarray,
                        min_bogus_tail: int = 200, z: float = 1.0) -> float:
    """min_t over bogus quantiles of a penalized CDF ratio; upper bound on rho.

    CDF_fraud(t) >= rho * CDF_bogus(t) everywhere, so the ratio bounds rho from
    above, with equality under irreducibility. A raw min over many noisy
    thresholds selects the most downward-fluctuating one and can undercut the
    true bound; following the penalized MPE estimators (Blanchard/Scott), the
    min is taken over upper-confidence ratios instead: the numerator gets a
    +z * binomial-SE allowance, which cancels the min-selection bias.
    """
    thresholds = np.quantile(s_bogus, np.linspace(0.05, 0.95, 91))
    best = 1.0
    n_fraud, n_bogus = len(s_fraud), len(s_bogus)
    for t in thresholds:
        nb = (s_bogus <= t).sum()
        if nb < min_bogus_tail:
            continue
        p_f = (s_fraud <= t).mean()
        p_b = nb / n_bogus
        p_f_ucb = p_f + z * np.sqrt(p_f * (1 - p_f) / n_fraud) + 1.0 / n_fraud
        best = min(best, p_f_ucb / p_b)
    return float(min(best, 1.0))


def density_ratio_estimator(s_fraud: np.ndarray, s_bogus: np.ndarray,
                            n_bins: int = 25, min_bogus_bin: int = 30) -> float:
    """Bin-wise min f_fraud/f_bogus over bins where bogus has real mass."""
    lo = min(s_fraud.min(), s_bogus.min())
    hi = max(s_fraud.max(), s_bogus.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    h_fraud, _ = np.histogram(s_fraud, bins=bins, density=True)
    h_bogus, counts_edges = np.histogram(s_bogus, bins=bins, density=True)
    counts_bogus, _ = np.histogram(s_bogus, bins=bins)
    valid = counts_bogus >= min_bogus_bin
    if not valid.any():
        return 1.0
    ratios = h_fraud[valid] / h_bogus[valid]
    return float(min(ratios.min(), 1.0))


def bootstrap_ci(estimator, s_fraud, s_bogus, n_boot=200, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        f = rng.choice(s_fraud, size=len(s_fraud), replace=True)
        b = rng.choice(s_bogus, size=len(s_bogus), replace=True)
        vals.append(estimator(f, b))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def estimate_for_space(df: pd.DataFrame, score_col: str, label_col: str,
                       split: str, n_boot: int, seed: int) -> dict:
    sub = df if split == "pooled" else df[df["split"] == split]
    s_fraud = sub.loc[sub[label_col] == 1, score_col].dropna().values.astype(float)
    s_bogus = sub.loc[sub[label_col] == 0, score_col].dropna().values.astype(float)
    if len(s_fraud) < 200 or len(s_bogus) < 200:
        return None
    s_fraud, s_bogus, flipped = _orient(s_fraud, s_bogus)

    rho_cdf = cdf_ratio_estimator(s_fraud, s_bogus)
    ci_lo, ci_hi = bootstrap_ci(cdf_ratio_estimator, s_fraud, s_bogus,
                                n_boot=n_boot, seed=seed)
    rho_density = density_ratio_estimator(s_fraud, s_bogus)

    # Tightness diagnostic: the upper bound is only tight when bogus and fraud
    # score distributions separate. AUC(bogus vs fraud) near 0.5 => the
    # estimate is a loose bound, not a rate.
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.zeros(len(s_bogus)), np.ones(len(s_fraud))])
    s = np.concatenate([s_bogus, s_fraud])
    separation_auc = float(roc_auc_score(y, s))

    return {
        "score_space": score_col,
        "split": split,
        "n_fraud": int(len(s_fraud)),
        "n_bogus": int(len(s_bogus)),
        "orientation_flipped": bool(flipped),
        "separation_auc": separation_auc,
        "rho_cdf": rho_cdf,
        "rho_cdf_ci_lo": ci_lo,
        "rho_cdf_ci_hi": ci_hi,
        "rho_density": rho_density,
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate fraud-label noise rate rho")
    parser.add_argument("--scores", type=str, required=True,
                        help="run_score_reliability.py output dir, or path to reliability_scores_all.csv")
    parser.add_argument("--gbm-csv", type=str, default=None,
                        help="Optional external-model CSV for the cross-check (e.g. production GBM)")
    parser.add_argument("--gbm-id-col", type=str, default="ar_case_no")
    parser.add_argument("--gbm-score-col", type=str, default="score_val")
    parser.add_argument("--id-col", type=str, default="ar_case_no")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--suspect-percentile", type=float, default=30.0,
                        help="Suspect fraction used downstream, for the interpretation log")
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    scores_path = Path(args.scores)
    if scores_path.is_dir():
        scores_path = scores_path / "reliability_scores_all.csv"
    df = pd.read_csv(scores_path)
    logger.info("Loaded %d scored rows from %s", len(df), scores_path)

    score_spaces = ["reliability_score", "mu"]

    if args.gbm_csv:
        gbm = pd.read_csv(args.gbm_csv, usecols=[args.gbm_id_col, args.gbm_score_col])
        gbm = gbm.rename(columns={args.gbm_id_col: args.id_col})
        before = len(df)
        df = df.merge(gbm.drop_duplicates(args.id_col), on=args.id_col, how="left")
        matched = int(df[args.gbm_score_col].notna().sum())
        logger.info("GBM merge: matched %d/%d rows on %s", matched, before, args.id_col)
        score_spaces.append(args.gbm_score_col)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = PROJECT_ROOT / "exp" / f"{timestamp}_noise_rate_estimate"
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["pooled"] + sorted(df["split"].unique().tolist())
    results = []
    for space in score_spaces:
        for split in splits:
            r = estimate_for_space(df, space, args.label_col, split, args.n_boot, args.seed)
            if r is None:
                logger.warning("Skipping %s / %s (too few rows)", space, split)
                continue
            results.append(r)
            logger.info("%-18s %-7s rho_cdf=%.3f [%.3f, %.3f]  rho_density=%.3f  sep_auc=%.3f  (n_fraud=%d, n_bogus=%d)",
                        r["score_space"], r["split"], r["rho_cdf"],
                        r["rho_cdf_ci_lo"], r["rho_cdf_ci_hi"],
                        r["rho_density"], r["separation_auc"],
                        r["n_fraud"], r["n_bogus"])
            if r["separation_auc"] < 0.75:
                logger.warning("  %s/%s: separation AUC %.3f is low — rho_cdf is a LOOSE upper "
                               "bound here, not a rate estimate.",
                               r["score_space"], r["split"], r["separation_auc"])

    df_res = pd.DataFrame(results)
    df_res.to_csv(output_dir / "noise_rate_estimates.csv", index=False)

    # Interpretation: headline = pooled reliability-space estimate (upper bound
    # under irreducibility); GBM space, if present, brackets from below.
    headline = df_res[(df_res.score_space == "reliability_score") & (df_res.split == "pooled")]
    if len(headline):
        rho = float(headline.iloc[0]["rho_cdf"])
        frac = args.suspect_percentile / 100.0
        logger.info("=== Interpretation ===")
        logger.info("Headline rho_hat (upper bound) = %.3f vs suspect fraction = %.2f", rho, frac)
        if rho < frac:
            logger.info("At most ~%.0f%% of the suspect bucket can be truly mislabeled; "
                        "the remaining ~%.0f%% are likely hard-but-correct fraud labels "
                        "that filtering removes. Consider --suspect-percentile %.0f downstream.",
                        100 * rho / frac, 100 * (1 - rho / frac), 100 * rho)
        else:
            logger.info("rho_hat >= suspect fraction: the 30%% suspect bucket is consistent "
                        "with (or smaller than) the estimated noise mass.")

    manifest = {
        "artifact_type": "noise_rate_estimate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "scores_csv": str(scores_path),
            "gbm_csv": args.gbm_csv,
            "suspect_percentile": args.suspect_percentile,
        },
        "method": {
            "estimators": ["cdf_ratio (upper bound under irreducibility)",
                           "density_ratio (cross-check)"],
            "assumptions": [
                "bogus labels are clean (anchor assumption); bogus-labeled rows provide f_B",
                "reliability/mu spaces share the TabICL probe assumption stack",
                "external-model space is biased LOW if that model was trained on the noisy labels",
            ],
            "n_boot": args.n_boot,
            "seed": args.seed,
        },
        "results": results,
    }
    with open(output_dir / "artifact_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved %s and artifact_manifest.json to %s",
                "noise_rate_estimates.csv", output_dir)


if __name__ == "__main__":
    main()
