"""Cheap pre-flight check of the anchor-selection KMeans (no GPU, no TabICL).

Replicates the anchor clustering step of build_anchored_support_sets on the
bogus training rows under both feature preparations:

  raw   — impute_for_clustering (NaN median-fill only; the preparation that
          produced ~44% singleton clusters on sentinel-scale columns)
  fixed — prepare_features_for_clustering (sentinel masking + standardization)

and prints the cluster-size distribution for each, plus the columns with
sentinel-scale values. Run this after any feature-set change and BEFORE
spending GPU time on run_score_reliability.py --probe-design anchored:
the fixed distribution should show few/no singletons and no giant clusters.

Usage:
    python check_anchor_clusters.py --setting setting5 --top-features 150
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent / "supporting_set_constr"))

from config import DataConfig
from data_loader import DataLoader
from support_set_selector import (
    SENTINEL_ABS_THRESHOLD,
    impute_for_clustering,
    prepare_features_for_clustering,
)

logger = logging.getLogger(__name__)


def cluster_stats(X_prepared: np.ndarray, n_clusters: int, random_state: int) -> dict:
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=3)
    km.fit(X_prepared)
    sizes = np.bincount(km.labels_, minlength=n_clusters)
    sizes = sizes[sizes > 0]
    return {
        "n_clusters_nonempty": int(len(sizes)),
        "n_singletons": int((sizes == 1).sum()),
        "singleton_pct": float(100 * (sizes == 1).mean()),
        "median_size": float(np.median(sizes)),
        "top5_sizes": sorted(sizes.tolist())[-5:],
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-flight check of anchor clustering balance")
    parser.add_argument("--setting", type=str, required=True)
    parser.add_argument("--top-features", type=int, default=150)
    parser.add_argument("--support-size", type=int, default=500,
                        help="Probe support size; anchor clustering uses size//2 clusters")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_config = DataConfig(setting=args.setting, top_features=args.top_features)
    loader = DataLoader(data_config)
    data = loader.load_all()
    X_train, y_train, _ = data["train"]

    pos_mask = y_train == data_config.positive_class
    X_pos = X_train.values[pos_mask]
    n_clusters = args.support_size // 2
    logger.info("Bogus training rows: %d, features: %d, anchor clusters: %d",
                len(X_pos), X_pos.shape[1], n_clusters)

    # Sentinel audit: columns with values at sentinel scale.
    with np.errstate(invalid="ignore"):
        sentinel_cells = np.abs(X_pos) >= SENTINEL_ABS_THRESHOLD
    col_counts = np.nansum(sentinel_cells, axis=0)
    offenders = np.argsort(col_counts)[::-1]
    logger.info("Sentinel audit (|x| >= %.0e): %d cells across %d columns",
                SENTINEL_ABS_THRESHOLD, int(col_counts.sum()), int((col_counts > 0).sum()))
    for i in offenders[:10]:
        if col_counts[i] == 0:
            break
        logger.info("  %-30s %6d sentinel cells (%.1f%% of bogus rows)",
                    X_train.columns[i], int(col_counts[i]), 100 * col_counts[i] / len(X_pos))

    if len(X_pos) <= n_clusters:
        logger.info("Fewer bogus rows than clusters — anchored design would use all positives; nothing to check.")
        return

    for name, prepare in [("raw (old)", impute_for_clustering),
                          ("fixed (sentinel-masked + standardized)", prepare_features_for_clustering)]:
        stats = cluster_stats(prepare(X_pos), n_clusters, args.seed)
        logger.info("%s: singletons=%d (%.1f%%), median size=%.0f, top-5 sizes=%s",
                    name, stats["n_singletons"], stats["singleton_pct"],
                    stats["median_size"], stats["top5_sizes"])

    logger.info("Healthy fixed distribution: singleton_pct in low single digits, "
                "top-5 sizes within a small multiple of the median.")


if __name__ == "__main__":
    main()
