"""FLOPs per test sample as a function of support-set (target) size.

Analytical model of TabICL v2 inference cost (supporting_set_constr/flops_model.py),
validated against torch.utils.flop_counter.FlopCounterMode to <0.1% on the
full, cache-store, and cache-use forward paths.

For each target size n (the final support set has n rows: n/2 pos + n/2 neg):
- fit FLOPs: one-time cost of TabICLClassifier.fit() with kv_cache=True
  (encodes the support set, includes the n^2 ICL attention term)
- per-query FLOPs: marginal cost per test sample given the fitted cache
  (exactly linear in n: per_query = a + b*n)
- per-call totals for --n-query test samples, cached and uncached

Pure computation — no GPU, no data, no checkpoint needed. The optional
--empirical flag cross-checks the model against FlopCounterMode on a
random-weight TabICL (CPU, a few minutes).

Usage:
    python run_flops_target_size.py --target-sizes 100,200,500,1000,2000,5000 --top-features 298
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "supporting_set_constr"))

from config import PROJECT_ROOT
from flops_model import D, fit_flops, per_query_flops, predict_flops

logger = logging.getLogger(__name__)


def per_query_linear_coefficients(n_features: int, n_estimators: int):
    """Per-query FLOPs is exactly a + b*n_support; return (a, b)."""
    f0 = per_query_flops(0, n_features, n_estimators)["total"]
    f1 = per_query_flops(1, n_features, n_estimators)["total"]
    return f0, f1 - f0


def run_empirical_check() -> pd.DataFrame:
    """Cross-check the analytical model against FlopCounterMode (CPU, random weights)."""
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from torch.utils.flop_counter import FlopCounterMode
    from tabicl._model.tabicl import TabICL

    torch.manual_seed(0)
    model = TabICL()
    model.eval()

    rows = []
    for n, q, H in [(100, 50, 10), (300, 100, 40), (600, 100, 100)]:
        X_train = torch.randn(1, n, H)
        y_train = torch.randint(0, 2, (1, n)).float()
        X_test = torch.randn(1, q, H)

        fc = FlopCounterMode(display=False)
        with fc, sdpa_kernel(SDPBackend.MATH):
            model.forward_with_cache(X_train=X_train, y_train=y_train, store_cache=True)
        measured = fc.get_total_flops()
        ana = fit_flops(n, H, n_estimators=1)["total"]
        rows.append({"phase": "fit", "n_support": n, "n_query": q, "n_features": H,
                     "measured": measured, "analytical": ana,
                     "err_pct": 100 * (ana - measured) / measured})

        fc = FlopCounterMode(display=False)
        with fc, sdpa_kernel(SDPBackend.MATH):
            model.forward_with_cache(X_test=X_test, use_cache=True, store_cache=False,
                                     return_logits=False)
        measured = fc.get_total_flops()
        ana = predict_flops(q, n, H, n_estimators=1)["total"]
        rows.append({"phase": "predict_cached", "n_support": n, "n_query": q, "n_features": H,
                     "measured": measured, "analytical": ana,
                     "err_pct": 100 * (ana - measured) / measured})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="TabICL FLOPs vs support-set size")
    parser.add_argument("--target-sizes", type=str, required=True,
                        help="Comma-separated support set sizes (e.g., 100,200,500,1000,2000,5000)")
    parser.add_argument("--top-features", type=int, default=150,
                        help="Number of model features (use 298 for all features)")
    parser.add_argument("--n-estimators", type=int, default=8, help="TabICL ensemble members")
    parser.add_argument("--n-query", type=int, default=10000,
                        help="Test-set size for the per-call totals")
    parser.add_argument("--empirical", action="store_true",
                        help="Also cross-check against FlopCounterMode (CPU, a few minutes)")
    args = parser.parse_args()

    target_sizes = [int(s.strip()) for s in args.target_sizes.split(",")]
    H, B, Q = args.top_features, args.n_estimators, args.n_query

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = PROJECT_ROOT / "exp" / f"{timestamp}_flops_target_size"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    a, b = per_query_linear_coefficients(H, B)
    logger.info("Config: n_features=%d, n_estimators=%d, n_query=%d", H, B, Q)
    logger.info("Per-query FLOPs (kv_cache) = %.4e + %.4e * n_support", a, b)

    rows = []
    base = None
    base_single = None
    for n in target_sizes:
        pq = per_query_flops(n, H, B)
        fit = fit_flops(n, H, B)
        call_cached = predict_flops(Q, n, H, B, kv_cache=True)
        call_uncached = predict_flops(Q, n, H, B, kv_cache=False)
        # Production single-sample cost with NO pre-built cache: one query per
        # call, context encoded within the call (includes the quadratic n^2
        # ICL term). Applies when the service is stateless or the support set
        # changes per request; with a persistent fitted classifier, the cached
        # per_query_flops applies instead.
        single_uncached = predict_flops(1, n, H, B, kv_cache=False)
        if base is None:
            base = pq["total"]
            base_single = single_uncached["total"]
        rows.append({
            "target_size": n,
            "n_features": H,
            "n_estimators": B,
            "per_query_flops": pq["total"],
            "per_query_col": pq["col"],
            "per_query_row": pq["row"],
            "per_query_icl": pq["icl"],
            "per_query_decoder": pq["decoder"],
            "per_query_vs_smallest": pq["total"] / base,
            "single_query_uncached_flops": single_uncached["total"],
            "single_query_uncached_vs_smallest": single_uncached["total"] / base_single,
            "fit_flops": fit["total"],
            "fit_amortized_per_query": fit["total"] / Q,
            "call_flops_cached": call_cached["total"],
            "call_flops_uncached": call_uncached["total"],
            "cache_speedup": call_uncached["total"] / call_cached["total"],
        })
        logger.info("n=%6d  per_query=%.3e (x%.2f)  single_uncached=%.3e (x%.2f)  fit=%.3e  cache_speedup(q=%d)=%.2fx",
                    n, pq["total"], pq["total"] / base,
                    single_uncached["total"], single_uncached["total"] / base_single,
                    fit["total"], Q, call_uncached["total"] / call_cached["total"])

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "flops_target_size.csv", index=False)
    logger.info("Results saved to %s", output_dir / "flops_target_size.csv")

    validation = None
    if args.empirical:
        logger.info("Running empirical cross-check (FlopCounterMode, CPU)...")
        vdf = run_empirical_check()
        vdf.to_csv(output_dir / "empirical_validation.csv", index=False)
        validation = {"max_abs_err_pct": float(vdf["err_pct"].abs().max())}
        logger.info("Empirical check: max |err| = %.3f%%", validation["max_abs_err_pct"])

    manifest = {
        "artifact_type": "flops_target_size",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "model_type": "TabICL v2 (tabicl-classifier-v2-20260212)",
            "architecture": {
                "embed_dim": D.embed_dim, "col_num_blocks": D.col_num_blocks,
                "col_num_inds": D.col_num_inds, "row_num_blocks": D.row_num_blocks,
                "row_num_cls": D.row_num_cls, "icl_num_blocks": D.icl_num_blocks,
                "icl_dim": D.icl_dim, "ff_factor": D.ff_factor,
            },
        },
        "experiment": {
            "target_sizes": target_sizes,
            "n_features": H,
            "n_estimators": B,
            "n_query": Q,
            "per_query_linear_model": {
                "intercept_flops": a,
                "slope_flops_per_support_row": b,
                "formula": "per_query_flops = intercept + slope * n_support (kv_cache=True)",
            },
            "counting_convention": "matmul FLOPs only (1 MAC = 2 FLOPs), matching torch FlopCounterMode",
            "empirical_validation": validation,
        },
    }
    with open(output_dir / "artifact_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved to %s", output_dir / "artifact_manifest.json")


if __name__ == "__main__":
    main()
