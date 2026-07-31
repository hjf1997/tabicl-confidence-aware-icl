import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "supporting_set_constr"))

import torch.multiprocessing as mp

from config import PipelineConfig, DataConfig, TabICLConfig, MultiGPUConfig, PROJECT_ROOT, DEFAULT_MODEL_PATH
from pipeline import ConfidenceAwarePipeline


def main():
    parser = argparse.ArgumentParser(description="Confidence-Aware Support Set Selection for TabICL")
    parser.add_argument("--setting", type=str, required=True, help="Data setting folder name (e.g., setting1)")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Local TabICL checkpoint path")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--top-features", type=int, default=150, help="Number of top SHAP features to use")
    parser.add_argument("--K", type=int, default=20, help="Number of diverse support sets")
    parser.add_argument("--support-size", type=int, default=500, help="Initial support set size per set")
    parser.add_argument("--target-size", type=int, default=None, help="Final support set size (n/2 pos + n/2 neg). If not set, uses all positives + equal negatives")
    parser.add_argument("--max-iterations", type=int, default=1, help="Max iterations (default 1, no iterative refinement)")
    parser.add_argument("--threshold-method", type=str, default="percentile", choices=["fixed", "percentile"], help="Threshold method: fixed (manual thresholds) or percentile (adaptive)")
    parser.add_argument("--reliable-threshold", type=float, default=0.75, help="(fixed method) Reliability score threshold for reliable negatives")
    parser.add_argument("--uncertain-threshold", type=float, default=0.45, help="(fixed method) Reliability score threshold for uncertain vs suspect")
    parser.add_argument("--reliable-percentile", type=float, default=30.0, help="(percentile method) Top X%% classified as reliable")
    parser.add_argument("--suspect-percentile", type=float, default=30.0, help="(percentile method) Bottom X%% classified as suspect")
    parser.add_argument("--neg-sampling", type=str, default="random", choices=["random", "reliable", "informed", "kmeans", "boundary", "hybrid"], help="Negative sampling strategy: random, reliable (top-n scores), informed (variance + reliability), kmeans (representative), boundary (hard examples), hybrid (mixture)")
    parser.add_argument("--eval-test", action="store_true", help="Run final evaluation on test set after pipeline converges")
    args = parser.parse_args()

    # Generate experiment output directory: exp/YYYYMMDD_HHMM_support_set_constr_settingX/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = f"{timestamp}_support_set_constr_{args.setting}"
    output_dir = PROJECT_ROOT / "exp" / exp_name

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = PipelineConfig(
        max_iterations=args.max_iterations,
        output_dir=output_dir,
        data=DataConfig(
            setting=args.setting,
            top_features=args.top_features,
        ),
        tabicl=TabICLConfig(model_path=args.model_path),
        gpu=MultiGPUConfig(
            num_gpus=args.num_gpus,
            devices=[f"cuda:{i}" for i in range(args.num_gpus)],
        ),
    )
    config.reliability.K = args.K
    config.reliability.support_set_size = args.support_size
    config.reliability.threshold_method = args.threshold_method
    config.reliability.reliable_threshold = args.reliable_threshold
    config.reliability.uncertain_threshold = args.uncertain_threshold
    config.reliability.reliable_percentile = args.reliable_percentile
    config.reliability.suspect_percentile = args.suspect_percentile
    config.support_set.target_size = args.target_size
    config.support_set.neg_sampling_strategy = args.neg_sampling

    logging.info("Experiment output: %s", output_dir)
    logging.info("Data setting: %s", args.setting)
    logging.info("Top features: %d", args.top_features)

    pipeline = ConfidenceAwarePipeline(config)
    results = pipeline.run()

    logging.info("Pipeline complete. Best %s: %.4f", config.eval_metric, results["best_metric"])
    logging.info("Support set saved to: %s", results["best_support_set_path"])

    for i, h in enumerate(results["history"]):
        logging.info("  Iter %d: %s", i + 1, h)

    if args.eval_test:
        from data_loader import DataLoader
        dl = DataLoader(config.data)
        data = dl.load_all()
        X_test, y_test, _ = data["test"]
        X_test_np = X_test.values
        test_metrics = pipeline.final_evaluation(X_test_np, y_test)
        logging.info("Test set metrics: %s", test_metrics)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
