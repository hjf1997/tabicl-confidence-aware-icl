import argparse
import logging
import torch.multiprocessing as mp
from pathlib import Path

from config import PipelineConfig, DataConfig, TabICLConfig, MultiGPUConfig
from pipeline import ConfidenceAwarePipeline


def main():
    parser = argparse.ArgumentParser(description="Confidence-Aware Support Set Selection for TabICL")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing training.csv, validation.csv, test.csv")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory for output files")
    parser.add_argument("--model-path", type=str, default=None, help="Local TabICL checkpoint path (for offline env)")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--K", type=int, default=20, help="Number of diverse support sets")
    parser.add_argument("--support-size", type=int, default=500, help="Initial support set size per set")
    parser.add_argument("--target-size", type=int, default=1000, help="Optimized support set size")
    parser.add_argument("--max-iterations", type=int, default=5, help="Max EM iterations")
    parser.add_argument("--eval-test", action="store_true", help="Run final evaluation on test set after pipeline converges")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = PipelineConfig(
        max_iterations=args.max_iterations,
        output_dir=Path(args.output_dir),
        data=DataConfig(data_dir=Path(args.data_dir)),
        tabicl=TabICLConfig(model_path=args.model_path),
        gpu=MultiGPUConfig(
            num_gpus=args.num_gpus,
            devices=[f"cuda:{i}" for i in range(args.num_gpus)],
        ),
    )
    config.reliability.K = args.K
    config.reliability.support_set_size = args.support_size
    config.support_set.target_size = args.target_size

    pipeline = ConfidenceAwarePipeline(config)
    results = pipeline.run()

    logging.info("Pipeline complete. Best %s: %.4f", config.eval_metric, results["best_metric"])
    logging.info("Support set saved to: %s", results["best_support_set_path"])

    for i, h in enumerate(results["history"]):
        logging.info("  Iter %d: %s", i + 1, h)

    if args.eval_test:
        from data_loader import DataLoader
        import pandas as pd
        dl = DataLoader(config.data)
        data = dl.load_all()
        X_test, y_test = data["test"]
        X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        test_metrics = pipeline.final_evaluation(X_test_np, y_test)
        logging.info("Test set metrics: %s", test_metrics)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
