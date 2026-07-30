import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from config import PipelineConfig
from data_loader import DataLoader
from multi_gpu_inference import MultiGPUInference
from reliability_scorer import ReliabilityScorer
from support_set_selector import SupportSetSelector
from evaluate import Evaluator

logger = logging.getLogger(__name__)


class ConfidenceAwarePipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data_loader = DataLoader(config.data)
        self.multi_gpu = MultiGPUInference(config.gpu, config.tabicl)
        self.scorer = ReliabilityScorer(config.reliability)
        self.selector = SupportSetSelector(config.support_set, config.reliability)
        self.evaluator = Evaluator()
        self.history: List[Dict] = []

    def run(self) -> Dict:
        config = self.config
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        data = self.data_loader.load_all()
        X_train, y_train = data["train"]
        X_val, y_val = data["val"]

        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train

        X_pos, y_pos = self.data_loader.get_positive_samples(X_train, y_train)
        X_neg, y_neg = self.data_loader.get_negative_samples(X_train, y_train)
        X_pos_np = X_pos.values if isinstance(X_pos, pd.DataFrame) else X_pos
        X_neg_np = X_neg.values if isinstance(X_neg, pd.DataFrame) else X_neg
        y_pos_np = y_pos
        y_neg_np = y_neg

        X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else X_val

        logger.info(
            "Data loaded: train=%d (pos=%d, neg=%d), val=%d",
            len(y_train), len(y_pos), len(y_neg), len(y_val),
        )

        # Stage A: initial random support sets
        logger.info("Stage A: Building %d initial random support sets (size=%d)",
                    config.reliability.K, config.reliability.support_set_size)
        support_sets = self.selector.build_initial_support_sets(
            X_train_np, y_train_np,
            K=config.reliability.K,
            size=config.reliability.support_set_size,
            positive_class=config.data.positive_class,
        )

        best_support_set = None
        best_metric = 0.0

        for iteration in range(config.max_iterations):
            logger.info("=== Iteration %d/%d ===", iteration + 1, config.max_iterations)

            # Stage B: compute prediction vectors for all fraud samples
            logger.info("Stage B: Running %d support sets on %d fraud samples",
                        len(support_sets), len(X_neg_np))
            predictions_matrix = self.multi_gpu.predict_proba_multi_support(
                support_sets=support_sets,
                X_query=X_neg_np,
            )

            # Compute reliability scores
            reliability_scores = self.scorer.compute_scores(predictions_matrix)

            # Stage C: classify fraud samples
            classification = self.scorer.classify_samples(reliability_scores)
            n_reliable = int(classification["reliable"].sum())
            n_uncertain = int(classification["uncertain"].sum())
            n_suspect = int(classification["suspect"].sum())
            logger.info("Stage C: Reliable=%d, Uncertain=%d, Suspect=%d",
                        n_reliable, n_uncertain, n_suspect)

            # Save reliability scores and components as CSV
            scores_path = output_dir / f"reliability_scores_iter{iteration + 1}.npy"
            np.save(scores_path, reliability_scores)

            df_scores = pd.DataFrame({
                "reliability_score": reliability_scores,
                "mu": self.scorer.components_["mu"],
                "sigma": self.scorer.components_["sigma"],
                "entropy": self.scorer.components_["H"],
                "agreement": self.scorer.components_["A"],
                "density": self.scorer.components_["D"],
                "classification": np.where(
                    classification["reliable"], "reliable",
                    np.where(classification["uncertain"], "uncertain", "suspect")
                ),
            })
            df_scores.to_csv(output_dir / f"reliability_scores_iter{iteration + 1}.csv", index=False)

            # Stage D: build optimized support set
            logger.info("Stage D: Building optimized support set (size=%d)", config.support_set.target_size)
            optimized_support = self.selector.build_optimized_support_set(
                X_positives=X_pos_np,
                y_positives=y_pos_np,
                X_negatives=X_neg_np,
                y_negatives=y_neg_np,
                neg_reliability_scores=reliability_scores,
                neg_prediction_vectors=self.scorer.prediction_vectors_,
                neg_classification=classification,
                random_state=42 + iteration,
            )

            # Stage E: evaluate on validation set
            logger.info("Stage E: Evaluating on validation set")
            X_support, y_support = optimized_support
            val_proba = self.multi_gpu.predict_proba_parallel(
                X_support, y_support, X_val_np,
            )
            metrics = self.evaluator.compute_all(y_val, val_proba)
            metrics["iteration"] = iteration + 1
            metrics["n_reliable"] = n_reliable
            metrics["n_uncertain"] = n_uncertain
            metrics["n_suspect"] = n_suspect
            self.history.append(metrics)
            logger.info("Metrics: %s", metrics)

            current_metric = metrics[config.eval_metric]
            if current_metric > best_metric:
                best_metric = current_metric
                best_support_set = optimized_support
                # Save best support set
                np.save(output_dir / "best_support_X.npy", X_support)
                np.save(output_dir / "best_support_y.npy", y_support)

            # Convergence check
            if iteration > 0:
                prev_metric = self.history[-2][config.eval_metric]
                improvement = current_metric - prev_metric
                logger.info("Improvement: %.5f (threshold: %.5f)",
                            improvement, config.convergence_threshold)
                if improvement < config.convergence_threshold:
                    logger.info("Converged at iteration %d", iteration + 1)
                    break

            # Prepare K new diverse support sets for next iteration
            support_sets = self.selector.build_K_optimized_support_sets(
                X_positives=X_pos_np,
                y_positives=y_pos_np,
                X_negatives=X_neg_np,
                y_negatives=y_neg_np,
                neg_reliability_scores=reliability_scores,
                neg_prediction_vectors=self.scorer.prediction_vectors_,
                neg_classification=classification,
                K=config.reliability.K,
                base_random_state=42 + (iteration + 1) * 100,
            )

        # Save metrics history as CSV
        df_history = pd.DataFrame(self.history)
        df_history.to_csv(output_dir / "metrics_history.csv", index=False)
        logger.info("Metrics history saved to %s", output_dir / "metrics_history.csv")

        # Save final support set as CSV
        if best_support_set is not None:
            X_best, y_best = best_support_set
            df_support = pd.DataFrame(X_best, columns=X_train.columns if isinstance(X_train, pd.DataFrame) else None)
            df_support[config.data.label_col] = y_best
            df_support.to_csv(output_dir / "best_support_set.csv", index=False)
            logger.info("Best support set saved to %s", output_dir / "best_support_set.csv")

        return {
            "best_metric": best_metric,
            "history": self.history,
            "best_support_set_path": str(output_dir / "best_support_set.csv"),
        }

    def final_evaluation(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Run final evaluation on test set using saved best support set."""
        output_dir = Path(self.config.output_dir)
        X_support = np.load(output_dir / "best_support_X.npy")
        y_support = np.load(output_dir / "best_support_y.npy")

        test_proba = self.multi_gpu.predict_proba_parallel(X_support, y_support, X_test)
        return self.evaluator.compute_all(y_test, test_proba)
