import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from config import PipelineConfig, FEATURE_IMPORTANCE_PATH
from data_loader import DataLoader
from multi_gpu_inference import MultiGPUInference, estimate_max_query_chunk
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
        X_train, y_train, ids_train = data["train"]
        X_val, y_val, ids_val = data["val"]

        feature_columns = X_train.columns.tolist()
        X_train_np = X_train.values
        y_train_np = y_train

        X_pos, y_pos, ids_pos = self.data_loader.get_positive_samples(X_train, y_train, ids_train)
        X_neg, y_neg, ids_neg = self.data_loader.get_negative_samples(X_train, y_train, ids_train)
        X_pos_np = X_pos.values
        X_neg_np = X_neg.values
        y_pos_np = y_pos
        y_neg_np = y_neg

        X_val_np = X_val.values

        logger.info(
            "Data loaded: train=%d (pos=%d, neg=%d), val=%d, features=%d",
            len(y_train), len(y_pos), len(y_neg), len(y_val), len(feature_columns),
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
        best_support_ids = None
        best_metric = 0.0

        for iteration in range(config.max_iterations):
            logger.info("=== Iteration %d/%d ===", iteration + 1, config.max_iterations)

            # Stage B: compute prediction vectors for all fraud samples
            logger.info("Stage B: Running %d support sets on %d fraud samples",
                        len(support_sets), len(X_neg_np))
            predictions_matrix = self.multi_gpu.predict_proba_multi_support(
                support_sets=support_sets,
                X_query=X_neg_np,
                desc=f"Stage B iter {iteration + 1}: scoring fraud samples",
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

            # Save reliability scores with ar_case_no
            df_scores = pd.DataFrame({
                config.data.id_col: ids_neg.values,
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
            optimized_support, selected_indices = self.selector.build_optimized_support_set(
                X_positives=X_pos_np,
                y_positives=y_pos_np,
                X_negatives=X_neg_np,
                y_negatives=y_neg_np,
                neg_reliability_scores=reliability_scores,
                neg_prediction_vectors=self.scorer.prediction_vectors_,
                neg_classification=classification,
                random_state=42 + iteration,
            )

            # Reconstruct ids for the support set
            pos_indices, neg_indices = selected_indices
            support_ids = pd.concat([
                ids_pos.iloc[pos_indices].reset_index(drop=True),
                ids_neg.iloc[neg_indices].reset_index(drop=True),
            ], ignore_index=True)

            # Stage E: evaluate on validation set
            logger.info("Stage E: Evaluating on validation set")
            X_support, y_support = optimized_support
            val_proba = self.multi_gpu.predict_proba_parallel(
                X_support, y_support, X_val_np,
                desc=f"Stage E iter {iteration + 1}: validation eval",
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
                best_support_ids = support_ids

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

        # Save final support set as CSV with ar_case_no and label
        if best_support_set is not None:
            X_best, y_best = best_support_set
            df_support = pd.DataFrame(X_best, columns=feature_columns)
            df_support.insert(0, config.data.id_col, best_support_ids.values)
            df_support[config.data.label_col] = y_best
            df_support.to_csv(output_dir / "best_support_set.csv", index=False)
            logger.info("Best support set saved to %s", output_dir / "best_support_set.csv")

        # Save artifact manifest
        chunk_size = estimate_max_query_chunk(
            n_support=config.reliability.support_set_size,
            n_features=config.data.top_features,
            n_estimators=config.tabicl.n_estimators,
        )
        manifest = {
            "artifact_type": "support_set_construction",
            "artifacts": {
                "best_support_set": str(output_dir / "best_support_set.csv"),
                "metrics_history": str(output_dir / "metrics_history.csv"),
                "reliability_scores": [
                    str(output_dir / f"reliability_scores_iter{i + 1}.csv")
                    for i in range(len(self.history))
                ],
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "data_setting": config.data.setting,
                "train_path": str(config.data.data_dir / config.data.train_file),
                "validation_path": str(config.data.data_dir / config.data.val_file),
                "test_path": str(config.data.data_dir / config.data.test_file),
                "n_train": len(y_train),
                "n_val": len(y_val),
            },
            "model": {
                "model_type": "TabICLClassifier",
                "model_path": config.tabicl.model_path,
                "n_estimators": config.tabicl.n_estimators,
                "batch_size": config.tabicl.batch_size,
                "softmax_temperature": config.tabicl.softmax_temperature,
                "kv_cache": config.tabicl.kv_cache,
                "random_state": config.tabicl.random_state,
                "chunk_size": chunk_size,
                "max_gpu_workers": config.gpu.num_gpus,
            },
            "preprocessing": {
                "feature_importance_path": str(FEATURE_IMPORTANCE_PATH),
                "top_features": config.data.top_features,
                "selected_features": feature_columns,
            },
            "pipeline": {
                "K": config.reliability.K,
                "support_set_size_initial": config.reliability.support_set_size,
                "target_support_set_size": config.support_set.target_size,
                "max_iterations": config.max_iterations,
                "convergence_threshold": config.convergence_threshold,
                "eval_metric": config.eval_metric,
                "iterations_run": len(self.history),
                "best_metric": best_metric,
            },
            "reliability_scoring": {
                "w_mean_prob": config.reliability.w_mean_prob,
                "w_stability": config.reliability.w_stability,
                "w_entropy": config.reliability.w_entropy,
                "w_agreement": config.reliability.w_agreement,
                "w_density": config.reliability.w_density,
                "n_neighbors": config.reliability.n_neighbors,
                "similarity_metric": config.reliability.similarity_metric,
                "reliable_threshold": config.reliability.reliable_threshold,
                "uncertain_threshold": config.reliability.uncertain_threshold,
            },
            "support_set_composition": {
                "positive_ratio": config.support_set.positive_ratio,
                "negative_reliable_ratio": config.support_set.negative_reliable_ratio,
                "negative_boundary_ratio": config.support_set.negative_boundary_ratio,
            },
        }
        with open(output_dir / "artifact_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Artifact manifest saved to %s", output_dir / "artifact_manifest.json")

        return {
            "best_metric": best_metric,
            "history": self.history,
            "best_support_set_path": str(output_dir / "best_support_set.csv"),
        }

    def final_evaluation(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Run final evaluation on test set using saved best support set."""
        output_dir = Path(self.config.output_dir)
        df_support = pd.read_csv(output_dir / "best_support_set.csv")
        y_support = df_support[self.config.data.label_col].values
        X_support = df_support[self.data_loader.selected_features].values.astype(np.float64)

        test_proba = self.multi_gpu.predict_proba_parallel(
            X_support, y_support, X_test,
            desc="Final test evaluation",
        )
        return self.evaluator.compute_all(y_test, test_proba)
