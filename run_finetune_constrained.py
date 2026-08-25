"""Constrained fine-tuning of TabICL: maximize bogus recall s.t. precision >= P.

Implements the primal-dual (Lagrangian) mechanism on top of
FinetunedTabICLClassifier -- no changes to the tabicl package. Per meta-batch:

  s_i      = sigmoid((f(x_i) - t) / tau)          soft "flagged as bogus" vote
  softTP   = sum s_i over bogus-labeled queries
  softFP   = sum s_i over fraud-labeled queries
  g        = [P * (softTP + softFP) - softTP] / n_query   (<= 0 iff precision >= P)
  loss     = -softRecall + lambda * g              primal: gradient descent on weights
  lambda  <- max(0, lambda + dual_lr * EMA(g))     dual: ascent on the multiplier

No reliability filtering: all training rows enter with their raw labels, and the
constraint / validation metric / final threshold all use raw labels (the business
metric). Early stopping and best-weight selection use raw-val recall @ precision
>= P; with --constrained off the script degrades to plain cross-entropy
fine-tuning under the identical evaluation protocol (the ablation arm).

class_shuffle_method is forced to "none" in both arms so ensemble views keep the
label identity (class 0 = bogus) that the precision/recall soft counts rely on.

After training: the operating threshold t* is calibrated on validation (max
recall s.t. precision >= P), then applied once to the test set. The frozen
pretrained checkpoint is evaluated under the same protocol as a baseline column
unless --skip-frozen-baseline is given.

Enterprise env notes: default --model-path points at the local checkpoint (no
download attempted); the tabicl fine-tuning loop imports `transformers`
(tabicl/train/_optim.py), so that package must be preinstalled.
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "supporting_set_constr"))

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, precision_recall_curve,
)

from config import DataConfig, PROJECT_ROOT, DEFAULT_MODEL_PATH
from data_loader import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threshold calibration and metrics (positive class = bogus = raw label 0)
# ---------------------------------------------------------------------------

def calibrate_threshold_at_precision(y_binary, pos_proba, precision_target):
    """Pick the threshold maximizing recall s.t. precision >= target.

    Returns (threshold, feasible). If no threshold reaches the target
    precision, returns the threshold of maximum precision (feasible=False).
    """
    prec, rec, thresholds = precision_recall_curve(y_binary, pos_proba)
    prec, rec = prec[:-1], rec[:-1]  # align with `thresholds`
    if len(thresholds) == 0:
        return 0.5, False
    ok = prec >= precision_target
    if ok.any():
        idx = np.flatnonzero(ok)[np.argmax(rec[ok])]
        return float(thresholds[idx]), True
    return float(thresholds[int(np.argmax(prec))]), False


def recall_at_precision_score(y_binary, pos_proba, precision_target):
    """Scalar model-selection metric: recall @ precision>=P when feasible
    (in (0, 1]); otherwise best_precision - P (negative), so any feasible
    model ranks above any infeasible one and both regimes order smoothly."""
    prec, rec, thresholds = precision_recall_curve(y_binary, pos_proba)
    prec, rec = prec[:-1], rec[:-1]
    if len(thresholds) == 0:
        return -precision_target
    ok = prec >= precision_target
    if ok.any():
        return float(rec[ok].max())
    return float(prec.max() - precision_target)


def compute_metrics(y_true, proba, precision_target, threshold=None):
    """threshold=None calibrates on this split (use for val); pass val's
    threshold when scoring test."""
    pos_proba = proba[:, 0]
    y_binary = (np.asarray(y_true) == 0).astype(int)

    feasible = None
    if threshold is None:
        threshold, feasible = calibrate_threshold_at_precision(
            y_binary, pos_proba, precision_target)

    y_pred = (pos_proba >= threshold).astype(int)
    out = {
        "roc_auc": float(roc_auc_score(y_binary, pos_proba)),
        "pr_auc": float(average_precision_score(y_binary, pos_proba)),
        "precision": float(precision_score(y_binary, y_pred, zero_division=0)),
        "recall": float(recall_score(y_binary, y_pred, zero_division=0)),
        "f1": float(f1_score(y_binary, y_pred)),
        "recall_at_precision_target": recall_at_precision_score(
            y_binary, pos_proba, precision_target),
        "threshold": float(threshold),
        "n_flagged_bogus": int(y_pred.sum()),
        "n_true_bogus": int(y_binary.sum()),
    }
    if feasible is not None:
        out["precision_target_feasible"] = bool(feasible)
    return out


# ---------------------------------------------------------------------------
# Primal-dual subclass (defers all plumbing to the tabicl base class)
# ---------------------------------------------------------------------------

def _build_classifier_class():
    """Import inside a function so --help works without torch/tabicl."""
    from tabicl import FinetunedTabICLClassifier
    from tabicl._finetune.base import ValidationMetrics

    class ConstrainedFinetunedTabICL(FinetunedTabICLClassifier):
        """Neyman-Pearson fine-tuning via the documented base-class hooks.

        Inherits __init__ untouched (keeps sklearn get_params/repr working);
        the constrained objective is armed post-construction via
        set_constrained_objective(). When not armed, behaves exactly like the
        stock cross-entropy fine-tuner apart from the validation metric,
        which is always raw-val recall @ precision >= P so that the two arms
        share model selection and the never-worse-than-pretrained guarantee.
        """

        def set_constrained_objective(self, *, enabled, precision_target,
                                      dual_lr, surrogate_temp, train_threshold,
                                      lambda_init, lambda_ema_beta):
            self._np = SimpleNamespace(
                enabled=enabled,
                precision_target=precision_target,
                dual_lr=dual_lr,
                surrogate_temp=surrogate_temp,
                train_threshold=train_threshold,
                lambda_ema_beta=lambda_ema_beta,
            )
            self._lam = float(lambda_init)
            self._g_ema = None
            self._dual_log = []  # (batch_idx, lambda, g, soft_recall, soft_precision)

        # ---- primal loss + dual update ----
        def _compute_batch_loss(self, batch, model):
            if not self._np.enabled:
                return super()._compute_batch_loss(batch, model)

            cfg = self._np
            logits = model(batch.X, batch.y_train.float())
            n_classes = int(batch.y_train.max().item()) + 1
            # class_shuffle_method="none" => column 0 is bogus in every view
            probs = torch.softmax(logits[..., :n_classes].float(), dim=-1)
            p_bogus = probs[..., 0]                      # (E, n_query)
            y = batch.y_query.long()                     # (E, n_query)

            s = torch.sigmoid((p_bogus - cfg.train_threshold) / cfg.surrogate_temp)
            is_bogus = (y == 0).float()
            soft_tp = (s * is_bogus).sum()
            soft_fp = (s * (1.0 - is_bogus)).sum()
            n_bogus = is_bogus.sum().clamp_min(1.0)
            soft_recall = soft_tp / n_bogus
            # normalize the gap by query count so dual_lr is batch-size invariant
            n_query = float(y.numel())
            g = (cfg.precision_target * (soft_tp + soft_fp) - soft_tp) / n_query

            loss = -soft_recall + self._lam * g

            # dual ascent (dL/dlambda = g exactly; no autograd involved)
            g_val = float(g.detach().item())
            if self._g_ema is None:
                self._g_ema = g_val
            else:
                b = cfg.lambda_ema_beta
                self._g_ema = (1.0 - b) * self._g_ema + b * g_val
            self._lam = max(0.0, self._lam + cfg.dual_lr * self._g_ema)

            soft_prec = float((soft_tp / (soft_tp + soft_fp).clamp_min(1e-8)).detach().item())
            self._dual_log.append((len(self._dual_log), self._lam, g_val,
                                   float(soft_recall.detach().item()), soft_prec))
            return loss

        # ---- model selection on the business metric (both arms) ----
        def _run_validation(self, inner, X_train, y_train, X_val, y_val):
            try:
                inner.fit(X_train, y_train)
                proba = inner.predict_proba(X_val)
            except (ValueError, RuntimeError) as e:
                logger.warning("Validation failed: %s", e)
                return ValidationMetrics(primary=float("nan"))

            # y_val is label-encoded; {0,1} encodes to identity => 0 = bogus
            y_binary = (np.asarray(y_val) == 0).astype(int)
            pos_proba = proba[:, 0]
            primary = recall_at_precision_score(
                y_binary, pos_proba, self._np.precision_target)
            secondary = {
                "recall_at_precision_target": primary,
                "roc_auc": float(roc_auc_score(y_binary, pos_proba)),
                "pr_auc": float(average_precision_score(y_binary, pos_proba)),
                "lambda": self._lam,
                "g_ema": self._g_ema if self._g_ema is not None else 0.0,
            }
            return ValidationMetrics(primary=primary, secondary=secondary)

        @property
        def _metric_name(self):
            return "recall@precision>=%.2f" % self._np.precision_target

    return ConstrainedFinetunedTabICL


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Constrained (primal-dual) TabICL fine-tuning: max recall s.t. precision >= P")
    parser.add_argument("--setting", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--top-features", type=int, default=150)
    # constrained objective
    parser.add_argument("--constrained", type=str, default="on", choices=["on", "off"],
                        help="off = plain cross-entropy fine-tuning (ablation arm)")
    parser.add_argument("--precision-target", type=float, default=0.70,
                        help="Precision floor P on the bogus class")
    parser.add_argument("--dual-lr", type=float, default=0.05,
                        help="Dual ascent step on lambda per meta-batch (gap normalized per query row)")
    parser.add_argument("--surrogate-temp", type=float, default=0.08,
                        help="Sigmoid temperature tau of the soft counts")
    parser.add_argument("--train-threshold", type=float, default=0.5,
                        help="Reference threshold t inside the surrogate (deployment t* is recalibrated on val)")
    parser.add_argument("--lambda-init", type=float, default=0.0)
    parser.add_argument("--lambda-ema-beta", type=float, default=0.1,
                        help="EMA coefficient on the constraint gap driving the dual update")
    # optimization (mirrors run_finetune.py)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-data-size", type=int, default=10000)
    parser.add_argument("--ctx-query-ratio", type=float, default=0.2)
    parser.add_argument("--n-estimators-finetune", type=int, default=2)
    parser.add_argument("--n-estimators-inference", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--freeze-col", action="store_true")
    parser.add_argument("--freeze-row", action="store_true")
    parser.add_argument("--freeze-icl", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-frozen-baseline", action="store_true",
                        help="Skip evaluating the pretrained checkpoint under the same protocol")
    args = parser.parse_args()

    constrained = args.constrained == "on"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    arm = "constrained" if constrained else "ce"
    exp_name = f"{timestamp}_finetune_{arm}_{args.setting}"
    output_dir = PROJECT_ROOT / "exp" / exp_name
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    data_config = DataConfig(setting=args.setting, top_features=args.top_features)
    loader = DataLoader(data_config)
    data = loader.load_all()
    X_train, y_train, _ = data["train"]
    X_val, y_val, ids_val = data["val"]
    X_test, y_test, ids_test = data["test"]

    logger.info("Data: train=%d, val=%d, test=%d, features=%d",
                len(y_train), len(y_val), len(y_test), X_train.shape[1])
    logger.info("Arm: %s | P=%.2f | output: %s", arm, args.precision_target, output_dir)

    ConstrainedFinetunedTabICL = _build_classifier_class()
    clf = ConstrainedFinetunedTabICL(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_data_size=args.max_data_size,
        finetune_ctx_query_ratio=args.ctx_query_ratio,
        n_estimators_finetune=args.n_estimators_finetune,
        n_estimators_inference=args.n_estimators_inference,
        patience=args.patience,
        freeze_col=args.freeze_col,
        freeze_row=args.freeze_row,
        freeze_icl=args.freeze_icl,
        model_path=args.model_path,
        allow_auto_download=False,
        # keep label identity across ensemble views: soft counts need to know
        # which logit column is bogus (see module docstring)
        class_shuffle_method="none",
        device=args.device,
        random_state=args.seed,
        verbose=True,
    )
    clf.set_constrained_objective(
        enabled=constrained,
        precision_target=args.precision_target,
        dual_lr=args.dual_lr,
        surrogate_temp=args.surrogate_temp,
        train_threshold=args.train_threshold,
        lambda_init=args.lambda_init,
        lambda_ema_beta=args.lambda_ema_beta,
    )

    logger.info("Fine-tuning up to %d epochs (early stopping on raw-val recall@precision>=%.2f, patience=%d)",
                args.epochs, args.precision_target, args.patience)
    clf.fit(X_train.values, y_train, X_val=X_val.values, y_val=y_val, output_dir=ckpt_dir)

    if clf._dual_log:
        pd.DataFrame(
            clf._dual_log,
            columns=["batch", "lambda", "g", "soft_recall", "soft_precision"],
        ).to_csv(output_dir / "dual_dynamics.csv", index=False)
        logger.info("Dual dynamics (%d batches) saved; final lambda=%.4f",
                    len(clf._dual_log), clf._lam)

    # ---- evaluation: calibrate t* on val, apply once to test ----
    rows, preds = [], {}

    def evaluate_arm(name, predict_proba):
        val_proba = predict_proba(X_val.values)
        val_m = compute_metrics(y_val, val_proba, args.precision_target)
        test_proba = predict_proba(X_test.values)
        test_m = compute_metrics(y_test, test_proba, args.precision_target,
                                 threshold=val_m["threshold"])
        logger.info("[%s] val:  %s", name, val_m)
        logger.info("[%s] test: %s", name, test_m)
        rows.append({"arm": name, "split": "val", **val_m})
        rows.append({"arm": name, "split": "test", **test_m})
        preds[name] = (val_proba, test_proba)
        return val_m, test_m

    logger.info("Evaluating fine-tuned model (threshold calibrated on val)")
    ft_val, ft_test = evaluate_arm("finetuned_" + arm, clf.predict_proba)

    frozen_val = frozen_test = None
    if not args.skip_frozen_baseline:
        logger.info("Evaluating frozen pretrained checkpoint under the same protocol")
        from tabicl import TabICLClassifier
        frozen = TabICLClassifier(
            n_estimators=args.n_estimators_inference,
            model_path=args.model_path,
            allow_auto_download=False,
            device=args.device,
            random_state=args.seed,
        )
        frozen.fit(X_train.values, y_train)
        frozen_val, frozen_test = evaluate_arm("frozen_pretrained", frozen.predict_proba)

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(output_dir / "finetune_metrics.csv", index=False)
    logger.info("Metrics saved to %s", output_dir / "finetune_metrics.csv")

    for name, (val_proba, test_proba) in preds.items():
        pd.DataFrame({
            "ar_case_no": ids_val, "split": "val", "label": y_val,
            "p_bogus": val_proba[:, 0],
        }).to_csv(output_dir / f"predictions_val_{name}.csv", index=False)
        pd.DataFrame({
            "ar_case_no": ids_test, "split": "test", "label": y_test,
            "p_bogus": test_proba[:, 0],
        }).to_csv(output_dir / f"predictions_test_{name}.csv", index=False)

    manifest = {
        "artifact_type": "finetune_constrained",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "data_setting": args.setting,
            "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
            "top_features": args.top_features,
        },
        "objective": {
            "constrained": constrained,
            "precision_target": args.precision_target,
            "dual_lr": args.dual_lr,
            "surrogate_temp": args.surrogate_temp,
            "train_threshold": args.train_threshold,
            "lambda_init": args.lambda_init,
            "lambda_ema_beta": args.lambda_ema_beta,
            "final_lambda": clf._lam,
            "reliability_filtering": False,
            "constraint_labels": "raw",
        },
        "finetune": {
            "base_checkpoint": args.model_path,
            "best_checkpoint": str(ckpt_dir / "best.ckpt"),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "max_data_size": args.max_data_size,
            "finetune_ctx_query_ratio": args.ctx_query_ratio,
            "n_estimators_finetune": args.n_estimators_finetune,
            "n_estimators_inference": args.n_estimators_inference,
            "patience": args.patience,
            "class_shuffle_method": "none",
            "freeze_col": args.freeze_col,
            "freeze_row": args.freeze_row,
            "freeze_icl": args.freeze_icl,
            "device": args.device,
            "seed": args.seed,
        },
        "metrics": {
            "finetuned": {"val": ft_val, "test": ft_test},
            "frozen_pretrained": (
                {"val": frozen_val, "test": frozen_test} if frozen_val else None
            ),
        },
    }
    with open(output_dir / "artifact_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved. Best checkpoint: %s", ckpt_dir / "best.ckpt")


if __name__ == "__main__":
    main()
