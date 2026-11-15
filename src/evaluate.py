"""
Evaluation, Probability Calibration & Threshold Optimization Module.
Computes ROC-AUC, PR-AUC, F1-Score, Brier Score, and supply chain operational thresholds.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, float]:
    """
    Evaluates binary classification predictions across production metrics.
    
    Returns:
        dict with ROC-AUC, PR-AUC, Brier Score, F1, Precision, Recall, and confusion counts.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "threshold": round(threshold, 3),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }
    return metrics


def optimize_classification_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.70,
    fn_cost_weight: float = 5.0,
    fp_cost_weight: float = 1.0
) -> Tuple[float, pd.DataFrame, Dict[str, Any]]:
    """
    Optimizes classification decision threshold on validation set.
    Balances high recall for supply chain delay intervention against operational false alarm fatigue.
    
    Cost Function:
        Total Cost = (fn_cost_weight * FN) + (fp_cost_weight * FP)
        
    Returns:
        (optimal_threshold, threshold_sweep_dataframe, optimal_metrics)
    """
    thresholds = np.linspace(0.05, 0.90, 86)
    records = []

    best_thresh = 0.50
    min_cost = float("inf")
    best_metrics = {}

    for t in thresholds:
        m = evaluate_predictions(y_true, y_prob, threshold=t)
        cost = (fn_cost_weight * m["false_negatives"]) + (fp_cost_weight * m["false_positives"])
        m["operational_cost"] = round(cost, 1)
        records.append(m)

        # Select threshold meeting recall target while minimizing business cost
        if m["recall"] >= min_recall and cost < min_cost:
            min_cost = cost
            best_thresh = t
            best_metrics = m

    # Fallback to max F1 if strict recall target was not reached
    if not best_metrics:
        sweep_df = pd.DataFrame(records)
        best_idx = sweep_df["f1_score"].idxmax()
        best_thresh = float(sweep_df.loc[best_idx, "threshold"])
        best_metrics = sweep_df.loc[best_idx].to_dict()
    else:
        sweep_df = pd.DataFrame(records)

    return best_thresh, sweep_df, best_metrics

