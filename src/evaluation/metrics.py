"""Model evaluation metrics for classification and flood models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)
from loguru import logger


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    label_names: list[str] | None = None,
) -> pd.DataFrame:
    """Per-label precision, recall, F1 for multi-label classification."""
    rows = []
    n_labels = y_true.shape[1] if y_true.ndim > 1 else 1
    for i in range(n_labels):
        yt = y_true[:, i] if y_true.ndim > 1 else y_true
        yp = y_pred[:, i] if y_pred.ndim > 1 else y_pred
        row = {
            "label": label_names[i] if label_names else str(i),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
            "support": int(yt.sum()),
        }
        if y_prob is not None:
            ypr = y_prob[:, i] if y_prob.ndim > 1 else y_prob
            row["ap"] = average_precision_score(yt, ypr)
        rows.append(row)
    return pd.DataFrame(rows)


def flood_model_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute ROC-AUC, PR-AUC, Brier score for binary flood prediction."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "recall_top10pct": _recall_top_k(y_true, y_prob, k=0.10),
    }


def _recall_top_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.10) -> float:
    """Recall among top-k% highest predicted risk locations."""
    n = max(1, int(len(y_prob) * k))
    top_idx = np.argsort(y_prob)[::-1][:n]
    return y_true[top_idx].sum() / max(1, y_true.sum())
