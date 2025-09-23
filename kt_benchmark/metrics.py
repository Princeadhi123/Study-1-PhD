from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    f1_score,
    precision_recall_fscore_support,
    brier_score_loss,
)


def safe_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # sanitize
    y_true = pd.to_numeric(pd.Series(y_true), errors="coerce").values
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    if mask.sum() == 0:
        return {"n": 0}
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    y_pred = (y_prob >= 0.5).astype(int)
    out["n"] = int(len(y_true))
    try:
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
    except Exception:
        out["accuracy"] = np.nan
    # Only compute ROC/PR if both classes present
    classes = np.unique(y_true)
    if len(classes) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            out["roc_auc"] = np.nan
        try:
            out["avg_precision"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            out["avg_precision"] = np.nan
        try:
            out["log_loss"] = float(log_loss(y_true, np.clip(y_prob, 1e-6, 1-1e-6)))
        except Exception:
            out["log_loss"] = np.nan
        try:
            out["f1"] = float(f1_score(y_true, y_pred))
        except Exception:
            out["f1"] = np.nan
        try:
            pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
            out["precision"] = float(pr)
            out["recall"] = float(rc)
            out["f1_binary"] = float(f1)
        except Exception:
            pass
        try:
            out["brier"] = float(brier_score_loss(y_true, y_prob))
        except Exception:
            out["brier"] = np.nan
    else:
        out["roc_auc"] = np.nan
        out["avg_precision"] = np.nan
        out["log_loss"] = np.nan
        out["f1"] = np.nan
        out["precision"] = np.nan
        out["recall"] = np.nan
        out["f1_binary"] = np.nan
        out["brier"] = np.nan
    return out
