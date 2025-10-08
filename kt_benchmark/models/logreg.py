from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import time

from .. import config
from ..utils import prepare_next_step_features_sparse_split


def run(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
    why = (
        "Logistic Regression is a strong, transparent ML baseline for binary correctness using tabular features. "
        "It is fast, robust to high-dimensional one-hot features, and provides calibrated probabilities."
    )
    if config.COL_RESP not in df.columns:
        return {"category": "Machine Learning", "name": "LogisticRegression", "why": why, "error": "response column missing"}

    X_tr, X_te, y_tr, y_te, rows_tr_next, rows_te_next = prepare_next_step_features_sparse_split(
        df,
        train_idx=train_idx,
        test_idx=test_idx,
        use_item=True,
        use_group=getattr(config, "LINEAR_USE_GROUP", True),
        use_sex=True,
        use_time=False,
    )
    if len(y_tr) == 0 or len(y_te) == 0:
        return {"category": "Machine Learning", "name": "LogisticRegression", "why": why, "error": "no valid next-step train/test rows"}

    
    start_time = time.perf_counter()
    budget = getattr(config, "TRAIN_TIME_BUDGET_S", None)
    
    C_grid = getattr(config, "LOGREG_C_GRID", [0.9, 1.0, 1.1])
    best_C = None
    best_score = -np.inf
    
    try:
        tr_idx_inner, va_idx_inner = train_test_split(
            np.arange(len(y_tr)), test_size=0.2, random_state=config.RANDOM_STATE, stratify=y_tr if len(np.unique(y_tr)) > 1 else None
        )
        X_tr_in, X_va_in = X_tr[tr_idx_inner], X_tr[va_idx_inner]
        y_tr_in, y_va_in = y_tr[tr_idx_inner], y_tr[va_idx_inner]
        # Allow tiny solver/class_weight tweaks from config (defaults are conservative)
        _solver = getattr(config, "LOGREG_SOLVER", "saga")  # saga and liblinear support sparse
        _cw = getattr(config, "LOGREG_CLASS_WEIGHT", "balanced")
        _max_iter = getattr(config, "LOGREG_MAX_ITER", 800)
        _tol = getattr(config, "LOGREG_TOL", 1e-3)
        _n_jobs = getattr(config, "LOGREG_N_JOBS", -1)
        for C in C_grid:
            if budget is not None and (time.perf_counter() - start_time) > float(budget):
                break
            clf_tmp = LogisticRegression(
                C=float(C),
                max_iter=_max_iter,
                tol=_tol,
                class_weight=_cw,
                solver=_solver,
                penalty="l2",
                random_state=config.RANDOM_STATE,
                n_jobs=_n_jobs,
            )
            clf_tmp.fit(X_tr_in, y_tr_in)
            prob_va = clf_tmp.predict_proba(X_va_in)[:, 1]
            # Prefer ROC AUC; if not defined due to one class, fall back to accuracy
            try:
                score = roc_auc_score(y_va_in, prob_va)
            except Exception:
                score = accuracy_score(y_va_in, (prob_va >= 0.5).astype(int))
            if score > best_score:
                best_score = score
                best_C = C
    except Exception:
        best_C = None

    clf = LogisticRegression(
        C=float(best_C) if best_C is not None else 1.0,
        max_iter=getattr(config, "LOGREG_MAX_ITER", 800),
        tol=getattr(config, "LOGREG_TOL", 1e-3),
        class_weight=getattr(config, "LOGREG_CLASS_WEIGHT", "balanced"),
        solver=getattr(config, "LOGREG_SOLVER", "saga"),
        penalty="l2",
        random_state=config.RANDOM_STATE,
        n_jobs=getattr(config, "LOGREG_N_JOBS", -1),
    )
    clf.fit(X_tr, y_tr)
    y_prob = clf.predict_proba(X_te)[:, 1]

    return {
        "category": "Machine Learning",
        "name": "LogisticRegression",
        "why": why,
        "y_true": y_te,
        "y_prob": y_prob,
        "test_rows": rows_te_next,
        "chosen_C": float(best_C) if best_C is not None else 1.0,
    }
