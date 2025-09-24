from __future__ import annotations

from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from .. import config
from ..utils import prepare_tabular_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import time


def _covariance(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    return C


def _mat_sqrt_inv_sqrt(C: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    # Eigen decomposition; ensure symmetry
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C + eps * np.eye(C.shape[0]))
    w = np.clip(w, eps, None)
    sqrt = (V * np.sqrt(w)) @ V.T
    invsqrt = (V * (1.0 / np.sqrt(w))) @ V.T
    return sqrt, invsqrt


def _coral_align(Xs: np.ndarray, Xt: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    # Center
    mu_s = Xs.mean(axis=0, keepdims=True)
    mu_t = Xt.mean(axis=0, keepdims=True)
    Xs0 = Xs - mu_s
    Xt0 = Xt - mu_t
    Cs = _covariance(Xs0)
    Ct = _covariance(Xt0)
    Ct_sqrt, Cs_invsqrt = _mat_sqrt_inv_sqrt(Ct)
    Xs_aligned = Xs0 @ Cs_invsqrt @ Ct_sqrt + mu_t
    params = {"mu_s": mu_s, "mu_t": mu_t, "Cs_invsqrt": Cs_invsqrt, "Ct_sqrt": Ct_sqrt}
    return Xs_aligned, params


def _coral_transform(X: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    mu_s = params["mu_s"]; mu_t = params["mu_t"]; Cs_invsqrt = params["Cs_invsqrt"]; Ct_sqrt = params["Ct_sqrt"]
    return (X - mu_s) @ Cs_invsqrt @ Ct_sqrt + mu_t


def _split_domains(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> Tuple[List[str], List[str]]:
    # Prefer target groups that are unseen in train; else alphabetical split 60/40
    if config.COL_GROUP not in df.columns:
        return [], []
    g_train = set(df.iloc[train_idx][config.COL_GROUP].astype(str).dropna().unique().tolist())
    g_test = set(df.iloc[test_idx][config.COL_GROUP].astype(str).dropna().unique().tolist())
    unseen = sorted(list(g_test - g_train))
    if unseen:
        return sorted(list(g_train)), unseen
    # Fallback: disjoint alphabetical split over all groups
    allg = sorted(set(df[config.COL_GROUP].astype(str).dropna().unique().tolist()))
    if len(allg) < 2:
        return allg, []
    k = max(1, int(0.6 * len(allg)))
    return allg[:k], allg[k:]


def run(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
    why = (
        "AdaptKT-lite performs unsupervised CORAL feature alignment from a source domain (groups in train) "
        "to a target domain (unseen or held-out groups in test), then trains a logistic classifier on aligned source and evaluates on aligned target."
    )
    if config.COL_RESP not in df.columns or config.COL_GROUP not in df.columns:
        return {"category": "Domain Adaptive", "name": "AdaptKT-lite", "why": why, "error": "Required columns missing (group/response)"}

    # Build tabular features
    X_all = prepare_tabular_features(df, use_item=True, use_group=True, use_sex=True, use_time=True)
    if X_all.empty:
        return {"category": "Domain Adaptive", "name": "AdaptKT-lite", "why": why, "error": "No features available"}

    y_all = pd.to_numeric(df[config.COL_RESP], errors="coerce")
    mask_bin = y_all.isin([0, 1]).values

    # Domain split
    src_groups, tgt_groups = _split_domains(df, train_idx, test_idx)
    if not tgt_groups:
        return {"category": "Domain Adaptive", "name": "AdaptKT-lite", "why": why, "error": "No distinct target domain groups"}

    is_src = df[config.COL_GROUP].astype(str).isin(src_groups).values
    is_tgt = df[config.COL_GROUP].astype(str).isin(tgt_groups).values

    # Labeled source = intersection of train_idx, src, and binary mask
    src_rows = np.intersect1d(np.where(mask_bin & is_src)[0], train_idx)
    # Unlabeled target = test rows in target groups (use all features to estimate target stats)
    tgt_rows_unl = np.intersect1d(np.where(is_tgt)[0], test_idx)
    # Evaluation rows = test rows in target groups with labels
    tgt_rows_eval = np.intersect1d(np.where(mask_bin & is_tgt)[0], test_idx)

    if src_rows.size == 0 or tgt_rows_unl.size == 0 or tgt_rows_eval.size == 0:
        return {"category": "Domain Adaptive", "name": "AdaptKT-lite", "why": why, "error": "Insufficient rows for domain adaptation"}

    Xs = X_all.iloc[src_rows].values.astype(np.float64)
    Xt_unl = X_all.iloc[tgt_rows_unl].values.astype(np.float64)
    y_s = y_all.iloc[src_rows].astype(int).values

    # Standardize per-feature for stability before CORAL
    from sklearn.preprocessing import StandardScaler
    std = StandardScaler(with_mean=True, with_std=True)
    Xs = std.fit_transform(Xs)
    Xt_unl = std.transform(Xt_unl)

    # CORAL align source to target
    Xs_aligned, params = _coral_align(Xs, Xt_unl)

    # Train logistic on aligned source with lightweight C tuning
    from sklearn.linear_model import LogisticRegression
    start_time = time.perf_counter()
    budget = getattr(config, "TRAIN_TIME_BUDGET_S", None)
    C_grid = getattr(config, "LOGREG_C_GRID", [1.0])
    best_C = None
    best_score = -np.inf
    try:
        tr_in, va_in = train_test_split(
            np.arange(len(y_s)), test_size=0.2, random_state=config.RANDOM_STATE, stratify=y_s if len(np.unique(y_s)) > 1 else None
        )
        Xtr_in, Xva_in = Xs_aligned[tr_in], Xs_aligned[va_in]
        ytr_in, yva_in = y_s[tr_in], y_s[va_in]
        for C in C_grid:
            if budget is not None and (time.perf_counter() - start_time) > float(budget):
                break
            clf_tmp = LogisticRegression(
                C=float(C),
                max_iter=getattr(config, "ADAPT_MAX_ITER", 1000),
                class_weight="balanced",
                solver="lbfgs",
                random_state=config.RANDOM_STATE,
            )
            clf_tmp.fit(Xtr_in, ytr_in)
            p_va = clf_tmp.predict_proba(Xva_in)[:, 1]
            try:
                score = roc_auc_score(yva_in, p_va)
            except Exception:
                score = accuracy_score(yva_in, (p_va >= 0.5).astype(int))
            if score > best_score:
                best_score = score
                best_C = C
    except Exception:
        best_C = None

    clf = LogisticRegression(
        C=float(best_C) if best_C is not None else 1.0,
        max_iter=getattr(config, "ADAPT_MAX_ITER", 1000),
        class_weight="balanced",
        solver="lbfgs",
        random_state=config.RANDOM_STATE,
    )
    clf.fit(Xs_aligned, y_s)

    # Evaluate on aligned target eval rows
    X_eval = X_all.iloc[tgt_rows_eval].values.astype(np.float64)
    X_eval = std.transform(X_eval)
    X_eval = _coral_transform(X_eval, params)
    y_prob = clf.predict_proba(X_eval)[:, 1]
    y_true = y_all.iloc[tgt_rows_eval].astype(int).values

    return {
        "category": "Domain Adaptive",
        "name": "AdaptKT-lite (CORAL)",
        "why": why,
        "y_true": y_true,
        "y_prob": y_prob,
        "domain_source_groups": src_groups,
        "domain_target_groups": tgt_groups,
        "test_rows": tgt_rows_eval,
        "chosen_C": float(best_C) if best_C is not None else 1.0,
    }
