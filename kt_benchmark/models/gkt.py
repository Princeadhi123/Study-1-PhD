from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from .. import config
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import time


def _build_group_graph(train_df: pd.DataFrame) -> pd.DataFrame:
    """Build a KC/KC co-occurrence graph between groups using train students.
    Weight = number of students appearing in both groups (co-participation).
    Returns a row-normalized adjacency (DataFrame) with index/columns = group labels.
    """
    gcol = config.COL_GROUP
    if gcol not in train_df.columns:
        return pd.DataFrame()
    # Student -> set(groups)
    sets = train_df.groupby(config.COL_ID)[gcol].apply(lambda s: set(s.dropna().astype(str))).tolist()
    groups = sorted({g for st in sets for g in st})
    if not groups:
        return pd.DataFrame()
    idx = {g: i for i, g in enumerate(groups)}
    W = np.zeros((len(groups), len(groups)), dtype=float)
    for st in sets:
        st = list(st)
        for i in range(len(st)):
            for j in range(i + 1, len(st)):
                a, b = idx[st[i]], idx[st[j]]
                W[a, b] += 1
                W[b, a] += 1
    # Row-normalize ignoring diagonal
    np.fill_diagonal(W, 0.0)
    row_sums = W.sum(axis=1, keepdims=True) + 1e-9
    Wn = W / row_sums
    adj = pd.DataFrame(Wn, index=groups, columns=groups)
    return adj


def run(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
    why = (
        "GKT-lite builds a knowledge-concept (KC) graph over groups (skills) via co-occurrence of student interactions, "
        "then smooths group difficulty over the graph to obtain predictions: p(g) blended with neighbors."
    )
    if config.COL_GROUP not in df.columns or config.COL_RESP not in df.columns:
        return {"category": "Graph", "name": "GKT-lite", "why": why, "error": "Required columns missing (group/response)"}

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Helper to compute smoothed probs dict for a frame and alpha
    def smooth_from_frame(frame: pd.DataFrame, alpha_val: float) -> tuple[Dict[str, float], float, pd.DataFrame]:
        grp_mean_loc = frame.groupby(config.COL_GROUP)[config.COL_RESP].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
        global_mean_loc = float(pd.to_numeric(frame[config.COL_RESP], errors="coerce").mean()) if not frame.empty else 0.5
        adj_loc = _build_group_graph(frame)
        sm: Dict[str, float] = {}
        for g in grp_mean_loc.index.astype(str):
            p_g = float(grp_mean_loc.get(g, global_mean_loc))
            if not adj_loc.empty and g in adj_loc.index:
                neigh = adj_loc.loc[g]
                p_nei = 0.0
                for h, w in neigh.items():
                    if w > 0:
                        p_nei += w * float(grp_mean_loc.get(h, global_mean_loc))
                p_smooth = alpha_val * p_g + (1 - alpha_val) * p_nei
            else:
                p_smooth = p_g
            sm[g] = float(np.clip(p_smooth, 1e-3, 1 - 1e-3))
        return sm, global_mean_loc, adj_loc

    # Lightweight alpha tuning using an inner validation split by students
    start_time = time.perf_counter()
    budget = getattr(config, "TRAIN_TIME_BUDGET_S", None)
    alpha_grid = getattr(config, "GKT_ALPHA_GRID", [getattr(config, "GKT_SMOOTH_ALPHA", 0.7)])

    best_alpha = None
    best_score = -np.inf

    # Build inner split only if ID column exists and enough students
    try:
        if config.COL_ID in train_df.columns:
            students = train_df[config.COL_ID].astype(str).unique()
            if len(students) >= 3:
                s_tr, s_va = train_test_split(students, test_size=0.2, random_state=config.RANDOM_STATE, shuffle=True)
                inner_tr = train_df[train_df[config.COL_ID].astype(str).isin(s_tr)].copy()
                inner_va = train_df[train_df[config.COL_ID].astype(str).isin(s_va)].copy()
            else:
                inner_tr, inner_va = train_df, pd.DataFrame(columns=train_df.columns)
        else:
            inner_tr, inner_va = train_df, pd.DataFrame(columns=train_df.columns)

        # Evaluate each alpha on validation
        for a in alpha_grid:
            if budget is not None and (time.perf_counter() - start_time) > float(budget):
                break
            sm_dict, global_mean_loc, _ = smooth_from_frame(inner_tr, float(a))
            if not inner_va.empty:
                g_series = inner_va[config.COL_GROUP].astype(str)
                y_true_va = pd.to_numeric(inner_va[config.COL_RESP], errors="coerce")
                mask = y_true_va.isin([0, 1]).values
                if mask.any():
                    y_true_v = y_true_va[mask].astype(int).values
                    y_prob_v = np.array([sm_dict.get(g, global_mean_loc) for g in g_series[mask]], dtype=float)
                    try:
                        score = roc_auc_score(y_true_v, y_prob_v)
                    except Exception:
                        score = accuracy_score(y_true_v, (y_prob_v >= 0.5).astype(int))
                else:
                    score = -np.inf
            else:
                # No validation available, use training likelihood proxy: mean abs error vs grp_mean
                g_series = inner_tr[config.COL_GROUP].astype(str)
                y_true_tr = pd.to_numeric(inner_tr[config.COL_RESP], errors="coerce")
                mask = y_true_tr.isin([0, 1]).values
                y_true_v = y_true_tr[mask].astype(int).values
                y_prob_v = np.array([sm_dict.get(g, global_mean_loc) for g in g_series[mask]], dtype=float)
                score = -float(np.mean(np.abs(y_true_v - y_prob_v))) if y_true_v.size else -np.inf
            if score > best_score:
                best_score = score
                best_alpha = float(a)
    except Exception:
        best_alpha = float(getattr(config, "GKT_SMOOTH_ALPHA", 0.7))

    # Fall back if tuning failed
    if best_alpha is None:
        best_alpha = float(getattr(config, "GKT_SMOOTH_ALPHA", 0.7))

    # Compute final smoothed dict on full train with best alpha
    smoothed, global_mean, _ = smooth_from_frame(train_df, best_alpha)

    # Predict for test rows by group lookup; keep only valid binary rows and return their original indices
    g_series = test_df[config.COL_GROUP].astype(str) if config.COL_GROUP in test_df.columns else pd.Series([], dtype=str)
    y_all = pd.to_numeric(test_df[config.COL_RESP], errors="coerce")
    mask = y_all.isin([0, 1])
    if not mask.any():
        return {"category": "Graph", "name": "GKT-lite", "why": why, "error": "No valid binary test rows"}
    rows = test_df.index[mask].to_numpy()
    y_true = y_all[mask].astype(int).values
    y_prob = np.array([smoothed.get(g, global_mean) for g in g_series[mask]], dtype=float)

    return {
        "category": "Graph",
        "name": "GKT-lite",
        "why": why,
        "y_true": y_true,
        "y_prob": y_prob,
        "test_rows": rows,
        "chosen_alpha": float(best_alpha),
    }
