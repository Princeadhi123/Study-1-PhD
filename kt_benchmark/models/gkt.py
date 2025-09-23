from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from .. import config


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

    # Baseline group difficulty from train
    grp_mean = train_df.groupby(config.COL_GROUP)[config.COL_RESP].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
    global_mean = float(pd.to_numeric(train_df[config.COL_RESP], errors="coerce").mean()) if not train_df.empty else 0.5

    # Build adjacency and compute neighbor-smoothed difficulty
    adj = _build_group_graph(train_df)
    smoothed: Dict[str, float] = {}
    alpha = float(getattr(config, "GKT_SMOOTH_ALPHA", 0.7))  # weight for own group difficulty
    for g in grp_mean.index.astype(str):
        p_g = float(grp_mean.get(g, global_mean))
        if not adj.empty and g in adj.index:
            neigh = adj.loc[g]
            p_nei = 0.0
            for h, w in neigh.items():
                if w > 0:
                    p_nei += w * float(grp_mean.get(h, global_mean))
            p_smooth = alpha * p_g + (1 - alpha) * p_nei
        else:
            p_smooth = p_g
        smoothed[g] = float(np.clip(p_smooth, 1e-3, 1 - 1e-3))

    # Predict for test rows by group lookup
    g_series = test_df[config.COL_GROUP].astype(str) if config.COL_GROUP in test_df.columns else pd.Series([], dtype=str)
    y_true = pd.to_numeric(test_df[config.COL_RESP], errors="coerce").values
    y_prob = np.array([smoothed.get(g, global_mean) for g in g_series], dtype=float)

    if y_prob.size == 0:
        return {"category": "Graph", "name": "GKT-lite", "why": why, "error": "No test rows"}

    return {
        "category": "Graph",
        "name": "GKT-lite",
        "why": why,
        "y_true": y_true,
        "y_prob": y_prob,
    }
