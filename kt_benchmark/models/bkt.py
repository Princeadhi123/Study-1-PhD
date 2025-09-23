from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd

from .. import config


@dataclass
class BKTParams:
    L0: float  # initial knowledge
    T: float   # learning probability
    G: float   # guess probability
    S: float   # slip probability


# Simple coarse grid to avoid complex EM; robust and fast (configurable)
GRID_L0 = getattr(config, "BKT_GRID_L0", [0.05, 0.10, 0.20, 0.30])
GRID_T = getattr(config, "BKT_GRID_T", [0.05, 0.10, 0.20])
GRID_G = getattr(config, "BKT_GRID_G", [0.05, 0.10, 0.20])
GRID_S = getattr(config, "BKT_GRID_S", [0.05, 0.10, 0.20])


def _ll_for_group(seq_list: List[np.ndarray], p: BKTParams) -> float:
    ll = 0.0
    for y in seq_list:
        if y.size == 0:
            continue
        L = p.L0
        for t in range(y.size):
            # Predictive correctness probability given latent L
            p_correct = L * (1 - p.S) + (1 - L) * p.G
            yt = int(y[t])
            pt = p_correct if yt == 1 else (L * p.S + (1 - L) * (1 - p.G))
            pt = float(np.clip(pt, 1e-6, 1 - 1e-6))
            ll += np.log(pt)
            # Posterior update then learning
            if yt == 1:
                num = L * (1 - p.S)
                den = L * (1 - p.S) + (1 - L) * p.G
            else:
                num = L * p.S
                den = L * p.S + (1 - L) * (1 - p.G)
            den = float(np.clip(den, 1e-9, None))
            post = num / den
            L = post + (1 - post) * p.T
    return float(ll)


def _fit_group_params(group_df: pd.DataFrame) -> BKTParams:
    # Build per-student sequences for this group
    seqs: List[np.ndarray] = []
    for sid, sub in group_df.sort_values([config.COL_ID, config.COL_ORDER]).groupby(config.COL_ID, dropna=False):
        y = pd.to_numeric(sub[config.COL_RESP], errors="coerce")
        y = y[(y == 0) | (y == 1)].astype(int).values
        if y.size:
            seqs.append(y)
    if not seqs:
        return BKTParams(0.1, 0.1, 0.1, 0.1)

    best = None
    best_ll = -np.inf
    for L0 in GRID_L0:
        for T in GRID_T:
            for G in GRID_G:
                for S in GRID_S:
                    p = BKTParams(L0, T, G, S)
                    ll = _ll_for_group(seqs, p)
                    if ll > best_ll:
                        best_ll, best = ll, p
    return best if best is not None else BKTParams(0.1, 0.1, 0.1, 0.1)


def _predict_sequence_probs(y_seq: np.ndarray, params: BKTParams) -> np.ndarray:
    # Produce per-step prediction BEFORE seeing y_t
    L = params.L0
    probs = []
    for yt in y_seq:
        p_correct = L * (1 - params.S) + (1 - L) * params.G
        probs.append(p_correct)
        # After observing yt, update L for next step
        if yt == 1:
            num = L * (1 - params.S)
            den = L * (1 - params.S) + (1 - L) * params.G
        else:
            num = L * params.S
            den = L * params.S + (1 - L) * (1 - params.G)
        den = float(np.clip(den, 1e-9, None))
        post = num / den
        L = post + (1 - post) * params.T
    return np.array(probs, dtype=float)


def run(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
    why = (
        "Standard BKT is the canonical Bayesian model for KT with interpretable parameters (L0, learn, guess, slip). "
        "We fit per-skill (group) using a coarse grid over parameters for robustness and speed."
    )
    if config.COL_GROUP not in df.columns or config.COL_RESP not in df.columns or config.COL_ID not in df.columns:
        return {"category": "Bayesian", "name": "BKT", "why": why, "error": "Required columns missing (group/response/ID)"}

    # Fit per-group params on train only
    train_df = df.iloc[train_idx].copy()
    train_df = train_df.dropna(subset=[config.COL_RESP])
    group_params: Dict[str, BKTParams] = {}
    for g, sub in train_df.groupby(config.COL_GROUP, dropna=False):
        p = _fit_group_params(sub)
        group_params[str(g)] = p

    # Predict on test: sequentially within each student per group
    test_df = df.iloc[test_idx].copy()
    test_df = test_df.sort_values([config.COL_ID, config.COL_GROUP, config.COL_ORDER])

    y_true_list: List[int] = []
    y_prob_list: List[float] = []

    for (sid, g), sub in test_df.groupby([config.COL_ID, config.COL_GROUP], dropna=False):
        y = pd.to_numeric(sub[config.COL_RESP], errors="coerce")
        mask = (y == 0) | (y == 1)
        y = y.where(mask).dropna().astype(int)
        if y.empty:
            continue
        params = group_params.get(str(g), BKTParams(0.1, 0.1, 0.1, 0.1))
        probs = _predict_sequence_probs(y.values, params)
        y_true_list.extend(y.values.tolist())
        y_prob_list.extend(probs.tolist())

    if not y_true_list:
        return {"category": "Bayesian", "name": "BKT", "why": why, "error": "No valid test sequences"}

    return {
        "category": "Bayesian",
        "name": "BKT",
        "why": why,
        "y_true": np.array(y_true_list),
        "y_prob": np.array(y_prob_list),
    }
