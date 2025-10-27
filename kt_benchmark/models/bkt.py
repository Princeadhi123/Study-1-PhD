from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any, List

import time
import numpy as np
import pandas as pd

from .. import config
try:
    from joblib import Parallel, delayed  # type: ignore
except Exception:  # joblib optional; fall back to serial
    Parallel = None
    delayed = None


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
    """
    Return NEXT-step probabilities for a given sequence.
    For each observed y_t, we update the latent knowledge and emit the probability
    for the next step t+1, i.e., p(y_{t+1}=1 | history up to t).
    Output length = len(y_seq) - 1, aligned to y[1:].
    """
    L = params.L0
    probs_next = []
    for t, yt in enumerate(y_seq):
        # Update posterior with current observation
        if yt == 1:
            num = L * (1 - params.S)
            den = L * (1 - params.S) + (1 - L) * params.G
        else:
            num = L * params.S
            den = L * params.S + (1 - L) * (1 - params.G)
        den = float(np.clip(den, 1e-9, None))
        post = num / den
        L = post + (1 - post) * params.T
        # Predict next-step probability after the update (except we don't know y_{t+1} yet)
        p_next = L * (1 - params.S) + (1 - L) * params.G
        probs_next.append(float(np.clip(p_next, 1e-6, 1 - 1e-6)))
    # The last emitted p_next corresponds to after observing the final y_L; there is no y_{L+1}
    # Drop the last probability to align to y[1:]
    if len(probs_next) > 0:
        probs_next = probs_next[:-1]
    return np.array(probs_next, dtype=float)


def run(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
    why = (
        "Standard BKT is the canonical Bayesian model for KT with interpretable parameters (L0, learn, guess, slip). "
        "We fit per-skill (group) using a coarse grid over parameters for robustness and speed."
    )
    if config.COL_GROUP not in df.columns or config.COL_RESP not in df.columns or config.COL_ID not in df.columns:
        return {"category": "Bayesian", "name": "BKT", "why": why, "error": "Required columns missing (group/response/ID)"}

    # Fit params on train only (respect time budget)
    train_df = df.iloc[train_idx].copy()
    train_df = train_df.dropna(subset=[config.COL_RESP])
    use_per_group: bool = getattr(config, "BKT_PER_GROUP", True)
    group_params: Dict[str, BKTParams] = {}
    global_params: BKTParams | None = None
    start_time = time.perf_counter()
    budget = getattr(config, "TRAIN_TIME_BUDGET_S", None)

    if use_per_group and (config.COL_GROUP in train_df.columns):
        groups = list(train_df.groupby(config.COL_GROUP, dropna=False))
        g_labels = [str(g) for g, _ in groups]
        n_jobs = int(getattr(config, "BKT_N_JOBS", -1))
        if Parallel is not None and n_jobs != 0:
            params_list = Parallel(n_jobs=n_jobs)(delayed(_fit_group_params)(sub) for _, sub in groups)
        else:
            params_list = [_fit_group_params(sub) for _, sub in groups]
        group_params = {g: p for g, p in zip(g_labels, params_list)}
    else:
        # Fit a single global BKT
        # Uses entire train_df sequences by student
        global_params = _fit_group_params(train_df)

    # Predict on test: sequentially within each student per group (NEXT item)
    test_df = df.iloc[test_idx].copy()
    test_df = test_df.sort_values([config.COL_ID, config.COL_GROUP, config.COL_ORDER])

    y_true_list: List[int] = []
    y_prob_list: List[float] = []
    row_idx_list: List[int] = []

    for (sid, g), sub in test_df.groupby([config.COL_ID, config.COL_GROUP], dropna=False):
        y = pd.to_numeric(sub[config.COL_RESP], errors="coerce")
        mask = (y == 0) | (y == 1)
        y = y.where(mask).dropna().astype(int)
        # Need at least 2 steps to form a next-item target
        if len(y) < 2:
            continue
        if use_per_group:
            params = group_params.get(str(g), BKTParams(0.1, 0.1, 0.1, 0.1))
        else:
            params = global_params or BKTParams(0.1, 0.1, 0.1, 0.1)
        probs = _predict_sequence_probs(y.values, params)  # length = len(y) - 1
        # Align to next-step labels y[1:] and corresponding row indices
        y_next = y.values[1:]
        rows_next = sub.index[mask][1:]
        y_true_list.extend(y_next.tolist())
        y_prob_list.extend(probs.tolist())
        row_idx_list.extend(rows_next.tolist())

    if not y_true_list:
        return {"category": "Bayesian", "name": "BKT", "why": why, "error": "No valid test sequences"}

    return {
        "category": "Bayesian",
        "name": "BKT",
        "why": why,
        "y_true": np.array(y_true_list),
        "y_prob": np.array(y_prob_list),
        "test_rows": np.array(row_idx_list, dtype=int),
    }
