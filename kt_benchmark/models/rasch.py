from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd

from .. import config


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class RaschResult:
    theta_by_user: Dict[str, float]
    beta_by_item: Dict[str, float]
    train_loss: float


class Rasch1PL:
    def __init__(self, max_iter: int = 30, inner_iter: int = 3):
        self.max_iter = max_iter
        self.inner_iter = inner_iter
        self.theta_by_user: Dict[str, float] = {}
        self.beta_by_item: Dict[str, float] = {}
        self.fitted_ = False

    def fit(self, df: pd.DataFrame, train_idx: np.ndarray) -> RaschResult:
        # Require ID, item (either name or index), response
        if config.COL_RESP not in df.columns:
            raise ValueError("response column missing")
        if config.COL_ID not in df.columns:
            raise ValueError("ID column missing")
        item_col = config.COL_ITEM if config.COL_ITEM in df.columns else config.COL_ITEM_INDEX if config.COL_ITEM_INDEX in df.columns else None
        if item_col is None:
            raise ValueError("item or item_index column required for Rasch")
        d = df.iloc[train_idx]
        d = d[[config.COL_ID, item_col, config.COL_RESP]].dropna()
        d[config.COL_RESP] = pd.to_numeric(d[config.COL_RESP], errors="coerce")
        d = d[(d[config.COL_RESP] == 0) | (d[config.COL_RESP] == 1)]
        if d.empty:
            raise ValueError("no valid binary responses for Rasch")

        users = d[config.COL_ID].astype(str).unique().tolist()
        items = d[item_col].astype(str).unique().tolist()
        u_index = {u: i for i, u in enumerate(users)}
        it_index = {it: j for j, it in enumerate(items)}

        u = d[config.COL_ID].astype(str).map(u_index).values
        v = d[item_col].astype(str).map(it_index).values
        y = d[config.COL_RESP].astype(int).values

        n_users = len(users)
        n_items = len(items)

        theta = np.zeros(n_users, dtype=float)
        # init beta by item difficulty: logit of item mean
        item_means = pd.Series(y).groupby(v).mean()
        beta = np.zeros(n_items, dtype=float)
        for j in range(n_items):
            p = float(item_means.get(j, 0.5))
            p = min(max(p, 1e-3), 1 - 1e-3)
            beta[j] = -math.log(p / (1 - p))  # inverse of ability-difficulty logit
        beta -= beta.mean()  # center identifiability

        # Precompute indices per user/item
        idx_by_user = [np.where(u == i)[0] for i in range(n_users)]
        idx_by_item = [np.where(v == j)[0] for j in range(n_items)]

        def nll() -> float:
            x = theta[u] - beta[v]
            p = _sigmoid(x)
            # avoid log 0
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

        last_loss = nll()
        for it in range(self.max_iter):
            # Update theta per user via Newton steps
            for i in range(n_users):
                idx = idx_by_user[i]
                if idx.size == 0:
                    continue
                yi = y[idx]
                bj = beta[v[idx]]
                ti = theta[i]
                for _ in range(self.inner_iter):
                    p = _sigmoid(ti - bj)
                    g = np.sum(yi - p)
                    h = -np.sum(p * (1 - p)) - 1e-6
                    step = g / h
                    step = float(np.clip(step, -1.0, 1.0))
                    ti -= step
                theta[i] = ti

            # Update beta per item via Newton steps
            for j in range(n_items):
                idx = idx_by_item[j]
                if idx.size == 0:
                    continue
                yj = y[idx]
                ti = theta[u[idx]]
                bj = beta[j]
                for _ in range(self.inner_iter):
                    p = _sigmoid(ti - bj)
                    g = np.sum(p - yj)
                    h = np.sum(p * (1 - p)) + 1e-6
                    step = g / h
                    step = float(np.clip(step, -1.0, 1.0))
                    bj -= step
                beta[j] = bj

            # Center beta to mean 0, shift theta accordingly
            b_mean = beta.mean()
            beta -= b_mean
            theta -= b_mean

            cur_loss = nll()
            if abs(last_loss - cur_loss) < 1e-4:
                break
            last_loss = cur_loss

        self.theta_by_user = {u_id: float(theta[i]) for u_id, i in u_index.items()}
        self.beta_by_item = {it_id: float(beta[j]) for it_id, j in it_index.items()}
        self.fitted_ = True
        return RaschResult(self.theta_by_user, self.beta_by_item, last_loss)

    def predict_proba(self, df: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
        assert self.fitted_, "Call fit() first"
        item_col = config.COL_ITEM if config.COL_ITEM in df.columns else config.COL_ITEM_INDEX if config.COL_ITEM_INDEX in df.columns else None
        u_series = df.loc[rows, config.COL_ID].astype(str)
        it_series = df.loc[rows, item_col].astype(str) if item_col else pd.Series(["" for _ in rows])
        thetas = np.array([self.theta_by_user.get(uid, 0.0) for uid in u_series])
        betas = np.array([self.beta_by_item.get(iid, 0.0) for iid in it_series])
        p = _sigmoid(thetas - betas)
        return p


def run(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
    why = (
        "Rasch (1PL) is the simplest IRT model: a single ability per student and difficulty per item, "
        "with logistic link. It provides an interpretable baseline in the psychometric family."
    )
    try:
        model = Rasch1PL(max_iter=getattr(config, "RASCH_MAX_ITER", 30), inner_iter=getattr(config, "RASCH_INNER_ITER", 3))
        model.fit(df, train_idx)
        y_true = pd.to_numeric(df.iloc[test_idx][config.COL_RESP], errors="coerce").values
        y_prob = model.predict_proba(df, test_idx)
        return {
            "category": "Psychometric (IRT)",
            "name": "Rasch1PL",
            "why": why,
            "y_true": y_true,
            "y_prob": y_prob,
        }
    except Exception as e:
        return {
            "category": "Psychometric (IRT)",
            "name": "Rasch1PL",
            "why": why,
            "error": str(e),
        }
