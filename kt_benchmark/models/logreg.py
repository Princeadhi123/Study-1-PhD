from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .. import config
from ..utils import prepare_tabular_features


def run(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
    why = (
        "Logistic Regression is a strong, transparent ML baseline for binary correctness using tabular features. "
        "It is fast, robust to high-dimensional one-hot features, and provides calibrated probabilities."
    )
    if config.COL_RESP not in df.columns:
        return {"category": "Machine Learning", "name": "LogisticRegression", "why": why, "error": "response column missing"}

    X = prepare_tabular_features(df, use_item=True, use_group=True, use_sex=True, use_time=False)
    y = pd.to_numeric(df[config.COL_RESP], errors="coerce")
    mask = y.isin([0, 1])

    tr = np.intersect1d(train_idx, np.where(mask)[0])
    te = np.intersect1d(test_idx, np.where(mask)[0])
    if tr.size == 0 or te.size == 0:
        return {"category": "Machine Learning", "name": "LogisticRegression", "why": why, "error": "no valid train/test rows"}

    X_tr = X.iloc[tr].values
    X_te = X.iloc[te].values
    y_tr = y.iloc[tr].astype(int).values
    y_te = y.iloc[te].astype(int).values

    scaler = StandardScaler(with_mean=False)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="saga", penalty="l2", random_state=config.RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    y_prob = clf.predict_proba(X_te)[:, 1]

    return {
        "category": "Machine Learning",
        "name": "LogisticRegression",
        "why": why,
        "y_true": y_te,
        "y_prob": y_prob,
        "test_rows": te,
    }
