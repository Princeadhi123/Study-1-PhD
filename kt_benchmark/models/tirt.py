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
        "TIRT-lite augments item effects with a simple temporal progress covariate (z-scored within-student time index). "
        "This emulates temporal Item Response Theory by allowing ability to drift over time."
    )
    if config.COL_RESP not in df.columns:
        return {"category": "Temporal/Sequential", "name": "TIRT-lite", "why": why, "error": "response column missing"}

    # Features: item and time_index_z only (avoid leakage from group if you want pure temporal effect)
    # We'll include group to stabilize if present.
    X = prepare_tabular_features(df, use_item=True, use_group=True, use_sex=False, use_time=True)
    y = pd.to_numeric(df[config.COL_RESP], errors="coerce")
    mask = y.isin([0, 1])

    tr = np.intersect1d(train_idx, np.where(mask)[0])
    te = np.intersect1d(test_idx, np.where(mask)[0])
    if tr.size == 0 or te.size == 0:
        return {"category": "Temporal/Sequential", "name": "TIRT-lite", "why": why, "error": "no valid train/test rows"}

    X_tr = X.iloc[tr].values
    X_te = X.iloc[te].values
    y_tr = y.iloc[tr].astype(int).values
    y_te = y.iloc[te].astype(int).values

    scaler = StandardScaler(with_mean=False)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    clf = LogisticRegression(max_iter=getattr(config, "TIRT_MAX_ITER", 1000), class_weight="balanced", solver="saga", penalty="l2", random_state=config.RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    y_prob = clf.predict_proba(X_te)[:, 1]

    return {
        "category": "Temporal/Sequential",
        "name": "TIRT-lite",
        "why": why,
        "y_true": y_te,
        "y_prob": y_prob,
        "test_rows": te,
    }
