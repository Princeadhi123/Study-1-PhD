from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .. import config
from ..utils import prepare_tabular_features_sparse_split


def run(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
    why = (
        "Logistic Regression is a strong, transparent ML baseline for binary correctness using tabular features. "
        "It is fast, robust to high-dimensional one-hot features, and provides calibrated probabilities."
    )
    if config.COL_RESP not in df.columns:
        return {"category": "Machine Learning", "name": "LogisticRegression", "why": why, "error": "response column missing"}

    X_tr, X_te, _ = prepare_tabular_features_sparse_split(
        df,
        train_idx=train_idx,
        test_idx=test_idx,
        use_item=True,
        use_group=getattr(config, "LINEAR_USE_GROUP", True),
        use_sex=True,
        use_time=False,
    )
    y = pd.to_numeric(df[config.COL_RESP], errors="coerce")
    mask = y.isin([0, 1])

    tr = np.intersect1d(train_idx, np.where(mask)[0])
    te = np.intersect1d(test_idx, np.where(mask)[0])
    if tr.size == 0 or te.size == 0:
        return {"category": "Machine Learning", "name": "LogisticRegression", "why": why, "error": "no valid train/test rows"}

    # Align with valid mask selections
    X_tr = X_tr[np.searchsorted(train_idx, tr)]
    X_te = X_te[np.searchsorted(test_idx, te)]
    y_tr = y.iloc[tr].astype(int).values
    y_te = y.iloc[te].astype(int).values

    clf = LogisticRegression(
        max_iter=getattr(config, "LOGREG_MAX_ITER", 1000),
        tol=getattr(config, "LOGREG_TOL", 1e-3),
        class_weight="balanced",
        solver="saga",
        penalty="l2",
        random_state=config.RANDOM_STATE,
    )
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
