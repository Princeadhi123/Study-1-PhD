from __future__ import annotations

from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from .. import config
from ..utils import student_sequences, one_hot


def run(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
    why = (
        "CLKT-lite: learn item embeddings via contrastive pretraining from train sequences (positive = co-occurring items), "
        "then fine-tune a logistic regression using learned embeddings + metadata to predict correctness."
    )
    # Torch is optional
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        return {"category": "Contrastive/Self-supervised", "name": "CLKT-lite", "why": why, "error": f"torch not available: {e}"}

    item_col = config.COL_ITEM if config.COL_ITEM in df.columns else config.COL_ITEM_INDEX if config.COL_ITEM_INDEX in df.columns else None
    if item_col is None or config.COL_RESP not in df.columns or config.COL_ID not in df.columns:
        return {"category": "Contrastive/Self-supervised", "name": "CLKT-lite", "why": why, "error": "Required columns missing (item/item_index, response, ID)"}

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Build vocabulary from training items only
    items_train = train_df[item_col].astype(str).unique().tolist()
    it2ix = {it: i for i, it in enumerate(items_train)}
    n_items = len(it2ix)
    if n_items < 2:
        return {"category": "Contrastive/Self-supervised", "name": "CLKT-lite", "why": why, "error": "Not enough items in training"}

    # Build positive (anchor, pos) pairs from sliding windows in student sequences
    seq_map = student_sequences(train_df[[config.COL_ID, item_col, config.COL_ORDER]].dropna(subset=[item_col]))
    window = 3
    positives: List[Tuple[int, int]] = []
    for _, sub in seq_map.items():
        it = sub.sort_values(config.COL_ORDER)[item_col].astype(str).map(it2ix).dropna().astype(int).values
        L = len(it)
        for i in range(L):
            for j in range(i + 1, min(L, i + 1 + window)):
                a, b = int(it[i]), int(it[j])
                if a != b:
                    positives.append((a, b))
                    positives.append((b, a))
    if not positives:
        return {"category": "Contrastive/Self-supervised", "name": "CLKT-lite", "why": why, "error": "No positive pairs formed"}

    # Contrastive model: learn item embeddings with triplet loss
    dim = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = nn.Embedding(n_items, dim).to(device)
    loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
    opt = torch.optim.Adam(emb.parameters(), lr=1e-2)

    def sample_batch(batch_size: int = 256):
        # sample positives, and random negatives different from anchor
        idx = np.random.randint(0, len(positives), size=batch_size)
        A = []
        P = []
        N = []
        for k in idx:
            a, p = positives[k]
            n = a
            # sample negative not equal to anchor and not equal to positive
            while n == a or n == p:
                n = np.random.randint(0, n_items)
            A.append(a)
            P.append(p)
            N.append(n)
        return torch.tensor(A, dtype=torch.long, device=device), torch.tensor(P, dtype=torch.long, device=device), torch.tensor(N, dtype=torch.long, device=device)

    emb.train()
    steps = 400  # small pretraining
    for _ in range(steps):
        a, p, n = sample_batch(256)
        va, vp, vn = emb(a), emb(p), emb(n)
        loss = loss_fn(va, vp, vn)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Extract embeddings
    with torch.no_grad():
        item_emb = emb.weight.detach().cpu().numpy().astype(np.float32)  # [n_items, dim]

    # Build supervised features using learned item embedding + group/sex/time
    def build_features(frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for it in frame[item_col].astype(str).values:
            j = it2ix.get(str(it), None)
            if j is None:
                rows.append(np.zeros(dim, dtype=np.float32))
            else:
                rows.append(item_emb[j])
        X_emb = pd.DataFrame(rows, index=frame.index, columns=[f"emb_{i}" for i in range(dim)])
        feats = [X_emb]
        if config.COL_GROUP in frame.columns:
            feats.append(one_hot(frame[config.COL_GROUP]))
        if config.COL_SEX in frame.columns:
            feats.append(one_hot(frame[config.COL_SEX]))
        # simple time index z-score within frame to keep consistent
        if config.COL_ORDER in frame.columns and config.COL_ID in frame.columns:
            tmp = frame[[config.COL_ID, config.COL_ORDER]].copy()
            tmp = tmp.sort_values([config.COL_ID, config.COL_ORDER])
            tmp["time_index"] = tmp.groupby(config.COL_ID).cumcount() + 1
            ti = (tmp["time_index"] - tmp["time_index"].mean()) / (tmp["time_index"].std(ddof=1) + 1e-9)
            feats.append(pd.DataFrame({"time_index_z": ti.values}, index=frame.index))
        return pd.concat(feats, axis=1)

    X_tr = build_features(train_df)
    X_te = build_features(test_df)

    y_tr = pd.to_numeric(train_df[config.COL_RESP], errors="coerce")
    y_te = pd.to_numeric(test_df[config.COL_RESP], errors="coerce")

    mask_tr = y_tr.isin([0, 1])
    mask_te = y_te.isin([0, 1])
    if not mask_tr.any() or not mask_te.any():
        return {"category": "Contrastive/Self-supervised", "name": "CLKT-lite", "why": why, "error": "No valid binary rows"}

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr = scaler.fit_transform(X_tr[mask_tr].values)
    Xte = scaler.transform(X_te[mask_te].values)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=config.RANDOM_STATE)
    clf.fit(Xtr, y_tr[mask_tr].astype(int).values)
    y_prob = np.full(shape=len(test_df), fill_value=np.nan, dtype=float)
    y_prob[mask_te.values] = clf.predict_proba(Xte)[:, 1]

    return {
        "category": "Contrastive/Self-supervised",
        "name": "CLKT-lite",
        "why": why,
        "y_true": y_te.values,
        "y_prob": y_prob,
        "test_rows": test_idx,
    }
