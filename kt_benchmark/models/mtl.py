from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd

from .. import config
from ..utils import prepare_tabular_features


def run(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
    why = (
        "FKT-lite: a small shared MLP predicts correctness (primary) and response time (aux) jointly. "
        "Multi-task learning can improve representation quality and calibration by leveraging correlated auxiliary signals."
    )
    # Optional torch dependency
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        return {"category": "Multi-task", "name": "FKT-lite", "why": why, "error": f"torch not available: {e}"}

    if config.COL_RESP not in df.columns:
        return {"category": "Multi-task", "name": "FKT-lite", "why": why, "error": "response column missing"}

    # Build tabular features, then drop log_rt column to avoid leakage into RT head
    X = prepare_tabular_features(df, use_item=True, use_group=True, use_sex=True, use_time=True)
    if "log_rt" in X.columns:
        X = X.drop(columns=["log_rt"])  # prevent trivial RT prediction

    y_corr = pd.to_numeric(df[config.COL_RESP], errors="coerce")
    mask_corr = y_corr.isin([0, 1])

    y_rt = pd.to_numeric(df[config.COL_RT], errors="coerce") if config.COL_RT in df.columns else pd.Series(np.nan, index=df.index)
    mask_rt = y_rt.notna()

    tr = np.intersect1d(train_idx, np.where(mask_corr)[0])
    te = np.intersect1d(test_idx, np.where(mask_corr)[0])
    if tr.size == 0 or te.size == 0:
        return {"category": "Multi-task", "name": "FKT-lite", "why": why, "error": "no valid train/test rows"}

    X_tr = X.iloc[tr].values.astype(np.float32)
    X_te = X.iloc[te].values.astype(np.float32)
    y_tr_corr = y_corr.iloc[tr].astype(int).values.astype(np.float32)
    y_te_corr = y_corr.iloc[te].astype(int).values.astype(np.float32)

    y_tr_rt = y_rt.iloc[tr].values.astype(np.float32)
    y_te_rt = y_rt.iloc[te].values.astype(np.float32)
    m_tr_rt = np.isfinite(y_tr_rt)
    m_te_rt = np.isfinite(y_te_rt)

    # Standardize features to zero mean/var
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # Torch dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt_corr = torch.tensor(y_tr_corr, dtype=torch.float32, device=device)
    yt_rt = torch.tensor(np.nan_to_num(y_tr_rt, nan=0.0), dtype=torch.float32, device=device)
    mt_rt = torch.tensor(m_tr_rt.astype(np.float32), dtype=torch.float32, device=device)

    class MTL(nn.Module):
        def __init__(self, d_in: int, d_hid: int = 64):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(d_in, d_hid), nn.ReLU(),
                nn.Linear(d_hid, d_hid), nn.ReLU(),
            )
            self.head_corr = nn.Sequential(nn.Linear(d_hid, 1))
            self.head_rt = nn.Sequential(nn.Linear(d_hid, 1))
        def forward(self, x):
            h = self.shared(x)
            logit = self.head_corr(h).squeeze(-1)
            rt = self.head_rt(h).squeeze(-1)
            return logit, rt

    model = MTL(d_in=X_tr.shape[1]).to(device)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    model.train()
    batch_size = 128
    n = Xt.shape[0]
    for epoch in range(max(1, config.EPOCHS_MTL)):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = Xt[idx]
            yb = yt_corr[idx]
            rb = yt_rt[idx]
            mb = mt_rt[idx]
            logit, rt_pred = model(xb)
            loss_corr = bce(logit, yb)
            loss_rt = (mse(rt_pred, rb) * mb).sum() / (mb.sum() + 1e-6)
            loss = loss_corr + 0.2 * loss_rt  # small weight on aux
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Eval
    Xte_t = torch.tensor(X_te, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        logit, rt_pred = model(Xte_t)
        y_prob = torch.sigmoid(logit).cpu().numpy()
        rt_pred_np = rt_pred.cpu().numpy()

    # Aux metrics where RT available
    if m_te_rt.any():
        rt_err = rt_pred_np[m_te_rt] - y_te_rt[m_te_rt]
        mae = float(np.mean(np.abs(rt_err)))
        rmse = float(np.sqrt(np.mean(rt_err**2)))
    else:
        mae = np.nan
        rmse = np.nan

    return {
        "category": "Multi-task",
        "name": "FKT-lite",
        "why": why,
        "y_true": y_te_corr,
        "y_prob": y_prob,
        "aux_mae_rt": mae,
        "aux_rmse_rt": rmse,
    }
