from __future__ import annotations

from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from .. import config
from ..utils import student_sequences


def run(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
    why = (
        "DKT (LSTM-based) is the classic deep learning KT model that captures sequential dependencies. "
        "Here we use a minimal version predicting current response from the running hidden state over item embeddings."
    )
    # Torch is optional
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except Exception as e:
        return {"category": "Deep Learning", "name": "DKT (minimal)", "why": why, "error": f"torch not available: {e}"}

    item_col = config.COL_ITEM if config.COL_ITEM in df.columns else config.COL_ITEM_INDEX if config.COL_ITEM_INDEX in df.columns else None
    if item_col is None or config.COL_RESP not in df.columns or config.COL_ID not in df.columns:
        return {"category": "Deep Learning", "name": "DKT (minimal)", "why": why, "error": "Required columns missing (item/item_index, response, ID)"}

    # Build train/test student sets
    train_students = set(df.iloc[train_idx][config.COL_ID].astype(str).unique().tolist())
    test_students = set(df.iloc[test_idx][config.COL_ID].astype(str).unique().tolist())

    # Vocabulary from training items only (avoid test leakage)
    items_train = df.iloc[train_idx][item_col].astype(str).unique().tolist()
    it2ix = {it: i + 1 for i, it in enumerate(items_train)}  # 0 = PAD
    pad_idx = 0
    vocab_size = len(it2ix) + 1

    # Build sequences per student
    seq_map = student_sequences(df[[config.COL_ID, item_col, config.COL_ORDER, config.COL_RESP]].dropna(subset=[config.COL_RESP]))

    def build_sequences(student_ids: List[str]) -> List[Tuple[np.ndarray, np.ndarray]]:
        seqs: List[Tuple[np.ndarray, np.ndarray]] = []
        for sid in student_ids:
            key = (sid,)
            if key not in seq_map:
                continue
            sub = seq_map[key].sort_values(config.COL_ORDER)
            items_all = sub[item_col].astype(str).map(lambda x: it2ix.get(x, 0)).values
            y_series = pd.to_numeric(sub[config.COL_RESP], errors="coerce")
            mask = y_series.isin([0, 1]).values
            if mask.sum() == 0:
                continue
            items = items_all[mask]
            y = y_series[mask].astype(int).values
            # Keep sequences with at least 2 steps
            if len(y) >= 2:  # need some signal
                seqs.append((items, y))
        return seqs

    train_seqs = build_sequences(sorted(train_students))
    test_seqs = build_sequences(sorted(test_students))
    if not train_seqs or not test_seqs:
        return {"category": "Deep Learning", "name": "DKT (minimal)", "why": why, "error": "No valid train/test sequences"}

    # Pad and batch via DataLoader
    class SeqDataset(torch.utils.data.Dataset):
        def __init__(self, seqs):
            self.seqs = seqs
        def __len__(self):
            return len(self.seqs)
        def __getitem__(self, idx):
            return self.seqs[idx]

    def collate(batch):
        # batch: list of (items, y)
        lengths = [len(x[0]) for x in batch]
        max_len = max(lengths)
        it_pad = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
        y_pad = torch.full((len(batch), max_len), -1, dtype=torch.long)  # -1 to mask loss
        for i, (it, y) in enumerate(batch):
            L = len(it)
            it_pad[i, :L] = torch.tensor(it, dtype=torch.long)
            y_pad[i, :L] = torch.tensor(y, dtype=torch.long)
        return it_pad, y_pad, torch.tensor(lengths, dtype=torch.long)

    train_loader = DataLoader(SeqDataset(train_seqs), batch_size=32, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(SeqDataset(test_seqs), batch_size=32, shuffle=False, collate_fn=collate)

    class DKT(nn.Module):
        def __init__(self, vocab: int, emb: int = 32, hid: int = 64):
            super().__init__()
            self.emb = nn.Embedding(vocab, emb, padding_idx=pad_idx)
            self.lstm = nn.LSTM(emb, hid, batch_first=True)
            self.out = nn.Linear(hid, 1)
        def forward(self, it, lengths):
            x = self.emb(it)
            # Pack padded for efficiency
            packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            h, _ = self.lstm(packed)
            h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
            logits = self.out(h).squeeze(-1)
            return logits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DKT(vocab=vocab_size).to(device)
    crit = nn.BCEWithLogitsLoss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    model.train()
    for epoch in range(max(1, config.EPOCHS_DKT)):
        for it_pad, y_pad, lengths in train_loader:
            it_pad = it_pad.to(device)
            y_pad = y_pad.to(device)
            lengths = lengths.to(device)
            logits = model(it_pad, lengths)
            y_float = (y_pad == 1).float()
            mask = (y_pad != -1).float()
            loss = crit(logits, y_float)
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Evaluate: collect all per-step predictions with valid labels
    model.eval()
    y_true_all: List[int] = []
    y_prob_all: List[float] = []
    with torch.no_grad():
        for it_pad, y_pad, lengths in test_loader:
            it_pad = it_pad.to(device)
            y_pad = y_pad.to(device)
            lengths = lengths.to(device)
            logits = model(it_pad, lengths)
            probs = torch.sigmoid(logits)
            mask = (y_pad != -1)
            y_true = (y_pad[mask]).float().cpu().numpy()
            y_prob = (probs[mask]).float().cpu().numpy()
            if y_true.size:
                y_true_all.extend(y_true.tolist())
                y_prob_all.extend(y_prob.tolist())

    if not y_true_all:
        return {"category": "Deep Learning", "name": "DKT (minimal)", "why": why, "error": "No eval points"}

    return {
        "category": "Deep Learning",
        "name": "DKT (minimal)",
        "why": why,
        "y_true": np.array(y_true_all),
        "y_prob": np.array(y_prob_all),
    }
