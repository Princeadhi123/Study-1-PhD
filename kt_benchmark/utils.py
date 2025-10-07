from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
try:
    from scipy import sparse as _sp
    _SP_AVAILABLE = True
except Exception:  # pragma: no cover
    _sp = None  # type: ignore
    _SP_AVAILABLE = False

from . import config


SEX_MAP = {
    # male
    "boy": "M", "b": "M", "male": "M", "m": "M", "man": "M",
    # female
    "girl": "F", "gir": "F", "g": "F", "female": "F", "f": "F", "woman": "F",
}


def normalize_sex(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower().str.replace(r"[^a-z]", "", regex=True)
    mapped = s.map(SEX_MAP)
    return mapped.fillna("U")


@dataclass
class Dataset:
    df: pd.DataFrame
    train_idx: np.ndarray
    test_idx: np.ndarray


def load_itemwise_df(path: Optional[str | pd.PathLike] = None) -> pd.DataFrame:
    path = str(path or config.INPUT_CSV)
    df = pd.read_csv(path)
    # Standardize columns
    df.columns = [c.strip() for c in df.columns]

    # Sex normalization (non-destructive inside memory)
    if config.COL_SEX in df.columns:
        df[config.COL_SEX] = normalize_sex(df[config.COL_SEX])

    # Coerce response to numeric 0/1 if possible
    if config.COL_RESP in df.columns:
        df[config.COL_RESP] = pd.to_numeric(df[config.COL_RESP], errors="coerce")
        # If values are not 0/1 but True/False or similar, map them
        if not set(np.unique(df[config.COL_RESP].dropna().values)).issubset({0, 1}):
            # Try common mappings
            m = {"true": 1, "false": 0, "yes": 1, "no": 0, "y": 1, "n": 0}
            nonnum = df[config.COL_RESP].copy()
            mask_nan = nonnum.isna()
            nonnum = nonnum.astype(str).str.lower().map(m)
            df.loc[mask_nan, config.COL_RESP] = nonnum
            df[config.COL_RESP] = pd.to_numeric(df[config.COL_RESP], errors="coerce")
        # Clip to {0,1}
        df[config.COL_RESP] = df[config.COL_RESP].clip(0, 1)

    # Coerce RT
    if config.COL_RT in df.columns:
        df[config.COL_RT] = pd.to_numeric(df[config.COL_RT], errors="coerce")

    # Ensure order for sequencing
    if config.COL_ORDER not in df.columns:
        # Fallback: within-student order by row
        df[config.COL_ORDER] = df.groupby(config.COL_ID).cumcount() + 1 if config.COL_ID in df.columns else np.arange(1, len(df) + 1)

    return df


def filter_min_events(df: pd.DataFrame, min_events: int) -> pd.DataFrame:
    if config.COL_ID not in df.columns:
        return df
    counts = df.groupby(config.COL_ID, dropna=False)[config.COL_ID].transform("size")
    return df.loc[counts >= min_events].reset_index(drop=True)


def train_test_split_by_student(df: pd.DataFrame, test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    if config.COL_ID not in df.columns:
        # Fallback: random split by rows
        idx = np.arange(len(df))
        tr, te = train_test_split(idx, test_size=test_size, random_state=random_state, shuffle=True)
        return tr, te
    students = df[config.COL_ID].astype(str).unique()
    tr_students, te_students = train_test_split(students, test_size=test_size, random_state=random_state, shuffle=True)
    tr_mask = df[config.COL_ID].astype(str).isin(tr_students).values
    te_mask = df[config.COL_ID].astype(str).isin(te_students).values
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask)[0]
    return tr_idx, te_idx


def add_time_index(df: pd.DataFrame) -> pd.DataFrame:
    if config.COL_ID not in df.columns:
        df["time_index"] = np.arange(len(df))
        return df
    df = df.sort_values([config.COL_ID, config.COL_ORDER]).copy()
    df["time_index"] = df.groupby(config.COL_ID).cumcount() + 1
    return df


def build_dataset(min_events: int | None = None) -> Dataset:
    df = load_itemwise_df()
    if min_events is None:
        min_events = config.MIN_STUDENT_EVENTS
    if min_events and min_events > 1:
        df = filter_min_events(df, min_events)
    df = add_time_index(df)
    tr_idx, te_idx = train_test_split_by_student(df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    return Dataset(df=df, train_idx=tr_idx, test_idx=te_idx)


# Feature builders for non-sequence models

def one_hot(series: pd.Series, prefix: str | None = None) -> pd.DataFrame:
    """
    Dense one-hot with stable, prefixed column names to avoid duplicates across
    different features (e.g., item vs group sharing the same category label).

    If prefix is None, use the Series' name when available; otherwise 'feat'.
    """
    pref = prefix if prefix is not None else (str(series.name) if series.name is not None else "feat")
    # Use smaller dtype to reduce memory when using dense dummies
    return pd.get_dummies(series.astype("category"), dummy_na=True, dtype=np.uint8, prefix=pref)


def prepare_tabular_features(df: pd.DataFrame, use_item: bool = True, use_group: bool = True, use_sex: bool = True, use_time: bool = True) -> pd.DataFrame:
    feats = []
    if use_item:
        if config.COL_ITEM in df.columns:
            feats.append(one_hot(df[config.COL_ITEM]))
        elif config.COL_ITEM_INDEX in df.columns:
            feats.append(one_hot(df[config.COL_ITEM_INDEX]))
    if use_group and config.COL_GROUP in df.columns:
        feats.append(one_hot(df[config.COL_GROUP]))
    if use_sex and config.COL_SEX in df.columns:
        feats.append(one_hot(df[config.COL_SEX]))
    if use_time and "time_index" in df.columns:
        ti = df["time_index"].astype(float)
        # scale
        ti = (ti - ti.mean()) / (ti.std(ddof=1) + 1e-9)
        feats.append(pd.DataFrame({"time_index_z": ti}))
    if config.COL_RT in df.columns:
        # add log RT as optional weak feature; robust to negatives/NaNs
        rts = pd.to_numeric(df[config.COL_RT], errors="coerce")
        # clip negatives to zero, then impute remaining NaNs with non-negative median
        rts = rts.clip(lower=0)
        med = rts[rts >= 0].median()
        rts = rts.fillna(0 if np.isnan(med) else med)
        # use log1p to avoid log(0) issues
        feats.append(pd.DataFrame({"log_rt": np.log1p(rts)}))
    if not feats:
        return pd.DataFrame(index=df.index)
    X = pd.concat(feats, axis=1)
    X.columns = [str(c) for c in X.columns]
    # Ensure no NaNs/Infs are passed to downstream models
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def _ohe_single_feature(values: pd.Series, feature_name: str):
    # scikit-learn >= 1.2 uses 'sparse_output' instead of 'sparse'
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    except TypeError:  # fallback for older versions
        enc = OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)
    X_part = enc.fit_transform(values.astype(str).to_frame())
    try:
        names = enc.get_feature_names_out([feature_name]).tolist()
    except AttributeError:
        # Older scikit-learn
        names = enc.get_feature_names([feature_name]).tolist()
    if _SP_AVAILABLE:
        return X_part.tocsr(), names
    else:
        # Return a dense numpy array fallback when SciPy is not available
        return X_part.toarray(), names


def prepare_tabular_features_sparse(
    df: pd.DataFrame,
    use_item: bool = True,
    use_group: bool = True,
    use_sex: bool = True,
    use_time: bool = True,
) -> tuple[_sp.csr_matrix, List[str]]:
    """
    Build a sparse design matrix for tabular models. This is functionally similar to
    prepare_tabular_features(), but returns a scipy.sparse CSR matrix and the list of
    feature names. Using sparse representations drastically reduces memory/time when
    item/group cardinalities are high (e.g., ASSISTments).
    """
    parts: List[object] = []
    names: List[str] = []

    # One-hot categorical parts
    if use_item:
        if config.COL_ITEM in df.columns:
            Xi, ni = _ohe_single_feature(df[config.COL_ITEM], config.COL_ITEM)
            parts.append(Xi)
            names.extend(ni)
        elif config.COL_ITEM_INDEX in df.columns:
            Xi, ni = _ohe_single_feature(df[config.COL_ITEM_INDEX].astype(str), config.COL_ITEM_INDEX)
            parts.append(Xi)
            names.extend(ni)

    if use_group and config.COL_GROUP in df.columns:
        Xg, ng = _ohe_single_feature(df[config.COL_GROUP], config.COL_GROUP)
        parts.append(Xg)
        names.extend(ng)

    if use_sex and config.COL_SEX in df.columns:
        Xs, ns = _ohe_single_feature(df[config.COL_SEX], config.COL_SEX)
        parts.append(Xs)
        names.extend(ns)

    # Continuous parts (as sparse columns)
    if use_time and "time_index" in df.columns:
        ti = df["time_index"].astype(float)
        ti = (ti - ti.mean()) / (ti.std(ddof=1) + 1e-9)
        vec = ti.to_numpy(dtype=np.float32).reshape(-1, 1)
        Xti = (_sp.csr_matrix(vec) if _SP_AVAILABLE else vec)
        parts.append(Xti)
        names.append("time_index_z")

    if config.COL_RT in df.columns:
        rts = pd.to_numeric(df[config.COL_RT], errors="coerce")
        rts = rts.clip(lower=0)
        med = rts[rts >= 0].median()
        rts = rts.fillna(0 if np.isnan(med) else med)
        vec = np.log1p(rts).to_numpy(dtype=np.float32).reshape(-1, 1)
        Xrt = (_sp.csr_matrix(vec) if _SP_AVAILABLE else vec)
        parts.append(Xrt)
        names.append("log_rt")

    if not parts:
        if _SP_AVAILABLE:
            return _sp.csr_matrix((len(df), 0), dtype=np.float32), []
        else:
            return np.zeros((len(df), 0), dtype=np.float32), []

    if _SP_AVAILABLE:
        X_sparse = _sp.hstack(parts, format="csr")
        return X_sparse, names
    else:
        # Dense fallback: concatenate numpy arrays
        dense_parts = [p if isinstance(p, np.ndarray) else p.toarray() for p in parts]  # type: ignore
        X_dense = np.concatenate(dense_parts, axis=1).astype(np.float32, copy=False)
        return X_dense, names


# Sequences for KT models

def sequences_by(df: pd.DataFrame, key_cols: List[str]) -> Dict[Tuple, pd.DataFrame]:
    d = {}
    df2 = df.sort_values(key_cols + [config.COL_ORDER]).copy()
    for keys, sub in df2.groupby(key_cols, dropna=False):
        d[keys if isinstance(keys, tuple) else (keys,)] = sub
    return d


def student_sequences(df: pd.DataFrame) -> Dict[Tuple[str], pd.DataFrame]:
    if config.COL_ID not in df.columns:
        return {(): df.copy()}
    df2 = df.copy()
    # Normalize ID to string so downstream lookups using str IDs match
    df2[config.COL_ID] = df2[config.COL_ID].astype(str)
    return sequences_by(df2, [config.COL_ID])


def student_group_sequences(df: pd.DataFrame) -> Dict[Tuple[str, str], pd.DataFrame]:
    if config.COL_ID not in df.columns or config.COL_GROUP not in df.columns:
        return {}
    return sequences_by(df, [config.COL_ID, config.COL_GROUP])


def next_step_row_indices(df: pd.DataFrame, row_positions: np.ndarray) -> np.ndarray:
    """
    Given a dataframe and a list/array of row positions (e.g., train_idx or test_idx),
    return the positions corresponding to the "next" steps within each student sequence.

    Concretely, for each student sorted by `config.COL_ORDER`, we drop the first event and
    keep rows 2..L. This aligns features at time t+1 with the label y_{t+1}, ensuring all
    models predict the next item.

    If student IDs or order are missing, we simply drop the first row of the provided
    positions as a conservative fallback.
    """
    try:
        pos = np.asarray(row_positions, dtype=int)
        if pos.size == 0:
            return pos
        if config.COL_ID not in df.columns or config.COL_ORDER not in df.columns:
            return pos[1:] if pos.size > 1 else np.array([], dtype=int)
        sub = df.iloc[pos][[config.COL_ID, config.COL_ORDER]].copy()
        sub[config.COL_ID] = sub[config.COL_ID].astype(str)
        sub_sorted = sub.sort_values([config.COL_ID, config.COL_ORDER])
        # Mark the first event per student; keep the rest
        is_first = ~sub_sorted[config.COL_ID].duplicated(keep="first")
        rows_next = sub_sorted.index[~is_first].to_numpy(dtype=int)
        return rows_next
    except Exception:
        # Fallback: drop first
        pos = np.asarray(row_positions, dtype=int)
        return pos[1:] if pos.size > 1 else np.array([], dtype=int)


def next_step_pairs_for_indices(df: pd.DataFrame, row_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For a given set of row positions, return aligned arrays (prev_rows, next_rows)
    containing consecutive within-student pairs (t, t+1), sorted by (ID, order).
    """
    pos = np.asarray(row_positions, dtype=int)
    if pos.size == 0:
        return pos, pos
    if config.COL_ID not in df.columns or config.COL_ORDER not in df.columns:
        return (pos[:-1] if pos.size > 1 else np.array([], dtype=int), pos[1:] if pos.size > 1 else np.array([], dtype=int))
    sub = df.iloc[pos][[config.COL_ID, config.COL_ORDER]].copy()
    sub[config.COL_ID] = sub[config.COL_ID].astype(str)
    sub = sub.sort_values([config.COL_ID, config.COL_ORDER])
    ids = sub[config.COL_ID].values
    idx_sorted = sub.index.values
    keep = (ids[:-1] == ids[1:])
    prev_rows = idx_sorted[:-1][keep]
    next_rows = idx_sorted[1:][keep]
    return prev_rows.astype(int), next_rows.astype(int)


def prepare_next_step_features_sparse_split(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    use_item: bool = True,
    use_group: bool = True,
    use_sex: bool = True,
    use_time: bool = True,
) -> tuple[object, object, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sparse features for next-step prediction.
    Features are taken from the NEXT rows (t+1) within each student, optionally including
    a numeric 'prev_correct' feature from the current row (t).

    Returns: X_tr, X_te, y_tr, y_te, rows_tr_next, rows_te_next
    """
    # Determine item column
    item_col = config.COL_ITEM if config.COL_ITEM in df.columns else (config.COL_ITEM_INDEX if config.COL_ITEM_INDEX in df.columns else None)
    prev_tr, next_tr = next_step_pairs_for_indices(df, train_idx)
    prev_te, next_te = next_step_pairs_for_indices(df, test_idx)

    # Labels at next rows
    y_tr_raw = pd.to_numeric(df.loc[next_tr, config.COL_RESP], errors="coerce") if config.COL_RESP in df.columns else pd.Series(np.nan, index=next_tr)
    y_te_raw = pd.to_numeric(df.loc[next_te, config.COL_RESP], errors="coerce") if config.COL_RESP in df.columns else pd.Series(np.nan, index=next_te)
    mtr = y_tr_raw.isin([0, 1]).values
    mte = y_te_raw.isin([0, 1]).values
    prev_tr, next_tr = prev_tr[mtr], next_tr[mtr]
    prev_te, next_te = prev_te[mte], next_te[mte]
    y_tr = y_tr_raw[mtr].astype(int).values
    y_te = y_te_raw[mte].astype(int).values

    parts_tr: list[object] = []
    parts_te: list[object] = []
    names: list[str] = []

    # One-hot for next-row item
    if use_item and item_col is not None:
        Xtr, Xte, nm = _ohe_fit_split(df.loc[next_tr, item_col].astype(str), df.loc[next_te, item_col].astype(str), item_col)
        parts_tr.append(Xtr); parts_te.append(Xte); names.extend(nm)

    # One-hot for next-row group
    if use_group and config.COL_GROUP in df.columns:
        Xtr, Xte, nm = _ohe_fit_split(df.loc[next_tr, config.COL_GROUP].astype(str), df.loc[next_te, config.COL_GROUP].astype(str), config.COL_GROUP)
        parts_tr.append(Xtr); parts_te.append(Xte); names.extend(nm)

    # One-hot sex (assumed constant per student but take next row for simplicity)
    if use_sex and config.COL_SEX in df.columns:
        Xtr, Xte, nm = _ohe_fit_split(df.loc[next_tr, config.COL_SEX].astype(str), df.loc[next_te, config.COL_SEX].astype(str), config.COL_SEX)
        parts_tr.append(Xtr); parts_te.append(Xte); names.extend(nm)

    # Time index z from next rows
    if use_time and "time_index" in df.columns:
        ti_tr = df.loc[next_tr, "time_index"].astype(float)
        mu = ti_tr.mean(); sd = ti_tr.std(ddof=1) + 1e-9
        ti_te = df.loc[next_te, "time_index"].astype(float)
        vtr = ((ti_tr - mu) / sd).to_numpy(dtype=np.float32).reshape(-1, 1)
        vte = ((ti_te - mu) / sd).to_numpy(dtype=np.float32).reshape(-1, 1)
        Xtr = (_sp.csr_matrix(vtr) if _SP_AVAILABLE else vtr)
        Xte = (_sp.csr_matrix(vte) if _SP_AVAILABLE else vte)
        parts_tr.append(Xtr); parts_te.append(Xte); names.append("time_index_z")

    # Prev correctness from prev rows (numeric)
    if config.COL_RESP in df.columns:
        prev_corr_tr = pd.to_numeric(df.loc[prev_tr, config.COL_RESP], errors="coerce").astype(float).to_numpy().reshape(-1, 1)
        prev_corr_te = pd.to_numeric(df.loc[prev_te, config.COL_RESP], errors="coerce").astype(float).to_numpy().reshape(-1, 1)
        prev_corr_tr = np.nan_to_num(prev_corr_tr, nan=0.0)
        prev_corr_te = np.nan_to_num(prev_corr_te, nan=0.0)
        Xtr = (_sp.csr_matrix(prev_corr_tr) if _SP_AVAILABLE else prev_corr_tr)
        Xte = (_sp.csr_matrix(prev_corr_te) if _SP_AVAILABLE else prev_corr_te)
        parts_tr.append(Xtr); parts_te.append(Xte); names.append("prev_correct")

    # Stack
    if _SP_AVAILABLE:
        X_tr = _sp.hstack(parts_tr, format="csr") if parts_tr else _sp.csr_matrix((len(next_tr), 0), dtype=np.float32)
        X_te = _sp.hstack(parts_te, format="csr") if parts_te else _sp.csr_matrix((len(next_te), 0), dtype=np.float32)
    else:
        X_tr = np.concatenate([p if isinstance(p, np.ndarray) else p.toarray() for p in parts_tr], axis=1) if parts_tr else np.zeros((len(next_tr), 0), dtype=np.float32)
        X_te = np.concatenate([p if isinstance(p, np.ndarray) else p.toarray() for p in parts_te], axis=1) if parts_te else np.zeros((len(next_te), 0), dtype=np.float32)

    return X_tr, X_te, y_tr, y_te, next_tr, next_te


def prepare_next_step_features_dense_split(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    use_item: bool = True,
    use_group: bool = True,
    use_sex: bool = True,
    use_time: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Dense DataFrame feature builder for next-step prediction using prepare_tabular_features
    on the NEXT rows. Adds a numeric 'prev_correct' column from the previous row.
    Returns: X_tr_df, X_te_df, y_tr, y_te, rows_tr_next, rows_te_next
    """
    prev_tr, next_tr = next_step_pairs_for_indices(df, train_idx)
    prev_te, next_te = next_step_pairs_for_indices(df, test_idx)

    y_tr_raw = pd.to_numeric(df.loc[next_tr, config.COL_RESP], errors="coerce") if config.COL_RESP in df.columns else pd.Series(np.nan, index=next_tr)
    y_te_raw = pd.to_numeric(df.loc[next_te, config.COL_RESP], errors="coerce") if config.COL_RESP in df.columns else pd.Series(np.nan, index=next_te)
    mtr = y_tr_raw.isin([0, 1]).values
    mte = y_te_raw.isin([0, 1]).values
    prev_tr, next_tr = prev_tr[mtr], next_tr[mtr]
    prev_te, next_te = prev_te[mte], next_te[mte]
    y_tr = y_tr_raw[mtr].astype(int).values
    y_te = y_te_raw[mte].astype(int).values

    X_tr = prepare_tabular_features(df.loc[next_tr], use_item=use_item, use_group=use_group, use_sex=use_sex, use_time=use_time)
    X_te = prepare_tabular_features(df.loc[next_te], use_item=use_item, use_group=use_group, use_sex=use_sex, use_time=use_time)
    # align columns
    all_cols = sorted(set(X_tr.columns).union(set(X_te.columns)))
    X_tr = X_tr.reindex(columns=all_cols, fill_value=0.0)
    X_te = X_te.reindex(columns=all_cols, fill_value=0.0)

    # add prev_correct
    prev_corr_tr = pd.to_numeric(df.loc[prev_tr, config.COL_RESP], errors="coerce").astype(float).fillna(0.0).values
    prev_corr_te = pd.to_numeric(df.loc[prev_te, config.COL_RESP], errors="coerce").astype(float).fillna(0.0).values
    X_tr.insert(0, "prev_correct", prev_corr_tr)
    X_te.insert(0, "prev_correct", prev_corr_te)

    return X_tr, X_te, y_tr, y_te, next_tr, next_te
# Train-aware sparse feature builder to eliminate leakage

def _ohe_fit_split(train_vals: pd.Series, test_vals: pd.Series, feature_name: str):
    # scikit-learn >= 1.2 uses 'sparse_output'
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)
    Xtr = enc.fit_transform(train_vals.astype(str).to_frame())
    Xte = enc.transform(test_vals.astype(str).to_frame())
    try:
        names = enc.get_feature_names_out([feature_name]).tolist()
    except AttributeError:
        names = enc.get_feature_names([feature_name]).tolist()
    if _SP_AVAILABLE:
        return Xtr.tocsr(), Xte.tocsr(), names
    else:
        return Xtr.toarray(), Xte.toarray(), names


def prepare_tabular_features_sparse_split(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    use_item: bool = True,
    use_group: bool = True,
    use_sex: bool = True,
    use_time: bool = True,
) -> tuple[object, object, List[str]]:
    parts_tr: List[object] = []
    parts_te: List[object] = []
    names: List[str] = []

    # Helpers to slice
    def tr(series: pd.Series) -> pd.Series:
        return series.iloc[train_idx]
    def te(series: pd.Series) -> pd.Series:
        return series.iloc[test_idx]

    # Categorical parts: fit on train, transform both
    if use_item:
        if config.COL_ITEM in df.columns:
            Xtr, Xte, nm = _ohe_fit_split(tr(df[config.COL_ITEM]), te(df[config.COL_ITEM]), config.COL_ITEM)
            parts_tr.append(Xtr); parts_te.append(Xte); names.extend(nm)
        elif config.COL_ITEM_INDEX in df.columns:
            Xtr, Xte, nm = _ohe_fit_split(tr(df[config.COL_ITEM_INDEX].astype(str)), te(df[config.COL_ITEM_INDEX].astype(str)), config.COL_ITEM_INDEX)
            parts_tr.append(Xtr); parts_te.append(Xte); names.extend(nm)

    if use_group and config.COL_GROUP in df.columns:
        Xtr, Xte, nm = _ohe_fit_split(tr(df[config.COL_GROUP]), te(df[config.COL_GROUP]), config.COL_GROUP)
        parts_tr.append(Xtr); parts_te.append(Xte); names.extend(nm)

    if use_sex and config.COL_SEX in df.columns:
        Xtr, Xte, nm = _ohe_fit_split(tr(df[config.COL_SEX]), te(df[config.COL_SEX]), config.COL_SEX)
        parts_tr.append(Xtr); parts_te.append(Xte); names.extend(nm)

    # Continuous parts: compute stats on train only
    if use_time and "time_index" in df.columns:
        ti_tr = tr(df["time_index"]).astype(float)
        mu = ti_tr.mean(); sd = ti_tr.std(ddof=1) + 1e-9
        ti_te = te(df["time_index"]).astype(float)
        vtr = ((ti_tr - mu) / sd).to_numpy(dtype=np.float32).reshape(-1, 1)
        vte = ((ti_te - mu) / sd).to_numpy(dtype=np.float32).reshape(-1, 1)
        Xtr = (_sp.csr_matrix(vtr) if _SP_AVAILABLE else vtr)
        Xte = (_sp.csr_matrix(vte) if _SP_AVAILABLE else vte)
        parts_tr.append(Xtr); parts_te.append(Xte); names.append("time_index_z")

    if config.COL_RT in df.columns:
        rtr = pd.to_numeric(tr(df[config.COL_RT]), errors="coerce").clip(lower=0)
        med = rtr[rtr >= 0].median()
        rtr = rtr.fillna(0 if np.isnan(med) else med)
        rte = pd.to_numeric(te(df[config.COL_RT]), errors="coerce").clip(lower=0)
        rte = rte.fillna(0 if np.isnan(med) else med)
        vtr = np.log1p(rtr).to_numpy(dtype=np.float32).reshape(-1, 1)
        vte = np.log1p(rte).to_numpy(dtype=np.float32).reshape(-1, 1)
        Xtr = (_sp.csr_matrix(vtr) if _SP_AVAILABLE else vtr)
        Xte = (_sp.csr_matrix(vte) if _SP_AVAILABLE else vte)
        parts_tr.append(Xtr); parts_te.append(Xte); names.append("log_rt")

    if not parts_tr:
        if _SP_AVAILABLE:
            Ztr = _sp.csr_matrix((len(train_idx), 0), dtype=np.float32)
            Zte = _sp.csr_matrix((len(test_idx), 0), dtype=np.float32)
        else:
            Ztr = np.zeros((len(train_idx), 0), dtype=np.float32)
            Zte = np.zeros((len(test_idx), 0), dtype=np.float32)
        return Ztr, Zte, []

    if _SP_AVAILABLE:
        X_tr = _sp.hstack(parts_tr, format="csr")
        X_te = _sp.hstack(parts_te, format="csr")
    else:
        X_tr = np.concatenate([p if isinstance(p, np.ndarray) else p.toarray() for p in parts_tr], axis=1).astype(np.float32, copy=False)
        X_te = np.concatenate([p if isinstance(p, np.ndarray) else p.toarray() for p in parts_te], axis=1).astype(np.float32, copy=False)

    return X_tr, X_te, names
