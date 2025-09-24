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

def one_hot(series: pd.Series) -> pd.DataFrame:
    # Use smaller dtype to reduce memory when using dense dummies
    return pd.get_dummies(series.astype("category"), dummy_na=True, dtype=np.uint8)


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
    return sequences_by(df, [config.COL_ID])


def student_group_sequences(df: pd.DataFrame) -> Dict[Tuple[str, str], pd.DataFrame]:
    if config.COL_ID not in df.columns or config.COL_GROUP not in df.columns:
        return {}
    return sequences_by(df, [config.COL_ID, config.COL_GROUP])
