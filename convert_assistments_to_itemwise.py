from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import shutil
from typing import Optional

import pandas as pd
import numpy as np


def pick_first_present(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_sex(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(['U'] * len(series.index) if hasattr(series, 'index') else ['U'], index=series.index if hasattr(series, 'index') else None)
    s = series.astype(str).str.strip().str.lower().str.replace(r"[^a-z]", "", regex=True)
    mapping = {
        "boy": "M", "b": "M", "male": "M", "m": "M", "man": "M",
        "girl": "F", "gir": "F", "g": "F", "female": "F", "f": "F", "woman": "F",
    }
    out = s.map(mapping).fillna('U')
    return out


def main():
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Convert ASSISTments 2009-2010 CSV to minimal itemwise format")
    parser.add_argument("--input", type=str, default=str((base_dir / "assistments_2009_2010.csv").resolve()), help="Path to ASSISTments CSV")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: *_itemwise.csv next to input)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite input CSV in place (backs up original)")
    args = parser.parse_args()

    input_csv = Path(args.input)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input not found: {input_csv}")

    # Decide output
    if args.overwrite:
        output_csv = input_csv
    else:
        output_csv = Path(args.output) if args.output else input_csv.with_name(input_csv.stem + "_itemwise" + input_csv.suffix)

    print(f"Reading: {input_csv}")
    df = pd.read_csv(input_csv)

    # Infer columns
    col_id = pick_first_present(df, [
        'studentid', 'user_id', 'userId', 'Anon Student Id', 'anonymized_student_id', 'student_id', 'student'
    ])
    col_item = pick_first_present(df, [
        'problem_id', 'item_id', 'question_id', 'problem', 'item'
    ])
    col_group = pick_first_present(df, [
        'KC(Default)', 'skill_name', 'skill', 'skill_id', 'kc', 'kc_default'
    ])
    # Prefer explicit answer type for group when available
    col_ans_type = pick_first_present(df, [
        'answer_type', 'Answer Type', 'ans_type'
    ])
    col_resp = pick_first_present(df, [
        'correct', 'is_correct', 'first_attempt', 'Outcome'
    ])
    col_rt_ms = pick_first_present(df, [
        'ms_first_response_time', 'ms_first_response', 'time_ms_first_response', 'ms_response', 'First Transaction Time (ms)'
    ])
    col_start = pick_first_present(df, [
        'start_time', 'StartTime', 'Start Time', 'startTime', 'start', 'timestamp'
    ])
    col_order = pick_first_present(df, [
        'order_id', 'sequence_id', 'Order Id', 'seq', 'row'
    ])
    col_sex = pick_first_present(df, [
        'sex', 'gender'
    ])

    if col_id is None or col_item is None or col_resp is None:
        raise ValueError("Required columns not found: need at least student id, item id, and correctness.")

    # Build response
    y = df[col_resp]
    # Normalize correctness to 0/1
    if y.dtype.kind in {'i', 'u', 'f'}:
        y_bin = pd.to_numeric(y, errors='coerce').round().clip(0, 1).astype('Int64')
    else:
        y_str = y.astype(str).str.strip().str.lower()
        y_bin = y_str.map({
            '1': 1, '0': 0,
            'correct': 1, 'incorrect': 0,
            'true': 1, 'false': 0,
            'right': 1, 'wrong': 0,
            'yes': 1, 'no': 0,
        }).astype('Int64')
    # Item
    item = df[col_item].astype(str)

    # Group: prefer 'answer_type' if present; otherwise KC/skill (if multiple KCs delimited by '~~', pick first)
    if col_ans_type is not None and col_ans_type in df.columns:
        group = df[col_ans_type].astype(str)
    elif col_group is not None and col_group in df.columns:
        grp_raw = df[col_group].astype(str)
        group = grp_raw.str.split('~~').str[0]
    else:
        group = pd.Series(np.nan, index=df.index)

    # Response time seconds
    if col_rt_ms is not None and col_rt_ms in df.columns:
        rt_sec = pd.to_numeric(df[col_rt_ms], errors='coerce') / 1000.0
    else:
        # Try to compute from start/end time if available
        col_end = pick_first_present(df, ['end_time', 'EndTime', 'End Time', 'end', 'endTime'])
        if col_start is not None and col_end is not None:
            try:
                t0 = pd.to_datetime(df[col_start], errors='coerce')
                t1 = pd.to_datetime(df[col_end], errors='coerce')
                dt = (t1 - t0).dt.total_seconds()
                rt_sec = dt
            except Exception:
                rt_sec = pd.Series(np.nan, index=df.index)
        else:
            rt_sec = pd.Series(np.nan, index=df.index)

    # Sex
    if col_sex is not None and col_sex in df.columns:
        sex = normalize_sex(df[col_sex])
    else:
        sex = pd.Series(['U'] * len(df), index=df.index)

    out = pd.DataFrame({
        'IDCode': df[col_id].astype(str),
        'item': item,
        'group': group,
        'response': y_bin,
        'response_time_sec': pd.to_numeric(rt_sec, errors='coerce'),
        'sex': sex,
    })

    # Filter to valid binary response rows
    mask = out['response'].isin([0, 1])
    out = out[mask].copy()

    # Recompute orig_order per IDCode after filtering so numbering is 1..n with no gaps
    out['orig_order'] = out.groupby('IDCode').cumcount() + 1
    out['orig_order'] = out['orig_order'].astype('Int64')
    # Reorder columns to keep expected layout
    out = out[['IDCode', 'orig_order', 'item', 'group', 'response', 'response_time_sec', 'sex']]

    # If overwriting, back up original
    if args.overwrite:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = input_csv.with_name(f"{input_csv.stem}_backup_{ts}{input_csv.suffix}")
        try:
            shutil.copy2(input_csv, backup)
            print("Backed up original to:", backup)
        except Exception as e:
            print("Warning: failed to back up original:", e)

    out.to_csv(output_csv, index=False)
    print("Saved:", output_csv)
    print(f"Rows: {len(out):,} | Columns: {len(out.columns)} -> {list(out.columns)}")


if __name__ == '__main__':
    main()
