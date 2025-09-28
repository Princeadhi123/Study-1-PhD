from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from . import config
from .utils import build_dataset
from .metrics import safe_metrics

# Import model runners
from .models import rasch, bkt, logreg, dkt, gkt, tirt, mtl, contrastive, adapt

import re


def _sanitize_model_name(name: str) -> str:
    s = name
    # Remove specific parenthetical qualifiers e.g., (minimal), (CORAL)
    s = re.sub(r"\s*\((?:minimal|coral)\)", "", s, flags=re.IGNORECASE)
    # Remove standalone word 'lite'
    s = re.sub(r"\blite\b", "", s, flags=re.IGNORECASE)
    # Clean up repeated spaces and dangling hyphens/underscores
    s = re.sub(r"\s+", " ", s)
    s = s.strip().strip("-_ ")
    return s


def ensure_outdir() -> Path:
    out = config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


essential_cols = [
    config.COL_ID,
    config.COL_ITEM,
    config.COL_ITEM_INDEX,
    config.COL_GROUP,
    config.COL_RESP,
    config.COL_RT,
    config.COL_ORDER,
    config.COL_SEX,
]


def summarize_result(name: str, category: str, why: str, result: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "category": category,
        "model": name,
        "why": why,
    }
    if "error" in result and result["error"]:
        row.update({
            "n": 0,
            "accuracy": np.nan,
            "roc_auc": np.nan,
            "avg_precision": np.nan,
            "log_loss": np.nan,
            "f1": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "brier": np.nan,
            "error": result["error"],
        })
        return row
    y_true = result.get("y_true", None)
    y_prob = result.get("y_prob", None)
    if y_true is None or y_prob is None:
        row.update({"n": 0, "error": "Missing predictions"})
        return row
    m = safe_metrics(np.array(y_true), np.array(y_prob))
    row.update(m)
    # Extra known fields
    if "aux_mae_rt" in result:
        row["aux_mae_rt"] = float(result["aux_mae_rt"]) if result["aux_mae_rt"] is not None else np.nan
    if "aux_rmse_rt" in result:
        row["aux_rmse_rt"] = float(result["aux_rmse_rt"]) if result["aux_rmse_rt"] is not None else np.nan
    return row


def save_predictions(outdir: Path, tag: str, result: Dict[str, Any]):
    if "y_true" not in result or "y_prob" not in result:
        return
    # Ensure directory exists (belt-and-suspenders)
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    y_true = np.asarray(result["y_true"])
    y_prob = np.asarray(result["y_prob"])
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    # Attach row indices if provided
    if "test_rows" in result and isinstance(result["test_rows"], (list, np.ndarray, pd.Series)):
        idx = np.array(result["test_rows"])  # could be a subset
        # best effort: keep index alongside, aligned to available predictions
        if idx.ndim > 1:
            idx = idx.reshape(-1)
        L = min(len(idx), len(df))
        if L > 0:
            df.insert(0, "df_row_index", np.asarray(idx[:L], dtype=int))
    df.to_csv(outdir / f"preds_{tag}.csv", index=False)


def main():
    outdir = ensure_outdir()

    # Load data and split
    ds = build_dataset()
    df = ds.df
    train_idx, test_idx = ds.train_idx, ds.test_idx

    # Persist a skim of input columns for reference
    cols_present = [c for c in essential_cols if c in df.columns]
    (outdir / "_columns_present.json").write_text(json.dumps(cols_present, indent=2), encoding="utf-8")

    # Register models
    runners = [
        (rasch.run, _sanitize_model_name("Rasch1PL"), "Psychometric (IRT)"),
        (bkt.run, _sanitize_model_name("BKT"), "Bayesian"),
        (logreg.run, _sanitize_model_name("LogisticRegression"), "Machine Learning"),
        (dkt.run, _sanitize_model_name("DKT (minimal)"), "Deep Learning"),
        (gkt.run, _sanitize_model_name("GKT-lite"), "Graph"),
        (tirt.run, _sanitize_model_name("TIRT-lite"), "Temporal/Sequential"),
        (mtl.run, _sanitize_model_name("FKT-lite"), "Multi-task"),
        (contrastive.run, _sanitize_model_name("CLKT-lite"), "Contrastive/Self-supervised"),
        (adapt.run, _sanitize_model_name("AdaptKT-lite (CORAL)"), "Domain Adaptive"),
    ]

    # Execute
    rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}

    for run_fn, name, category in runners:
        print(f"Running {category} :: {name}")
        t0 = time.perf_counter()
        res = run_fn(df, train_idx, test_idx)
        runtime_sec = float(time.perf_counter() - t0)
        # Keep a copy for details
        details[name] = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in res.items() if k not in ("y_true", "y_prob")}
        details[name]["runtime_sec"] = runtime_sec

        # Save predictions when available
        tag = name.lower().replace(" ", "_").replace("/", "-")
        save_predictions(outdir, tag, res)

        # Summarize metrics
        row = summarize_result(name=name, category=category, why=res.get("why", ""), result=res)
        row["runtime_sec"] = runtime_sec
        rows.append(row)

    # Summary table
    summ = pd.DataFrame(rows)
    summ.to_csv(outdir / "metrics_summary.csv", index=False)

    # Markdown report
    lines: List[str] = []
    lines.append(f"# KT Benchmark Summary\n\n")
    lines.append(f"Run time: {datetime.now().isoformat()}\n\n")
    lines.append("## Models\n\n")
    for r in rows:
        lines.append(f"- **{r['category']} — {r['model']}**: {r.get('why','').strip()}\n")
    lines.append("\n## Metrics (higher is better unless noted)\n\n")
    show_cols = [
        "category", "model", "n", "roc_auc", "accuracy", "avg_precision", "f1", "precision", "recall", "brier", "log_loss", "runtime_sec",
    ]
    present = [c for c in show_cols if c in summ.columns]
    if present:
        # Format a compact table with safe fallback if tabulate is missing
        tbl = summ[present].copy()
        try:
            md_tbl = tbl.to_markdown(index=False)
        except Exception:
            # Fallback to CSV-style in a fenced code block
            header = ", ".join(map(str, tbl.columns))
            body = [", ".join(map(lambda x: f"{x}", row)) for row in tbl.values]
            csv_text = "\n".join([header] + body)
            md_tbl = "```text\n" + csv_text + "\n```"
        lines.append(md_tbl)
        lines.append("\n")

    # Best per metric
    lines.append("## Best by metric\n\n")
    for metric in ["roc_auc", "accuracy", "avg_precision", "f1"]:
        if metric in summ.columns and summ[metric].notna().any():
            idx = int(summ[metric].astype(float).idxmax())
            r = summ.loc[idx]
            lines.append(f"- **{metric}**: {r['model']} ({r['category']}) = {r[metric]:.4f} (n={int(r['n'])})\n")
    lines.append("\nNotes: Brier and log_loss are lower-is-better. Metrics skip models that failed.\n")

    (outdir / "metrics_summary.md").write_text("".join(lines), encoding="utf-8")

    # Save details JSON for reproducibility
    (outdir / "details.json").write_text(json.dumps(details, indent=2, default=str), encoding="utf-8")

    print("Saved:", outdir / "metrics_summary.csv")
    print("Saved:", outdir / "metrics_summary.md")


if __name__ == "__main__":
    main()
