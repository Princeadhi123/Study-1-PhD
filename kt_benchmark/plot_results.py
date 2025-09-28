from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.calibration import calibration_curve

from . import config
from .utils import load_itemwise_df, add_time_index


def name_to_tag(name: str) -> str:
    # mimic run_benchmark.py tagging
    return name.lower().replace(" ", "_").replace("/", "-")


def _sanitize_model_name(name: str) -> str:
    """Mirror the name normalization in run_benchmark._sanitize_model_name so
    plots can find the correct preds files regardless of '(minimal)', '(CORAL)', or 'lite' suffixes.
    """
    import re
    s = name
    s = re.sub(r"\s*\((?:minimal|coral)\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\blite\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    s = s.strip().strip("-_ ")
    return s


def _ensure_plot_dirs(base: Path) -> Dict[str, Path]:
    plots_dir = base / "plots"
    per_model_dir = plots_dir / "per_model"
    summary_dir = plots_dir / "summary"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    return {"base": plots_dir, "per_model": per_model_dir, "summary": summary_dir}


def _load_summary(outdir: Path) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    metrics_csv = outdir / "metrics_summary.csv"
    details_json = outdir / "details.json"
    metrics = pd.read_csv(metrics_csv)
    if details_json.exists():
        details = json.loads(details_json.read_text(encoding="utf-8"))
    else:
        details = {}
    return metrics, details


def _load_predictions(outdir: Path, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    tag = name_to_tag(model_name)
    p = outdir / f"preds_{tag}.csv"
    if not p.exists():
        # Some names include parentheses in tag; try a looser search
        candidates = list(outdir.glob(f"preds_{tag}*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No prediction file for model '{model_name}' (expected {p})")
        p = candidates[0]
    df = pd.read_csv(p)
    y_true = pd.to_numeric(df["y_true"], errors="coerce").to_numpy()
    y_prob = pd.to_numeric(df["y_prob"], errors="coerce").to_numpy()
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    return y_true[mask].astype(int), y_prob[mask].astype(float)


def _load_predictions_with_index(outdir: Path, model_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load predictions and optional original row indices if present."""
    tag = name_to_tag(model_name)
    p = outdir / f"preds_{tag}.csv"
    if not p.exists():
        candidates = list(outdir.glob(f"preds_{tag}*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No prediction file for model '{model_name}' (expected {p})")
        p = candidates[0]
    df = pd.read_csv(p)
    y_true = pd.to_numeric(df.get("y_true"), errors="coerce").to_numpy()
    y_prob = pd.to_numeric(df.get("y_prob"), errors="coerce").to_numpy()
    idx = pd.to_numeric(df.get("df_row_index"), errors="coerce").to_numpy() if "df_row_index" in df.columns else None
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    if idx is not None:
        mask = mask & np.isfinite(idx)
        return y_true[mask].astype(int), y_prob[mask].astype(float), idx[mask].astype(int)
    else:
        return y_true[mask].astype(int), y_prob[mask].astype(float), np.array([], dtype=int)


def _collect_model_predictions_with_dfrows(metrics: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    """Return {model: {y_true, y_prob, rows}} only for models that have row indices stored."""
    out = {}
    for _, row in metrics.iterrows():
        name = row["model"]
        try:
            yt, yp, idx = _load_predictions_with_index(config.OUTPUT_DIR, name)
            if idx.size > 0:
                out[name] = {"y_true": yt, "y_prob": yp, "rows": idx}
        except Exception:
            continue
    return out


def plot_student_trajectories_all_models(
    metrics: pd.DataFrame,
    outdir: Path,
    max_students: int = 2,
    min_len: int = 30,
    max_len: int = 60,
    max_models: int = 2,
    models: Optional[List[str]] = None,
):
    """
    For the current dataset, pick up to max_students students and plot trajectories of
    predicted probabilities over time for ALL models in one figure (dataset-wise).
    Correct vs. incorrect points highlighted by marker face color (green=correct, red=wrong) per model.
    """
    df = add_time_index(load_itemwise_df())
    if config.COL_ID not in df.columns or config.COL_ORDER not in df.columns:
        return
    model_preds = _collect_model_predictions_with_dfrows(metrics)
    if not model_preds:
        return
    # Map df row index -> (student, order)
    id_map = df[config.COL_ID].astype(str)
    order_map = df[config.COL_ORDER]

    # Candidate students: limit by interaction counts in dataset
    sizes = df.groupby(df[config.COL_ID].astype(str))[config.COL_ID].size()
    allowed = set(sizes[(sizes >= min_len) & (sizes <= max_len)].index.astype(str).tolist())

    # If user provided explicit models, prefer students present in ALL of them
    intersection_students: Optional[set] = None
    if models:
        models_norm = [_sanitize_model_name(m) for m in models]
        present_sets = []
        for mdl in models_norm:
            if mdl in model_preds:
                sids = set(id_map.loc[model_preds[mdl]["rows"]].astype(str))
                present_sets.append(sids)
        if present_sets:
            inter_full = set.intersection(*present_sets)
            inter_allowed = inter_full.intersection(allowed)
            intersection_students = inter_allowed if inter_allowed else inter_full

    # Score by overlap coverage across models
    sid_counts: Dict[str, int] = {}
    for m in model_preds.values():
        sids = id_map.loc[m["rows"]].astype(str)
        for s in sids:
            if s in allowed:
                sid_counts[s] = sid_counts.get(s, 0) + 1

    if intersection_students:
        # Choose from intersection first, sorted by coverage (desc)
        top_students = [s for s, _ in sorted(((s, sid_counts.get(s, 0)) for s in intersection_students), key=lambda x: -x[1])][:max_students]
        # If intersection too small, pad with top by coverage
        if len(top_students) < max_students:
            remainder = [s for s, _ in sorted(sid_counts.items(), key=lambda x: -x[1]) if s not in top_students]
            top_students += remainder[: (max_students - len(top_students))]
    else:
        top_students = [s for s, _ in sorted(sid_counts.items(), key=lambda x: -x[1])][:max_students]
    if not top_students:
        return

    # Choose models: explicit list from caller/config if provided, else top by metric order
    if models:
        # Normalize provided names to match saved/sanitized keys and include ALL provided
        models_norm = [_sanitize_model_name(m) for m in models]
        model_list = [m for m in models_norm if m in model_preds]
        if not model_list:
            ordered_models = [row["model"] for _, row in metrics.iterrows() if row["model"] in model_preds]
            model_list = ordered_models if ordered_models else list(model_preds.keys())
    else:
        ordered_models = [row["model"] for _, row in metrics.iterrows() if row["model"] in model_preds]
        model_list = ordered_models[:max_models] if ordered_models else list(model_preds.keys())[:max_models]
    # Highly distinct colors across hues (one unique hue per model)
    colors = sns.color_palette("husl", n_colors=len(model_list) if len(model_list) > 0 else 8)
    linestyles = ["solid"]  # keep all lines solid for easier tracing
    color_by_model = {mdl: colors[i % len(colors)] for i, mdl in enumerate(model_list)}
    style_by_model = {mdl: linestyles[i % len(linestyles)] for i, mdl in enumerate(model_list)}
    # Per-line markers for additional distinguishability
    markers = ["D", "o", "s", "^", "v", "P", "X", "*", "<", ">"]
    marker_by_model = {mdl: markers[i % len(markers)] for i, mdl in enumerate(model_list)}

    n = len(top_students)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, 2.7 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, sid in zip(axes, top_students):
        # Ground truth points from dataset for this student (bold black dots at y=0/1)
        sub = df.loc[id_map == sid]
        y_true_all = pd.to_numeric(sub[config.COL_RESP], errors="coerce")
        m = y_true_all.isin([0, 1])
        ord_true = sub.loc[m, config.COL_ORDER].to_numpy()
        gt = y_true_all[m].astype(int).to_numpy()
        ax.scatter(ord_true, gt, s=35, c="black", edgecolors="white", linewidths=0.6, zorder=4, label="Ground truth")

        # Model prediction lines (thicker, higher-contrast)
        for mdl in model_list:
            data = model_preds[mdl]
            mask_sid = id_map.loc[data["rows"]].astype(str).values == sid
            if not mask_sid.any():
                continue
            rows_sid = data["rows"][mask_sid]
            ord_sid = order_map.loc[rows_sid].to_numpy()
            order_idx = np.argsort(ord_sid)
            ord_sid = ord_sid[order_idx]
            yp = data["y_prob"][mask_sid][order_idx]
            ax.plot(
                ord_sid,
                yp,
                color=color_by_model[mdl],
                lw=2.2,
                alpha=0.9,
                linestyle=style_by_model[mdl],
                marker=marker_by_model[mdl],
                markevery=5,
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor=color_by_model[mdl],
                markeredgewidth=0.9,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
                label=mdl,
            )
        ax.set_ylim(0, 1)
        ax.set_ylabel("p(correct)")
        ax.set_title(f"Student {sid}")
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.02)
    axes[-1].set_xlabel("Interaction order")
    # Shared legend below the plots (outside) with many columns and strong color separation
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=min(8, len(labels)),
            fontsize=9,
            frameon=True,
        )
        fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    else:
        fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    fig.suptitle("Example Student Trajectories", y=0.985)
    fig.savefig(outdir / "trajectories_all_models.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "trajectories_all_models.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_prediction_heatmaps(
    metrics: pd.DataFrame,
    outdir: Path,
    max_students: int = 5,
    min_len: int = 20,
    max_len: int = 60,
    max_steps_cap: int = 60,
    annot_items: bool = False,
):
    """
    For the current dataset, pick 4-5 students, and for ALL models produce a combined figure
    with one heatmap per model (dataset-wise). Each heatmap shows rows=students, columns=interaction order,
    and color = predicted probability. Item IDs are overlaid as small text per cell when available.
    """
    df = add_time_index(load_itemwise_df())
    if config.COL_ID not in df.columns or config.COL_ORDER not in df.columns:
        return
    model_preds = _collect_model_predictions_with_dfrows(metrics)
    if not model_preds:
        return
    id_map = df[config.COL_ID].astype(str)
    order_map = df[config.COL_ORDER]
    item_col = config.COL_ITEM if config.COL_ITEM in df.columns else (config.COL_ITEM_INDEX if config.COL_ITEM_INDEX in df.columns else None)
    item_map = df[item_col].astype(str) if item_col else pd.Series(["" for _ in range(len(df))], index=df.index)

    # Choose students by most coverage across models, restricted to length range
    sizes = df.groupby(df[config.COL_ID].astype(str))[config.COL_ID].size()
    allowed = set(sizes[(sizes >= min_len) & (sizes <= max_len)].index.astype(str).tolist())
    sid_counts: Dict[str, int] = {}
    for m in model_preds.values():
        sids = id_map.loc[m["rows"]].astype(str)
        for s in sids:
            if s in allowed:
                sid_counts[s] = sid_counts.get(s, 0) + 1
    students = [s for s, _ in sorted(sid_counts.items(), key=lambda x: -x[1])][:max_students]
    if not students:
        return

    M = len(model_preds)
    ncols = 3
    nrows = int(np.ceil(M / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 2.6 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    cmap = sns.color_palette("viridis", as_cmap=True)

    mdl_list = list(model_preds.keys())
    for i, mdl in enumerate(mdl_list):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]
        data = model_preds[mdl]
        # For each student, align by order, build a dense vector with NaNs for missing
        rows_mat = []
        annotations: List[List[str]] = []
        max_steps = 0
        for sid in students:
            mask_sid = id_map.loc[data["rows"]].astype(str).values == sid
            if not mask_sid.any():
                rows_mat.append(np.array([]))
                annotations.append([])
                continue
            rows = data["rows"][mask_sid]
            ords = order_map.loc[rows].to_numpy()
            args = np.argsort(ords)
            ords = ords[args]
            probs = data["y_prob"][mask_sid][args]
            items = item_map.loc[rows].astype(str).values[args]
            # Cap sequence length for readability
            if len(ords) > max_steps_cap:
                ords = ords[:max_steps_cap]
                probs = probs[:max_steps_cap]
                items = items[:max_steps_cap]
            max_steps = max(max_steps, len(ords))
            rows_mat.append(probs)
            annotations.append(items.tolist())
        # Build a 2D array with NaNs padded to max_steps
        mat = np.full((len(students), max_steps), np.nan, dtype=float)
        ann = [["" for _ in range(max_steps)] for _ in range(len(students))]
        for s_idx, rowv in enumerate(rows_mat):
            if rowv.size:
                L = min(max_steps, len(rowv))
                mat[s_idx, :L] = rowv[:L]
                for j in range(L):
                    ann[s_idx][j] = annotations[s_idx][j]
        # Render with mask so NaNs show as background
        mask = ~np.isfinite(mat)
        sns.heatmap(mat, ax=ax, cmap=cmap, vmin=0, vmax=1, mask=mask, cbar=True, cbar_kws={"shrink": 0.6})
        ax.set_title(mdl)
        ax.set_yticks(range(len(students)))
        ax.set_yticklabels([str(s) for s in students])
        ax.set_xticks(range(max_steps))
        ax.set_xticklabels([str(k + 1) for k in range(max_steps)], fontsize=8)
        ax.set_xlabel("Order")
        if c == 0:
            ax.set_ylabel("Student")
        # Optional overlay of item ids
        if annot_items:
            for r_i in range(len(students)):
                for c_j in range(max_steps):
                    itxt = ann[r_i][c_j]
                    if itxt:
                        ax.text(c_j + 0.5, r_i + 0.5, str(itxt), ha="center", va="center", fontsize=6, color="white")

    # Remove any unused axes
    for j in range(i + 1, nrows * ncols):
        r = j // ncols
        c = j % ncols
        fig.delaxes(axes[r, c])

    fig.suptitle("Prediction Probability Heatmaps (rows=students, cols=order)", y=0.995)
    fig.tight_layout()
    fig.savefig(outdir / "heatmaps_all_models.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "heatmaps_all_models.pdf", bbox_inches="tight")
    plt.close(fig)


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> Tuple[float, Dict[str, float]]:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_val = -np.inf
    best_stats = {}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            val = f1_score(y_true, y_pred, zero_division=0)
        else:
            # default to F1
            val = f1_score(y_true, y_pred, zero_division=0)
        if val > best_val:
            best_val = val
            best_t = float(t)
            best_stats = {
                "f1": float(val),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            }
    return best_t, best_stats


def plot_roc_all(metrics: pd.DataFrame, outdir: Path):
    sns.set(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = sns.color_palette("tab10", n_colors=len(metrics))
    for i, row in metrics.iterrows():
        name = row["model"]
        try:
            y_true, y_prob = _load_predictions(config.OUTPUT_DIR, name)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", color=colors[i % len(colors)], lw=2)
        except Exception as e:
            print(f"[ROC] Skipping {name}: {e}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Test)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "roc_all.png", dpi=300)
    fig.savefig(outdir / "roc_all.pdf")
    plt.close(fig)


def plot_pr_all(metrics: pd.DataFrame, outdir: Path):
    sns.set(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = sns.color_palette("tab10", n_colors=len(metrics))
    for i, row in metrics.iterrows():
        name = row["model"]
        try:
            y_true, y_prob = _load_predictions(config.OUTPUT_DIR, name)
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            base = y_true.mean() if len(y_true) else 0.0
            ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})", color=colors[i % len(colors)], lw=2)
        except Exception as e:
            print(f"[PR] Skipping {name}: {e}")
    # baseline
    ax.hlines(base, 0, 1, colors="gray", linestyles="--", label=f"Baseline={base:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (Test)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "pr_all.png", dpi=300)
    fig.savefig(outdir / "pr_all.pdf")
    plt.close(fig)


def plot_calibration_grid(metrics: pd.DataFrame, outdir: Path):
    sns.set(style="whitegrid", context="paper")
    m = len(metrics)
    ncols = 3
    nrows = int(np.ceil(m / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.5, 2.8 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)
    for i, row in metrics.reset_index(drop=True).iterrows():
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]
        name = row["model"]
        try:
            y_true, y_prob = _load_predictions(config.OUTPUT_DIR, name)
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
            ax.plot(prob_pred, prob_true, marker="o", lw=1.5, label=name)
            ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
            ax.set_title(name, fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, f"No data\n{name}", ha="center", va="center")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if r == nrows - 1:
            ax.set_xlabel("Predicted probability")
        if c == 0:
            ax.set_ylabel("Observed frequency")
    # Remove empty axes
    for j in range(i + 1, nrows * ncols):
        r = j // ncols
        c = j % ncols
        fig.delaxes(axes[r, c])
    fig.suptitle("Calibration (Reliability) Curves", y=0.995)
    fig.tight_layout()
    fig.savefig(outdir / "calibration_grid.png", dpi=300)
    fig.savefig(outdir / "calibration_grid.pdf")
    plt.close(fig)


def plot_confusion_matrices(metrics: pd.DataFrame, outdir: Path):
    """
    Plot a grid of confusion matrices with CONSISTENT colors for TN/FP/FN/TP across all models.
    Color mapping (fixed):
      TN -> blue, FP -> orange, FN -> red, TP -> greenish/teal.
    """
    sns.set(style="white", context="paper")
    m = len(metrics)
    ncols = 3
    nrows = int(np.ceil(m / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.5, 2.9 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    # Fixed color mapping per cell index (0..3) across all matrices
    # cm layout: [[TN, FP], [FN, TP]]
    color_list = ["#4C78A8", "#F58518", "#E45756", "#72B7B2"]  # TN, FP, FN, TP
    cmap = ListedColormap(color_list)
    color_index = np.array([[0, 1], [2, 3]])

    # For a shared legend
    patches = [
        mpatches.Patch(color=color_list[0], label="TN"),
        mpatches.Patch(color=color_list[1], label="FP"),
        mpatches.Patch(color=color_list[2], label="FN"),
        mpatches.Patch(color=color_list[3], label="TP"),
    ]

    for i, row in metrics.reset_index(drop=True).iterrows():
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]
        name = row["model"]
        try:
            y_true, y_prob = _load_predictions(config.OUTPUT_DIR, name)
            t, stats = best_threshold(y_true, y_prob, metric="f1")
            y_pred = (y_prob >= t).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            # Draw colored cells with fixed mapping, independent of count magnitudes
            ax.imshow(color_index, cmap=cmap, vmin=0, vmax=3)
            # Annotate counts
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            counts = np.array([[tn, fp], [fn, tp]])
            for (rr, cc), val in np.ndenumerate(counts):
                ax.text(cc, rr, f"{val}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            # Axes cosmetics
            ax.set_xticks([0, 1], labels=["0", "1"])  # predicted
            ax.set_yticks([0, 1], labels=["0", "1"])  # true
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            # Title: model name only (remove threshold/metrics)
            ax.set_title(f"{name}", fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, f"No data\n{name}", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])

    # Remove empty axes
    for j in range(i + 1, nrows * ncols):
        r = j // ncols
        c = j % ncols
        fig.delaxes(axes[r, c])

    # Shared legend and title
    fig.legend(handles=patches, loc="lower center", ncol=4, title="Cells", title_fontsize=9, fontsize=8)
    fig.suptitle("Confusion Matrices (threshold chosen by max F1)", y=0.995)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(outdir / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "confusion_matrices.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(metrics: pd.DataFrame, outdir: Path):
    sns.set(style="whitegrid", context="paper")
    metric_cols = [
        ("roc_auc", "ROC AUC (↑)"),
        ("accuracy", "Accuracy (↑)"),
        ("avg_precision", "Average Precision (↑)"),
        ("f1", "F1 (↑)"),
        ("brier", "Brier (↓)"),
        ("log_loss", "Log Loss (↓)"),
    ]
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
    axes = np.array(axes).reshape(2, 3)
    for (col, title), ax in zip(metric_cols, axes.ravel()):
        if col not in metrics.columns:
            ax.axis("off")
            continue
        df = metrics[["model", col]].copy()
        df = df.sort_values(col, ascending=(col in {"brier", "log_loss"}))
        sns.barplot(data=df, x=col, y="model", ax=ax, palette="viridis")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        # annotate values
        for i, v in enumerate(df[col].values):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    fig.suptitle("Model Comparison on Test Metrics", y=0.995)
    fig.tight_layout()
    fig.savefig(outdir / "metric_bars.png", dpi=300)
    fig.savefig(outdir / "metric_bars.pdf")
    plt.close(fig)


def plot_per_model(metrics: pd.DataFrame, outdir: Path):
    sns.set(style="whitegrid", context="paper")
    for _, row in metrics.iterrows():
        name = row["model"]
        try:
            y_true, y_prob = _load_predictions(config.OUTPUT_DIR, name)
        except Exception as e:
            print(f"[Per-model] Skipping {name}: {e}")
            continue
        # ROC + PR + hist in one figure
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.2))
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
        axes[0].plot(fpr, tpr, lw=2)
        axes[0].set_title(f"ROC (AUC={auc_val:.3f})")
        axes[0].set_xlabel("FPR")
        axes[0].set_ylabel("TPR")
        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        base = y_true.mean() if len(y_true) else 0.0
        axes[1].plot(recall, precision, lw=2)
        axes[1].hlines(base, 0, 1, colors="gray", linestyles="--")
        axes[1].set_title(f"PR (AP={ap:.3f})")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        # Histogram
        axes[2].hist(y_prob, bins=25, color="#4c72b0")
        axes[2].set_title("Predicted Probabilities")
        axes[2].set_xlabel("p(y=1)")
        axes[2].set_ylabel("Count")
        fig.suptitle(name)
        fig.tight_layout()
        tag = name_to_tag(name)
        fig.savefig(outdir / f"{tag}__overview.png", dpi=300)
        fig.savefig(outdir / f"{tag}__overview.pdf")
        plt.close(fig)


def main():
    outdir = config.OUTPUT_DIR
    plot_dirs = _ensure_plot_dirs(outdir)
    metrics, details = _load_summary(outdir)

    # Order models by ROC AUC descending where available
    if "roc_auc" in metrics.columns:
        metrics = metrics.sort_values("roc_auc", ascending=False, na_position="last").reset_index(drop=True)

    # Summary plots
    plot_roc_all(metrics, plot_dirs["summary"])  # ROC overlays
    plot_pr_all(metrics, plot_dirs["summary"])   # PR overlays
    plot_metric_bars(metrics, plot_dirs["summary"])  # bar charts for metrics
    plot_calibration_grid(metrics, plot_dirs["summary"])  # reliability curves
    plot_confusion_matrices(metrics, plot_dirs["summary"])  # confusion matrices grid

    # Per-model overviews
    plot_per_model(metrics, plot_dirs["per_model"])

    # Qualitative plots using aligned row indices (if available)
    try:
        plot_student_trajectories_all_models(
            metrics,
            plot_dirs["summary"],
            max_students=2,
            models=getattr(config, "TRAJECTORY_MODELS", None),
        )
    except Exception as e:
        print(f"[Trajectories] Skipped: {e}")

    print("Saved plots to:", plot_dirs["base"].resolve())


if __name__ == "__main__":
    main()
