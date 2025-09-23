from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


def name_to_tag(name: str) -> str:
    # mimic run_benchmark.py tagging
    return name.lower().replace(" ", "_").replace("/", "-")


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
    sns.set(style="white", context="paper")
    m = len(metrics)
    ncols = 3
    nrows = int(np.ceil(m / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.5, 2.8 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)
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
            disp = ConfusionMatrixDisplay(cm)
            disp.plot(ax=ax, colorbar=False)
            ax.set_title(f"{name}\nT={t:.2f} | F1={stats['f1']:.3f} Acc={stats['accuracy']:.3f}", fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, f"No data\n{name}", ha="center", va="center")
    # Remove empty axes
    for j in range(i + 1, nrows * ncols):
        r = j // ncols
        c = j % ncols
        fig.delaxes(axes[r, c])
    fig.suptitle("Confusion Matrices (threshold chosen by max F1)", y=0.995)
    fig.tight_layout()
    fig.savefig(outdir / "confusion_matrices.png", dpi=300)
    fig.savefig(outdir / "confusion_matrices.pdf")
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

    print("Saved plots to:", plot_dirs["base"].resolve())


if __name__ == "__main__":
    main()
