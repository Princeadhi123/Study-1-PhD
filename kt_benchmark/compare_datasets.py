from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

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
)

from . import config
from . import plot_results as pr
from .run_benchmark import main as run_once


def tagify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "-")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_metrics(outdir: Path) -> pd.DataFrame:
    p = outdir / "metrics_summary.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing metrics_summary.csv in {outdir}")
    df = pd.read_csv(p)
    return df


def load_preds_from(outdir: Path, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    tag = tagify(model_name)
    p = outdir / f"preds_{tag}.csv"
    if not p.exists():
        # fallback loose search (handles parentheses variations)
        cands = list(outdir.glob(f"preds_{tag}*.csv"))
        if not cands:
            raise FileNotFoundError(f"No preds file for {model_name} in {outdir}")
        p = cands[0]
    df = pd.read_csv(p)
    y_true = pd.to_numeric(df["y_true"], errors="coerce").to_numpy()
    y_prob = pd.to_numeric(df["y_prob"], errors="coerce").to_numpy()
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    return y_true[mask].astype(int), y_prob[mask].astype(float)


def combined_metrics_bars(metrics_map: Dict[str, pd.DataFrame], outdir: Path):
    sns.set(style="whitegrid", context="paper")
    metric_cols = [
        ("roc_auc", "ROC AUC (↑)"),
        ("accuracy", "Accuracy (↑)"),
        ("avg_precision", "Average Precision (↑)"),
        ("f1", "F1 (↑)"),
        ("brier", "Brier (↓)"),
        ("log_loss", "Log Loss (↓)"),
    ]
    # Build long DF: dataset, model, metric, value
    frames = []
    for ds, df in metrics_map.items():
        m = df.copy()
        m["dataset"] = ds
        frames.append(m)
    allm = pd.concat(frames, ignore_index=True)

    # Save combined CSV/MD
    allm.to_csv(outdir / "combined_metrics.csv", index=False)
    try:
        (outdir / "combined_metrics.md").write_text(allm.to_markdown(index=False), encoding="utf-8")
    except Exception:
        pass

    # For each metric, clustered bars by model, hue=dataset
    for col, title in metric_cols:
        if col not in allm.columns:
            continue
        g = allm[["dataset", "model", col]].dropna()
        if g.empty:
            continue
        # sort models by mean performance
        order = g.groupby("model")[col].mean().sort_values(ascending=(col in {"brier", "log_loss"})).index.tolist()
        plt.figure(figsize=(10, max(4, 0.35 * len(order) + 2)))
        ax = sns.barplot(data=g, x=col, y="model", hue="dataset", order=order, palette="tab10")
        ax.set_title(f"{title} — Combined")
        ax.set_xlabel("")
        ax.set_ylabel("")
        # annotate values
        for p in ax.patches:
            width = p.get_width()
            y = p.get_y() + p.get_height() / 2
            ax.text(width, y, f" {width:.3f}", va="center", ha="left", fontsize=7)
        plt.tight_layout()
        plt.savefig(outdir / f"bars__{col}.png", dpi=300)
        plt.savefig(outdir / f"bars__{col}.pdf")
        plt.close()


def combined_roc_pr_per_model(outdirs: Dict[str, Path], metrics_intersection: List[str], outdir: Path):
    sns.set(style="whitegrid", context="paper")
    colors = sns.color_palette("tab10", n_colors=len(outdirs))
    ds_names = list(outdirs.keys())

    for model_name in metrics_intersection:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3.4))
        # ROC overlay
        for i, ds in enumerate(ds_names):
            try:
                y_true, y_prob = load_preds_from(outdirs[ds], model_name)
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc_val = roc_auc_score(y_true, y_prob)
                axes[0].plot(fpr, tpr, lw=2, color=colors[i], label=f"{ds} (AUC={auc_val:.3f})")
            except Exception:
                continue
        axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
        axes[0].set_title("ROC")
        axes[0].set_xlabel("FPR")
        axes[0].set_ylabel("TPR")
        axes[0].legend(fontsize=8, loc="lower right")

        # PR overlay
        for i, ds in enumerate(ds_names):
            try:
                y_true, y_prob = load_preds_from(outdirs[ds], model_name)
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                ap = average_precision_score(y_true, y_prob)
                base = y_true.mean() if len(y_true) else 0.0
                axes[1].plot(recall, precision, lw=2, color=colors[i], label=f"{ds} (AP={ap:.3f})")
                axes[1].hlines(base, 0, 1, colors=colors[i], linestyles=":", alpha=0.4)
            except Exception:
                continue
        axes[1].set_title("Precision-Recall")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].legend(fontsize=8, loc="lower left")

        fig.suptitle(model_name)
        fig.tight_layout()
        tag = tagify(model_name)
        plt.savefig(outdir / f"overlay__{tag}.png", dpi=300)
        plt.savefig(outdir / f"overlay__{tag}.pdf")
        plt.close(fig)


def combined_confusions_per_model(outdirs: Dict[str, Path], metrics_intersection: List[str], outdir: Path):
    sns.set(style="white", context="paper")
    ds_names = list(outdirs.keys())
    for model_name in metrics_intersection:
        n = len(ds_names)
        fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(3.2 * n, 3.2))
        if n == 1:
            axes = [axes]
        for i, ds in enumerate(ds_names):
            ax = axes[i]
            try:
                y_true, y_prob = load_preds_from(outdirs[ds], model_name)
                # Pick threshold by max F1 per dataset
                ts = np.linspace(0.01, 0.99, 99)
                best_t, best_f1 = 0.5, -np.inf
                for t in ts:
                    y_pred = (y_prob >= t).astype(int)
                    # safe F1
                    tp = ((y_true == 1) & (y_pred == 1)).sum()
                    fp = ((y_true == 0) & (y_pred == 1)).sum()
                    fn = ((y_true == 1) & (y_pred == 0)).sum()
                    pr = tp / (tp + fp + 1e-9)
                    rc = tp / (tp + fn + 1e-9)
                    f1 = 2 * pr * rc / (pr + rc + 1e-9)
                    if f1 > best_f1:
                        best_f1, best_t = f1, float(t)
                y_pred = (y_prob >= best_t).astype(int)
                cm = confusion_matrix(y_true, y_pred)
                disp = ConfusionMatrixDisplay(cm)
                disp.plot(ax=ax, colorbar=False)
                ax.set_title(f"{ds}\nT={best_t:.2f}")
            except Exception:
                ax.axis("off")
                ax.set_title(f"{ds}: no data")
        fig.suptitle(model_name)
        fig.tight_layout()
        tag = tagify(model_name)
        plt.savefig(outdir / f"confusions__{tag}.png", dpi=300)
        plt.savefig(outdir / f"confusions__{tag}.pdf")
        plt.close(fig)


def run_for_dataset(name: str, input_csv: Path, base_out: Path, skip_if_present: bool = False):
    # Point config to this dataset
    config.INPUT_CSV = str(input_csv)
    ds_out = base_out / name
    config.OUTPUT_DIR = ds_out
    ensure_dir(ds_out)

    # If already ran and skipping is requested, skip heavy work
    if skip_if_present and (ds_out / "metrics_summary.csv").exists():
        print(f"[SKIP] {name}: metrics already present at {ds_out}")
    else:
        print(f"[RUN] {name}: input={input_csv}")
        run_once()  # runs all 9 models and saves preds + metrics

    # Per-dataset plots using existing pipeline
    print(f"[PLOTS] {name}")
    pr.main()  # uses current config.OUTPUT_DIR


def main():
    base_dir = Path(r"c:/Users/pdaadh/Desktop/Study 2")
    default_assist = base_dir / "assistments_09_10_itemwise.csv"
    default_digi = base_dir / "DigiArvi_25_itemwise.csv"

    parser = argparse.ArgumentParser(description="Run 9 KT models on two datasets and create combined plots.")
    parser.add_argument("--assistments", type=str, default=str(default_assist), help="Path to Assistments itemwise CSV")
    parser.add_argument("--digi", type=str, default=str(default_digi), help="Path to DigiArvi itemwise CSV")
    parser.add_argument("--skip-if-present", action="store_true", help="Skip re-running a dataset if metrics already exist")
    args = parser.parse_args()

    assist_csv = Path(args.assistments)
    digi_csv = Path(args.digi)
    if not assist_csv.exists():
        raise FileNotFoundError(f"Assistments CSV not found: {assist_csv}")
    if not digi_csv.exists():
        raise FileNotFoundError(f"DigiArvi CSV not found: {digi_csv}")

    # Where all outputs go
    root_out = ensure_dir(base_dir / "kt_benchmark" / "output")

    # Run both datasets
    runs = {
        "Assistments_09_10": assist_csv,
        "DigiArvi_25": digi_csv,
    }
    for name, path in runs.items():
        run_for_dataset(name=name, input_csv=path, base_out=root_out, skip_if_present=args.skip_if_present)

    # Load metrics for combined analysis
    metrics_map: Dict[str, pd.DataFrame] = {}
    outdirs: Dict[str, Path] = {}
    for name in runs.keys():
        ds_dir = root_out / name
        try:
            metrics_map[name] = load_metrics(ds_dir)
            outdirs[name] = ds_dir
        except Exception as e:
            print(f"[WARN] Could not load metrics for {name}: {e}")

    if len(metrics_map) < 2:
        print("[WARN] Need both datasets' metrics to build combined plots. Exiting.")
        return

    # Combined outputs directory
    combo_dir = ensure_dir(root_out / "combined")
    plots_dir = ensure_dir(combo_dir / "plots")

    # Save a quick combined markdown table with timestamp
    frames = []
    for ds, df in metrics_map.items():
        t = df.copy()
        t["dataset"] = ds
        frames.append(t)
    allm = pd.concat(frames, ignore_index=True)
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    try:
        md = allm.to_markdown(index=False)
    except Exception:
        md = allm.to_csv(index=False)
    (combo_dir / "README.md").write_text(f"# Combined Results\n\nGenerated: {stamp}\n\n" + md + "\n", encoding="utf-8")

    # Metric bars (grouped by dataset)
    combined_metrics_bars(metrics_map, plots_dir)

    # Determine which models appear in both metrics tables
    common_models = sorted(set(metrics_map[list(metrics_map.keys())[0]]["model"]).intersection(
        set(metrics_map[list(metrics_map.keys())[1]]["model"]) ))

    # Overlay ROC/PR per model across datasets
    combined_roc_pr_per_model(outdirs, common_models, plots_dir)

    # Side-by-side confusion matrices per model
    combined_confusions_per_model(outdirs, common_models, plots_dir)

    print("Saved combined outputs to:", combo_dir.resolve())


if __name__ == "__main__":
    main()
