from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from . import config
from . import run_benchmark as single_run
from . import plot_results as per_dataset_plots


# --------------------------
# Helpers
# --------------------------

def _tag_from_path(p: Path) -> str:
    base = p.name.replace('.csv', '')
    # keep alphanum and underscore only
    clean = ''.join(ch if ch.isalnum() or ch in ['_', '-'] else '_' for ch in base)
    return clean.lower()


def _dataset_label_from_path(p: Path) -> str:
    """Human-friendly dataset label for legends/CSV that removes the word 'itemwise'."""
    base = p.stem
    # Remove 'itemwise' with optional preceding separators, case-insensitive
    s = re.sub(r"(?i)(?:^|[ _\-])itemwise(?:$)?", "", base)
    # Collapse remaining underscores/hyphens to spaces
    s = re.sub(r"[_\-]+", " ", s)
    return s.strip()


def _set_dataset_for_run(dataset_csv: Path, out_subdir: Path):
    # Mutate global config in-place so downstream modules read new paths
    config.INPUT_CSV = dataset_csv
    config.OUTPUT_DIR = out_subdir
    out_subdir.mkdir(parents=True, exist_ok=True)


def _run_one_dataset(dataset_csv: Path) -> Path:
    tag = _tag_from_path(dataset_csv)
    outdir = config.BASE_DIR / 'kt_benchmark' / 'output' / tag
    _set_dataset_for_run(dataset_csv, outdir)
    print(f"\n=== Running benchmark for dataset: {dataset_csv.name} ===")
    single_run.main()
    print(f"=== Generating per-dataset plots for: {dataset_csv.name} ===")
    per_dataset_plots.main()
    return outdir


# --------------------------
# Comparative plotting
# --------------------------

COMPARATIVE_METRICS = [
    ('roc_auc', 'ROC AUC (↑)'),
    ('accuracy', 'Accuracy (↑)'),
    ('avg_precision', 'Average Precision (↑)'),
    ('f1', 'F1 (↑)'),
]


def _load_summary(outdir: Path) -> pd.DataFrame:
    p = outdir / 'metrics_summary.csv'
    if not p.exists():
        raise FileNotFoundError(f"Missing metrics_summary.csv in {outdir}")
    df = pd.read_csv(p)
    # enforce numeric
    for col, _ in COMPARATIVE_METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _combine_summaries(out1: Path, label1: str, out2: Path, label2: str) -> pd.DataFrame:
    df1 = _load_summary(out1).copy()
    df1['dataset'] = label1
    df2 = _load_summary(out2).copy()
    df2['dataset'] = label2
    comb = pd.concat([df1, df2], ignore_index=True, sort=False)
    return comb


def _plot_grouped_bars(comb: pd.DataFrame, compare_dir: Path):
    sns.set(style='whitegrid', context='paper')
    models = sorted(comb['model'].dropna().unique().tolist())
    datasets = sorted(comb['dataset'].dropna().unique().tolist())

    # Determine common models and adjust figure height to prevent overlap
    modelsets_by_ds = {ds: set(comb.loc[comb['dataset'] == ds, 'model'].dropna().unique()) for ds in datasets}
    common_models_global = sorted(set.intersection(*modelsets_by_ds.values())) if len(datasets) >= 2 else models
    n_models = max(1, len(common_models_global))

    nrows, ncols = 2, 2
    height = max(8.0, 0.4 * n_models)  # dynamic height based on number of models
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, height))
    axes = np.array(axes).reshape(nrows, ncols)

    handles_labels = None
    for (metric, title), ax in zip(COMPARATIVE_METRICS, axes.ravel()):
        if metric not in comb.columns:
            ax.axis('off')
            continue
        dfp = comb[['model', 'dataset', metric]].dropna()
        # Keep models present in both datasets for fair comparison
        common_models = sorted(
            set(dfp.loc[dfp['dataset'] == datasets[0], 'model']).intersection(
                set(dfp.loc[dfp['dataset'] == datasets[1], 'model'])
            )
        ) if len(datasets) >= 2 else sorted(dfp['model'].unique())
        dfp = dfp[dfp['model'].isin(common_models)]
        # Sort models by mean metric across datasets
        order = (
            dfp.groupby('model')[metric]
            .mean()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
        sns.barplot(data=dfp, x=metric, y='model', hue='dataset', hue_order=datasets, order=order, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('')
        ax.set_ylabel('')
        # Improve readability
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=8)
        if len(order) <= 15:
            for c in ax.containers:
                ax.bar_label(c, fmt='%.3f', fontsize=8, padding=2)
        # Defer legend: collect from the first axis only, then use a single figure legend
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()
        ax.legend_.remove() if ax.get_legend() else None

    fig.suptitle('Comparative Model Performance (Test Set): Grouped Bars', y=0.995)
    # Single shared legend at bottom center
    if handles_labels and len(handles_labels[0]) > 0:
        fig.legend(handles_labels[0], handles_labels[1], loc='lower center', ncol=len(datasets), title='Dataset', title_fontsize=9, fontsize=8)
        fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    (compare_dir / 'comparative_grouped_bars.png').parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(compare_dir / 'comparative_grouped_bars.png', dpi=300, bbox_inches='tight')
    fig.savefig(compare_dir / 'comparative_grouped_bars.pdf', bbox_inches='tight')
    plt.close(fig)


def _pick_top_models_for_radar(comb: pd.DataFrame, k: int = 3) -> List[str]:
    # Rank models by average of available comparative metrics across datasets
    dfp = comb.dropna(subset=[m for m, _ in COMPARATIVE_METRICS]).copy()
    if dfp.empty:
        return []
    # average per model across datasets and metrics
    scores = (
        dfp.groupby('model')[[m for m, _ in COMPARATIVE_METRICS]].mean().mean(axis=1)
    )
    top = scores.sort_values(ascending=False).head(k).index.tolist()
    return top


def _radar_factory(num_vars: int):
    # Based on Matplotlib radar example
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection
    import matplotlib.spines as spines

    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def fill(self, *args, **kwargs):
            closed = kwargs.pop('closed', True)
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                line.set_clip_on(False)
            return lines

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

    register_projection(RadarAxes)
    return theta


def _plot_radar(comb: pd.DataFrame, compare_dir: Path, tag1: str, tag2: str, top_k: int = 3):
    metrics = [m for m, _ in COMPARATIVE_METRICS]
    metric_labels = [lbl for _, lbl in COMPARATIVE_METRICS]
    datasets = [tag1, tag2]
    top_models = _pick_top_models_for_radar(comb, k=top_k)
    if not top_models:
        print('[Radar] No models available for radar chart.')
        return

    theta = _radar_factory(len(metrics))

    sns.set(style='whitegrid', context='paper')
    nrows = int(np.ceil(len(top_models) / 2))
    ncols = 2 if len(top_models) > 1 else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows), subplot_kw=dict(projection='radar'))
    axes = np.array(axes).reshape(nrows, ncols)

    colors = sns.color_palette('Set2', n_colors=2)

    for i, model in enumerate(top_models):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        vals = {}
        for ds in datasets:
            row = comb[(comb['model'] == model) & (comb['dataset'] == ds)][metrics]
            if row.empty:
                vals[ds] = np.zeros(len(metrics))
            else:
                vals[ds] = row.iloc[0].to_numpy(dtype=float)
                vals[ds] = np.nan_to_num(vals[ds], nan=0.0)

        # Close the polygon by repeating first value
        for j, ds in enumerate(datasets):
            data = np.r_[vals[ds], vals[ds][0]]
            ax.plot(np.r_[theta, theta[0]], data, color=colors[j], lw=2, label=ds)
            ax.fill(np.r_[theta, theta[0]], data, color=colors[j], alpha=0.25)

        ax.set_title(model)
        ax.set_ylim(0, 1)
        ax.set_varlabels(metric_labels)
        ax.tick_params(labelsize=9, pad=8)

    # Remove empty axes if any
    total = nrows * ncols
    for j in range(len(top_models), total):
        r = j // ncols
        c = j % ncols
        fig.delaxes(axes[r, c])

    handles, labels = axes[0, 0].get_legend_handles_labels() if axes.size else ([], [])
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=2)
    fig.suptitle('Radar Charts: Top Models across Datasets', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(compare_dir / 'radar_top_models.png', dpi=300, bbox_inches='tight')
    fig.savefig(compare_dir / 'radar_top_models.pdf', bbox_inches='tight')
    plt.close(fig)


def _plot_radar_datasetwise_all_models(comb: pd.DataFrame, compare_dir: Path, tag1: str, tag2: str):
    """
    For each dataset, plot ONE radar chart that overlays ALL models across the comparative metrics.
    This addresses the request for dataset-wise radar with all models in one.
    """
    metrics = [m for m, _ in COMPARATIVE_METRICS]
    metric_labels = [lbl for _, lbl in COMPARATIVE_METRICS]
    datasets = [tag1, tag2]
    theta = _radar_factory(len(metrics))

    for ds in datasets:
        sub = comb[comb['dataset'] == ds].copy()
        models = sorted(sub['model'].dropna().unique().tolist())
        if not models:
            continue

        # Dynamic figure size to accommodate legend and prevent overlaps
        height = 7 + 0.2 * min(30, len(models))
        fig, ax = plt.subplots(figsize=(8, height), subplot_kw=dict(projection='radar'))
        colors = sns.color_palette('tab20', n_colors=max(3, len(models)))

        for i, mdl in enumerate(models):
            row = sub[sub['model'] == mdl][metrics]
            if row.empty:
                vals = np.zeros(len(metrics))
            else:
                vals = row.iloc[0].to_numpy(dtype=float)
                vals = np.nan_to_num(vals, nan=0.0)
            data = np.r_[vals, vals[0]]
            ax.plot(np.r_[theta, theta[0]], data, color=colors[i % len(colors)], lw=1.6, alpha=0.9, label=mdl)
            # avoid fill to reduce clutter when many models

        # Centered overall title as figure suptitle
        fig.suptitle(f"Radar: {ds} (All Models)", y=0.98)
        ax.set_ylim(0, 1)
        ax.set_varlabels(metric_labels)
        ax.tick_params(labelsize=9, pad=8)

        # Legend at bottom center with multiple columns if many models
        ncol = 2 if len(models) <= 14 else 3 if len(models) <= 30 else 4
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', ncol=ncol, title='Models', title_fontsize=9, fontsize=8)

        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig.savefig(compare_dir / f'radar_all_models__{ds}.png', dpi=300, bbox_inches='tight')
        fig.savefig(compare_dir / f'radar_all_models__{ds}.pdf', bbox_inches='tight')
        plt.close(fig)


def _plot_radar_datasetwise_all_models_combined(comb: pd.DataFrame, compare_dir: Path, tag1: str, tag2: str):
    """One figure with two radar subplots: left=dataset1, right=dataset2; single legend at bottom."""
    metrics = [m for m, _ in COMPARATIVE_METRICS]
    metric_labels = [lbl for _, lbl in COMPARATIVE_METRICS]
    datasets = [tag1, tag2]
    theta = _radar_factory(len(metrics))

    # Prepare data
    subs = [comb[comb['dataset'] == ds].copy() for ds in datasets]
    models_list = [sorted(s['model'].dropna().unique().tolist()) for s in subs]
    if not any(models_list):
        return

    # Figure
    height = 6 + 0.15 * max(len(m) for m in models_list)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, height), subplot_kw=dict(projection='radar'))
    colors = sns.color_palette('tab20', n_colors=max(6, max(len(m) for m in models_list)))

    shared_handles = []
    shared_labels = []
    for idx, (ds, sub, ax, models) in enumerate(zip(datasets, subs, axes, models_list)):
        for i, mdl in enumerate(models):
            row = sub[sub['model'] == mdl][metrics]
            if row.empty:
                vals = np.zeros(len(metrics))
            else:
                vals = row.iloc[0].to_numpy(dtype=float)
                vals = np.nan_to_num(vals, nan=0.0)
            data = np.r_[vals, vals[0]]
            h = ax.plot(np.r_[theta, theta[0]], data, color=colors[i % len(colors)], lw=1.6, alpha=0.95, label=mdl)[0]
            if idx == 0:
                shared_handles.append(h)
                shared_labels.append(mdl)
        ax.set_title(ds)
        ax.set_ylim(0, 1)
        ax.set_varlabels(metric_labels)
        ax.tick_params(labelsize=9, pad=8)

    fig.suptitle('Radar: All Models per Dataset', y=0.995)
    # Bottom shared legend
    n_models_total = len(shared_labels)
    ncol = 3 if n_models_total <= 18 else 4 if n_models_total <= 28 else 5
    if shared_handles:
        fig.legend(shared_handles, shared_labels, loc='lower center', ncol=ncol, title='Models', title_fontsize=9, fontsize=8)
        fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(compare_dir / 'radar_all_models__both.png', dpi=300, bbox_inches='tight')
    fig.savefig(compare_dir / 'radar_all_models__both.pdf', bbox_inches='tight')
    plt.close(fig)


# --------------------------
# Main Orchestration
# --------------------------

def main(dataset_paths: List[Path] | None = None, reuse: bool = False):
    # Resolve datasets
    if not dataset_paths:
        # Fallback to two most relevant CSVs if present
        candidates = [
            config.BASE_DIR / 'assistments_09_10_itemwise.csv',
            config.BASE_DIR / 'DigiArvi_25_itemwise.csv',
            config.BASE_DIR / 'EQTd_DAi_25_itemwise_with_10_quartile.csv',
        ]
        present = [p for p in candidates if p.exists()]
        if len(present) < 2:
            raise RuntimeError('Please provide at least two dataset CSV paths to compare.')
        # Pick first two
        dataset_paths = present[:2]

    if len(dataset_paths) != 2:
        raise ValueError('Exactly two dataset CSVs are required.')

    ds1, ds2 = [Path(p).resolve() for p in dataset_paths]
    tag1, tag2 = _tag_from_path(ds1), _tag_from_path(ds2)
    label1, label2 = _dataset_label_from_path(ds1), _dataset_label_from_path(ds2)

    # Run each dataset end-to-end (or reuse existing outputs)
    if reuse:
        # infer output dirs by tag and validate presence
        out1 = config.BASE_DIR / 'kt_benchmark' / 'output' / tag1
        out2 = config.BASE_DIR / 'kt_benchmark' / 'output' / tag2
        if not (out1 / 'metrics_summary.csv').exists() or not (out2 / 'metrics_summary.csv').exists():
            raise FileNotFoundError('Reuse requested, but per-dataset results not found. Re-run without --reuse to generate them.')
        # Also regenerate per-dataset plots using latest plotting code
        _set_dataset_for_run(ds1, out1)
        per_dataset_plots.main()
        _set_dataset_for_run(ds2, out2)
        per_dataset_plots.main()
    else:
        out1 = _run_one_dataset(ds1)
        out2 = _run_one_dataset(ds2)

    # Comparative directory
    compare_dir = config.BASE_DIR / 'kt_benchmark' / 'output' / f'compare_{tag1}_vs_{tag2}'
    compare_dir.mkdir(parents=True, exist_ok=True)

    # Combine summaries and save
    comb = _combine_summaries(out1, label1, out2, label2)
    comb.to_csv(compare_dir / 'metrics_summary_both.csv', index=False)

    # Grouped bar plots (radar charts disabled per request)
    _plot_grouped_bars(comb, compare_dir)

    print('Saved comparative outputs to:', compare_dir.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run KT benchmark on two datasets and generate comparative plots (including radar charts).')
    parser.add_argument('--datasets', nargs=2, metavar=('CSV1', 'CSV2'), help='Paths to two dataset CSVs to compare')
    parser.add_argument('--reuse', action='store_true', help='Reuse existing per-dataset results and regenerate comparative plots only')
    args = parser.parse_args()
    ds_paths = [Path(p) for p in args.datasets] if args.datasets else None
    main(ds_paths, reuse=bool(args.reuse))
