import os
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=UserWarning)
sns.set(context="notebook", style="whitegrid")


def ensure_outdir(base: Path) -> Path:
    outdir = base / "eda_output"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def load_data(path: Path) -> pd.DataFrame:
    print(f"Reading: {path}")
    df = pd.read_csv(path)
    # Basic cleaning: standardize column names just in case
    df.columns = [c.strip() for c in df.columns]
    return df


def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = []
    categorical_cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols


def numeric_descriptive_stats(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    # Compute requested stats
    stats = []
    percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors='coerce')
        desc = {
            'feature': c,
            'count': s.count(),
            'mean': s.mean(),
            'median': s.median(),
            'variance': s.var(ddof=1),
            'std': s.std(ddof=1),
            'skewness': s.skew(),
            'kurtosis': s.kurt(),
            'min': s.min(),
            'max': s.max(),
        }
        for p in percentiles:
            desc[f"p{int(p*100)}"] = s.quantile(p)
        stats.append(desc)
    return pd.DataFrame(stats)


def categorical_descriptive_stats(df: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    card_rows = []
    freq_frames = []
    for c in categorical_cols:
        vc = df[c].astype('category')
        card_rows.append({
            'feature': c,
            'cardinality': int(vc.nunique(dropna=True)),
            'num_missing': int(vc.isna().sum())
        })
        freq = vc.value_counts(dropna=False).rename('count').reset_index().rename(columns={'index': c})
        freq['feature'] = c
        freq_frames.append(freq)
    cardinality_df = pd.DataFrame(card_rows)
    freq_df = pd.concat(freq_frames, ignore_index=True) if freq_frames else pd.DataFrame(columns=['feature','value','count'])
    return cardinality_df, freq_df


def class_balance(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns:
        return pd.DataFrame()
    s = df[target_col]
    if not pd.api.types.is_numeric_dtype(s):
        # try to map to numeric if typical labels
        mapping = {"yes":1, "true":1, "y":1, "no":0, "false":0, "n":0}
        s = s.astype(str).str.lower().map(mapping)
    counts = s.value_counts(dropna=False)
    perc = (counts / counts.sum() * 100.0).round(2)
    out = pd.DataFrame({'class': counts.index, 'count': counts.values, 'percent': perc.values})
    return out


def save_dataframe(df: pd.DataFrame, outpath: Path):
    if df is None or df.empty:
        print(f"No data to save for {outpath.name}")
        return
    df.to_csv(outpath, index=False)
    print(f"Saved: {outpath}")


def plot_hist_kde(series: pd.Series, title: str, outpath: Path):
    plt.figure(figsize=(8,5))
    sns.histplot(series.dropna(), kde=True, bins=30, color='#4472C4')
    plt.title(title)
    plt.xlabel(series.name)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def dimensionality_reduction_and_feature_importance(df: pd.DataFrame,
                                                    numeric_cols: List[str],
                                                    categorical_cols: List[str],
                                                    target_col: str,
                                                    outdir: Path):
    """
    Build a one-hot encoded feature matrix, run PCA (unsupervised) to estimate
    how many components are needed for a chosen variance threshold, and (if a
    binary target is available) fit a RandomForest to compute feature importances.

    Saves:
      - dr_pca_explained_variance.csv
      - dr_pca_summary.txt
      - dr_pca_loadings.csv
      - dr_supervised_feature_importance.csv (if target available)
      - dr_feature_importance_summary.txt (if target available)
    """
    # Build design matrix similar to pca_and_clustering
    use_cols = numeric_cols + categorical_cols
    tmp = df[use_cols].copy()

    # Coerce numerics
    for c in numeric_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors='coerce')

    # One-hot encode categoricals
    tmp = pd.get_dummies(tmp, columns=categorical_cols, dummy_na=True)

    # Drop columns with all NaN, then fill remaining NaN with column mean
    tmp = tmp.dropna(axis=1, how='all')
    tmp = tmp.fillna(tmp.mean(numeric_only=True))

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(tmp.values)
    feature_names = list(tmp.columns)

    # PCA on the full feature space
    pca = PCA(n_components=None, random_state=42)
    X_pca = pca.fit_transform(X)

    var = pca.explained_variance_ratio_
    cumvar = np.cumsum(var)
    pcs = np.arange(1, len(var) + 1)
    pca_df = pd.DataFrame({
        'PC': pcs,
        'explained_variance_ratio': var,
        'cumulative_variance_ratio': cumvar,
    })
    pca_df.to_csv(outdir / 'dr_pca_explained_variance.csv', index=False)

    # Scree plot (explained variance per component)
    plt.figure(figsize=(8,5))
    plt.plot(pcs, var, marker='o')
    plt.title('PCA Scree Plot (Explained Variance per PC)')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / 'dr_pca_scree.png', dpi=150)
    plt.close()

    # Cumulative variance plot with thresholds
    plt.figure(figsize=(8,5))
    plt.plot(pcs, cumvar, marker='o', label='Cumulative variance')
    for thr, n in [(0.80, None), (0.90, None), (0.95, None)]:
        plt.axhline(thr, color='gray', linestyle='--', linewidth=1)
    # Compute counts for vlines after cumvar is defined
    def n_for(thr: float) -> int:
        return int(np.searchsorted(cumvar, thr) + 1) if len(cumvar) else 0
    n80, n90, n95 = n_for(0.80), n_for(0.90), n_for(0.95)
    for thr, n in [(0.80, n80), (0.90, n90), (0.95, n95)]:
        if n:
            plt.axvline(n, color='red', linestyle=':', linewidth=1)
            plt.text(n, thr + 0.02, f'{int(thr*100)}% @ {n}', color='red')
    plt.ylim(0, 1.02)
    plt.title('PCA Cumulative Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Variance Ratio')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / 'dr_pca_cumulative.png', dpi=150)
    plt.close()

    # Summary for common thresholds
    # n80, n90, n95 computed above for plots
    with open(outdir / 'dr_pca_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"PCA components needed for variance thresholds\n")
        f.write(f"80%: {n80}\n90%: {n90}\n95%: {n95}\n")

    # Loadings: features x components
    loadings = pd.DataFrame(pca.components_.T, index=feature_names, columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])])
    loadings.to_csv(outdir / 'dr_pca_loadings.csv')

    # Biplot for top loading features on PC1/PC2
    if loadings.shape[1] >= 2:
        pc1 = loadings['PC1']
        pc2 = loadings['PC2']
        strength = np.sqrt(pc1.values**2 + pc2.values**2)
        top_n = 12
        idx = np.argsort(-strength)[:top_n]
        plt.figure(figsize=(8,8))
        # Draw unit circle for reference
        circle = plt.Circle((0,0), 1.0, color='lightgray', fill=False, linestyle='--')
        ax = plt.gca()
        ax.add_artist(circle)
        scale = 1.0  # loadings already scaled for standardized features
        for i in idx:
            x, y = pc1.values[i]*scale, pc2.values[i]*scale
            plt.arrow(0, 0, x, y, head_width=0.02, head_length=0.03, fc='tab:blue', ec='tab:blue', alpha=0.8, length_includes_head=True)
            plt.text(x*1.08, y*1.08, loadings.index[i], fontsize=8)
        plt.axhline(0, color='black', linewidth=1)
        plt.axvline(0, color='black', linewidth=1)
        plt.title('PCA Biplot: Top Loadings on PC1/PC2')
        plt.xlabel('PC1 loadings')
        plt.ylabel('PC2 loadings')
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(outdir / 'dr_pca_biplot_top12.png', dpi=150)
        plt.close()

    # Supervised feature importance (if valid target available)
    if target_col and target_col in df.columns:
        y = pd.to_numeric(df[target_col], errors='coerce')
        # Restrict to rows with valid target (0/1) and not NaN
        mask_valid = y.isin([0, 1])
        X_valid = X[mask_valid.values]
        y_valid = y[mask_valid].astype(int).values
        if len(y_valid) >= 10 and X_valid.shape[0] == len(y_valid):
            rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
            rf.fit(X_valid, y_valid)
            imp = rf.feature_importances_
            imp_df = pd.DataFrame({'feature': feature_names, 'importance': imp}).sort_values('importance', ascending=False)
            imp_df.to_csv(outdir / 'dr_supervised_feature_importance.csv', index=False)

            # Bar plot for top-N features
            top_n = 20 if len(imp_df) >= 20 else len(imp_df)
            top_df = imp_df.head(top_n).iloc[::-1]  # reverse for horizontal bar from smallest to largest
            plt.figure(figsize=(8, max(5, top_n * 0.35)))
            plt.barh(top_df['feature'], top_df['importance'], color='#5B9BD5')
            plt.title(f'RandomForest Feature Importance (Top {top_n})')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(outdir / 'dr_rf_feature_importance_topN.png', dpi=150)
            plt.close()

            # Cumulative importance curve
            plt.figure(figsize=(8,5))
            cum = imp_df['importance'].cumsum() / imp_df['importance'].sum()
            plt.plot(range(1, len(cum)+1), cum, marker='o')
            plt.axhline(0.90, color='gray', linestyle='--')
            k90 = int(np.searchsorted(cum.values, 0.90) + 1) if len(cum) else 0
            if k90:
                plt.axvline(k90, color='red', linestyle=':')
                plt.text(k90, 0.92, f'90% @ {k90}', color='red')
            plt.ylim(0, 1.02)
            plt.title('Cumulative RandomForest Feature Importance')
            plt.xlabel('Number of top features')
            plt.ylabel('Cumulative importance')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(outdir / 'dr_rf_feature_importance_cumulative.png', dpi=150)
            plt.close()

            # Number of top features to reach 90% cumulative importance
            cumsum = imp_df['importance'].cumsum() / imp_df['importance'].sum()
            k90 = int(np.searchsorted(cumsum.values, 0.90) + 1) if len(cumsum) else 0
            with open(outdir / 'dr_feature_importance_summary.txt', 'w', encoding='utf-8') as f:
                f.write(f"Top features to reach 90% cumulative RF importance: {k90}\n")

            print(f"[DR] PCA n80={n80}, n90={n90}, n95={n95} | RF top-k for 90% importance: {k90}")
        else:
            print(f"[DR] Skipped supervised feature importance (insufficient valid target rows). PCA n80={n80}, n90={n90}, n95={n95}")
    else:
        print(f"[DR] No target column provided; reported PCA only. n80={n80}, n90={n90}, n95={n95}")

def plot_box(series: pd.Series, title: str, outpath: Path):
    plt.figure(figsize=(6,5))
    sns.boxplot(x=series.dropna(), color='#ED7D31')
    plt.title(title)
    plt.xlabel(series.name)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_bar_counts(df: pd.DataFrame, col: str, title: str, outpath: Path):
    plt.figure(figsize=(8,5))
    vc = df[col].value_counts(dropna=False)
    sns.barplot(x=vc.index.astype(str), y=vc.values, color='#70AD47')
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_box_by_category(df: pd.DataFrame, num_col: str, cat_col: str, title: str, outpath: Path):
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x=cat_col, y=num_col)
    plt.title(title)
    plt.xlabel(cat_col)
    plt.ylabel(num_col)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_violin_by_category(df: pd.DataFrame, num_col: str, cat_col: str, title: str, outpath: Path):
    plt.figure(figsize=(10,6))
    sns.violinplot(data=df, x=cat_col, y=num_col, inner='box')
    plt.title(title)
    plt.xlabel(cat_col)
    plt.ylabel(num_col)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_strip_by_category(df: pd.DataFrame, num_col: str, cat_col: str, title: str, outpath: Path):
    plt.figure(figsize=(10,6))
    sns.stripplot(data=df, x=cat_col, y=num_col, alpha=0.4, jitter=0.25, color='#5B9BD5')
    plt.title(title)
    plt.xlabel(cat_col)
    plt.ylabel(num_col)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_scatter_num_vs_binary(df: pd.DataFrame, num_col: str, bin_col: str, title: str, outpath: Path):
    # Jitter the binary column for visualization
    plt.figure(figsize=(8,6))
    y = df[bin_col].astype(float)
    y_jitter = y + (np.random.rand(len(y)) - 0.5) * 0.1
    plt.scatter(df[num_col], y_jitter, alpha=0.2, s=10)
    plt.yticks([0,1], ['0','1'])
    plt.title(title)
    plt.xlabel(num_col)
    plt.ylabel(bin_col)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str], title: str, outpath: Path):
    corr = df[numeric_cols].corr(method='pearson')
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_pairplot(df: pd.DataFrame, numeric_cols: List[str], title: str, outpath: Path):
    pp = sns.pairplot(df[numeric_cols].dropna(), corner=True, diag_kind='hist')
    pp.fig.suptitle(title, y=1.02)
    pp.savefig(outpath, dpi=150)
    plt.close('all')


def pca_and_clustering(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str], color_col: str, outdir: Path):
    # Build feature matrix with one-hot for categoricals
    use_cols = numeric_cols + categorical_cols
    tmp = df[use_cols].copy()

    # Coerce numerics
    for c in numeric_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors='coerce')

    # One-hot encode categoricals
    tmp = pd.get_dummies(tmp, columns=categorical_cols, dummy_na=True)

    # Drop columns with all NaN
    tmp = tmp.dropna(axis=1, how='all')

    # Fill remaining NaN with column means (for numeric dummies will be 0/1 mostly)
    tmp = tmp.fillna(tmp.mean(numeric_only=True))

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(tmp.values)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    pca_df = pd.DataFrame({
        'PC1': X_pca[:,0],
        'PC2': X_pca[:,1],
    })
    if color_col in df.columns:
        pca_df[color_col] = df[color_col].values

    # Plot PCA colored by target if present
    plt.figure(figsize=(8,6))
    if color_col in pca_df.columns:
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=color_col, palette='tab10', alpha=0.6, s=30)
        plt.legend(title=color_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6, s=30)
    var_exp = pca.explained_variance_ratio_
    plt.title(f"PCA (PC1 {var_exp[0]:.1%}, PC2 {var_exp[1]:.1%})")
    plt.tight_layout()
    plt.savefig(outdir / "multivariate_pca_scatter.png", dpi=150)
    plt.close()

    # KMeans clustering and plot (use explicit n_init for compatibility)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    pca_df['cluster'] = clusters

    plt.figure(figsize=(8,6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='Set2', alpha=0.6, s=30)
    plt.title("KMeans Clusters (k=3) in PCA space")
    plt.legend(title='cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(outdir / "multivariate_kmeans_in_pca.png", dpi=150)
    plt.close()


def groupwise_aggregates(df: pd.DataFrame, num_col: str, group_col: str) -> pd.DataFrame:
    g = df.groupby(group_col, dropna=False)[num_col]
    agg = g.agg(['count','mean','std'])
    agg['sem'] = agg['std'] / np.sqrt(agg['count'].replace(0, np.nan))
    agg = agg.reset_index().rename(columns={group_col: 'group'})
    return agg


def plot_groupwise_bar_with_ci(agg_df: pd.DataFrame, group_name_col: str, mean_col: str, sem_col: str, title: str, outpath: Path):
    plt.figure(figsize=(12,6))
    x = np.arange(len(agg_df[group_name_col]))
    heights = agg_df[mean_col].values
    yerr = agg_df[sem_col].values
    plt.bar(x, heights, yerr=yerr, color='#A5A5A5', alpha=0.9, capsize=4)
    plt.title(title)
    plt.xlabel(group_name_col)
    plt.ylabel(mean_col)
    plt.xticks(x, agg_df[group_name_col].astype(str), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    base_dir = Path(__file__).parent
    default_csv = base_dir / "EQTd_DAi_25_itemwise_minimal.csv"

    # Allow passing a different CSV via CLI
    csv_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else default_csv
    if not csv_path.exists():
        print(f"ERROR: Input CSV not found: {csv_path}")
        sys.exit(1)

    outdir = ensure_outdir(base_dir)

    df = load_data(csv_path)

    # Identify conventional columns present in this dataset
    candidate_target = 'response' if 'response' in df.columns else None
    time_col = 'response_time_sec' if 'response_time_sec' in df.columns else None
    sex_col = 'sex' if 'sex' in df.columns else None
    group_col = 'group' if 'group' in df.columns else None

    numeric_cols, categorical_cols = split_columns(df)

    # Descriptive stats for numerics
    num_stats = numeric_descriptive_stats(df, numeric_cols)
    save_dataframe(num_stats, outdir / "numeric_descriptive_stats.csv")

    # Descriptive stats for categoricals
    card_df, freq_df = categorical_descriptive_stats(df, categorical_cols)
    save_dataframe(card_df, outdir / "categorical_cardinality.csv")
    save_dataframe(freq_df, outdir / "categorical_frequencies.csv")

    # Class balance
    if candidate_target is not None:
        balance = class_balance(df, candidate_target)
        save_dataframe(balance, outdir / "target_class_balance.csv")

    # Univariate plots
    if time_col is not None:
        plot_hist_kde(df[time_col], f"Histogram & KDE of {time_col}", outdir / f"univariate_hist_kde_{time_col}.png")
        plot_box(df[time_col], f"Boxplot of {time_col}", outdir / f"univariate_box_{time_col}.png")

    if candidate_target is not None:
        plot_bar_counts(df, candidate_target, f"Class Balance: {candidate_target}", outdir / f"univariate_bar_{candidate_target}.png")

    if sex_col is not None:
        plot_bar_counts(df, sex_col, f"Frequency: {sex_col}", outdir / f"univariate_bar_{sex_col}.png")

    if group_col is not None:
        plot_bar_counts(df, group_col, f"Frequency: {group_col}", outdir / f"univariate_bar_{group_col}.png")

    # Bivariate plots & comparisons
    if time_col is not None and candidate_target is not None and pd.api.types.is_numeric_dtype(df[candidate_target]):
        plot_scatter_num_vs_binary(df.dropna(subset=[time_col, candidate_target]), time_col, candidate_target,
                                   f"{time_col} vs {candidate_target}", outdir / f"bivariate_scatter_{time_col}_vs_{candidate_target}.png")

        plot_box_by_category(df, time_col, candidate_target,
                             f"{time_col} by {candidate_target}", outdir / f"bivariate_box_{time_col}_by_{candidate_target}.png")

    if time_col is not None and sex_col is not None:
        plot_box_by_category(df, time_col, sex_col, f"{time_col} by {sex_col}", outdir / f"bivariate_box_{time_col}_by_{sex_col}.png")
        plot_violin_by_category(df, time_col, sex_col, f"Violin: {time_col} by {sex_col}", outdir / f"bivariate_violin_{time_col}_by_{sex_col}.png")
        plot_strip_by_category(df, time_col, sex_col, f"Strip: {time_col} by {sex_col}", outdir / f"bivariate_strip_{time_col}_by_{sex_col}.png")

    if time_col is not None and group_col is not None:
        plot_box_by_category(df, time_col, group_col, f"{time_col} by {group_col}", outdir / f"bivariate_box_{time_col}_by_{group_col}.png")
        # Group-wise aggregates with CI
        agg = groupwise_aggregates(df.dropna(subset=[time_col]), time_col, group_col)
        save_dataframe(agg, outdir / f"groupwise_{time_col}_by_{group_col}.csv")
        plot_groupwise_bar_with_ci(agg, 'group', 'mean', 'sem', f"Mean {time_col} by {group_col} (±1 SEM)", outdir / f"bivariate_bar_ci_{time_col}_by_{group_col}.png")

    # Correlation heatmap on numeric columns
    if len(numeric_cols) >= 2:
        plot_correlation_heatmap(df, numeric_cols, "Correlation Heatmap (numeric features)", outdir / "bivariate_correlation_heatmap.png")

    # Pair plot for numeric features (may be small)
    if len(numeric_cols) >= 2:
        plot_pairplot(df, numeric_cols, "Pair Plot (numeric features)", outdir / "multivariate_pairplot.png")

    # Multivariate PCA & clustering using all features
    # Build inputs: keep candidate target as color label if exists
    cat_cols_for_ml = categorical_cols.copy()
    num_cols_for_ml = numeric_cols.copy()
    color_col = candidate_target if candidate_target is not None else (sex_col if sex_col is not None else (group_col if group_col is not None else None))

    # Exclude ID-like columns from features
    for id_like in ['IDCode', 'orig_order']:
        if id_like in num_cols_for_ml:
            num_cols_for_ml.remove(id_like)
        if id_like in cat_cols_for_ml:
            cat_cols_for_ml.remove(id_like)

    # Exclude the target column from features (avoid leakage)
    if candidate_target:
        if candidate_target in num_cols_for_ml:
            num_cols_for_ml.remove(candidate_target)
        if candidate_target in cat_cols_for_ml:
            cat_cols_for_ml.remove(candidate_target)

    if num_cols_for_ml or cat_cols_for_ml:
        pca_and_clustering(df, num_cols_for_ml, cat_cols_for_ml, color_col=color_col if color_col else '', outdir=outdir)

    # Dimensionality reduction and feature importance reports
    if num_cols_for_ml or cat_cols_for_ml:
        dimensionality_reduction_and_feature_importance(
            df,
            num_cols_for_ml,
            cat_cols_for_ml,
            target_col=candidate_target if candidate_target else '',
            outdir=outdir
        )

    print("All outputs saved to:", outdir)


if __name__ == "__main__":
    main()
