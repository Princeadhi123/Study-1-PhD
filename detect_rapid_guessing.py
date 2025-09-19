import sys
import math
import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


METHOD = Literal["quantile", "sdlog", "ln_quantile", "mixture"]


def ensure_outdir(base: Path) -> Path:
    out = base / "eda_output" / "rapid_guessing"
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_csv(path: Path) -> pd.DataFrame:
    print(f"Reading: {path}")
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df


def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _gmm_intersection(mu1: float, s1: float, w1: float, mu2: float, s2: float, w2: float) -> float | None:
    """
    Solve for x where w1*N(x|mu1,s1) == w2*N(x|mu2,s2) for normal PDFs.
    Returns x (in the same space as mu/sigma), or None if no valid solution.
    """
    # Handle degenerate cases
    if s1 <= 0 or s2 <= 0:
        return None
    A = 1.0 / (2 * s2 * s2) - 1.0 / (2 * s1 * s1)
    B = mu2 / (s2 * s2) - mu1 / (s1 * s1)
    C = (mu1 * mu1) / (2 * s1 * s1) - (mu2 * mu2) / (2 * s2 * s2) + math.log((w2 * s1) / (w1 * s2))

    if abs(A) < 1e-12:
        # Linear case
        if abs(B) < 1e-12:
            return None
        x = -C / B
        return x
    disc = B * B - 4 * A * C
    if disc < 0:
        return None
    sqrt_disc = math.sqrt(max(0.0, disc))
    x1 = (-B + sqrt_disc) / (2 * A)
    x2 = (-B - sqrt_disc) / (2 * A)
    # Prefer a solution between the means
    lo, hi = (mu1, mu2) if mu1 < mu2 else (mu2, mu1)
    for cand in (x1, x2):
        if lo <= cand <= hi:
            return cand
    # Otherwise pick the closer one to the interval
    return x1 if abs(x1 - (lo + hi) / 2) < abs(x2 - (lo + hi) / 2) else x2


def compute_thresholds_by_unit(
    df: pd.DataFrame,
    unit_cols: list[str],
    rt_col: str,
    method: METHOD = "ln_quantile",
    q: float = 0.10,
    sd_k: float = 2.0,
    min_floor: float = 0.5,
    max_cap: float | None = None,
) -> pd.DataFrame:
    """
    Compute response time thresholds per analysis unit (group, item, or item-within-group).

    Methods:
    - quantile: empirical quantile(q) of RTs
    - sdlog: exp(mu_log - sd_k * sd_log) using log RTs
    - ln_quantile: exp(mu_log + z_q * sd_log) where z_q = Phi^{-1}(q)
    - mixture: 2-component GMM on log RTs; threshold is intersection of components in log-space

    Returns a DataFrame with unit_cols + [rg_threshold_sec, rg_method, rg_param, n_in_unit]
    """
    use_cols = unit_cols + [rt_col]
    d = df[use_cols].copy()
    d[rt_col] = coerce_numeric(d[rt_col])
    d = d.dropna(subset=[rt_col])

    rows: list[dict] = []
    z_q = float(pd.Series([q]).pipe(lambda s: s.iloc[0]))  # ensure float
    z_norm = None
    if method == "ln_quantile":
        from scipy.stats import norm as _norm  # optional; fallback below if not available
        try:
            z_norm = float(_norm.ppf(q))
        except Exception:
            # Fallback approximate for common tails
            z_map = {0.05: -1.64485, 0.10: -1.28155}
            z_norm = z_map.get(round(q, 2), -1.28155)

    for keys, sub in d.groupby(unit_cols, dropna=False):
        rts = sub[rt_col].astype(float).values
        thr: float | None = None
        if len(rts) == 0:
            thr = np.nan
        elif method == "quantile":
            thr = float(np.quantile(rts, q))
        elif method == "sdlog":
            safe = rts[rts > 0]
            if len(safe) >= 3:
                logs = np.log(safe)
                thr = float(np.exp(logs.mean() - sd_k * logs.std(ddof=1)))
            else:
                thr = float(np.quantile(rts, q))
        elif method == "ln_quantile":
            safe = rts[rts > 0]
            if len(safe) >= 3:
                logs = np.log(safe)
                mu, sd = logs.mean(), logs.std(ddof=1)
                thr = float(np.exp(mu + (z_norm if z_norm is not None else -1.28155) * sd))
            else:
                thr = float(np.quantile(rts, q))
        elif method == "mixture":
            safe = rts[rts > 0]
            if len(safe) >= 10:  # need enough data for a stable 2-comp fit
                logs = np.log(safe).reshape(-1, 1)
                gmm = GaussianMixture(n_components=2, random_state=42, n_init=10)
                gmm.fit(logs)
                mu = gmm.means_.flatten()
                sd = np.sqrt(gmm.covariances_.flatten())
                w = gmm.weights_.flatten()
                # Identify fast (lower mean) vs slow (higher mean)
                order = np.argsort(mu)
                mu1, mu2 = float(mu[order[0]]), float(mu[order[1]])
                s1, s2 = float(sd[order[0]]), float(sd[order[1]])
                w1, w2 = float(w[order[0]]), float(w[order[1]])
                x_star = _gmm_intersection(mu1, s1, w1, mu2, s2, w2)
                if x_star is None or not np.isfinite(x_star):
                    # Numeric fallback: find where posteriors are equal on a fine grid
                    grid = np.linspace(mu1 - 4 * s1, mu2 + 4 * s2, 1000)
                    # Compute posterior responsibility for component 0 (fast)
                    from scipy.stats import norm
                    p1 = w1 * norm.pdf(grid, mu1, s1)
                    p2 = w2 * norm.pdf(grid, mu2, s2)
                    idx = np.argmin(np.abs(p1 - p2))
                    x_star = grid[idx]
                thr = float(np.exp(x_star))
            else:
                # Fallback to empirical lower quantile when data is sparse
                thr = float(np.quantile(rts, q))
        else:
            raise ValueError(f"Unknown method: {method}")

        if thr is not None and not np.isnan(thr):
            if min_floor is not None:
                thr = max(thr, min_floor)
            if max_cap is not None:
                thr = min(thr, max_cap)

        row = {"rg_threshold_sec": thr, "rg_method": method, "rg_param": q if method in ("quantile", "ln_quantile") else (sd_k if method == "sdlog" else "gmm2"), "n_in_unit": len(rts)}
        if isinstance(keys, tuple):
            for col, val in zip(unit_cols, keys):
                row[col] = val
        else:
            row[unit_cols[0]] = keys
        rows.append(row)

    cols = unit_cols + ["rg_threshold_sec", "rg_method", "rg_param", "n_in_unit"]
    return pd.DataFrame(rows)[cols]


def add_rapid_guessing_flags(
    df: pd.DataFrame,
    unit_cols: list[str],
    id_col: str,
    rt_col: str,
    resp_col: str | None,
    thr_df: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()
    out[rt_col] = coerce_numeric(out[rt_col])

    merge_cols = unit_cols.copy()
    out = out.merge(thr_df[merge_cols + ["rg_threshold_sec"]], on=merge_cols, how="left")

    out["is_rapid_guess"] = (out[rt_col].notna()) & (out["rg_threshold_sec"].notna()) & (out[rt_col] <= out["rg_threshold_sec"])

    if resp_col and resp_col in out.columns:
        # Try to interpret response as numeric correctness (1/0)
        resp_num = pd.to_numeric(out[resp_col], errors="coerce")
        out["is_rapid_incorrect"] = out["is_rapid_guess"] & (resp_num == 0)
        out["is_rapid_correct"] = out["is_rapid_guess"] & (resp_num == 1)
    else:
        out["is_rapid_incorrect"] = False
        out["is_rapid_correct"] = False

    return out


def summarize_by_participant(df_rg: pd.DataFrame, id_col: str) -> pd.DataFrame:
    g = df_rg.groupby(id_col, dropna=False)
    summ = g.agg(
        n_items=(id_col, "size"),
        n_rg=("is_rapid_guess", "sum"),
        n_rg_incorrect=("is_rapid_incorrect", "sum"),
        n_rg_correct=("is_rapid_correct", "sum"),
        median_rt=("response_time_sec", "median"),
        mean_rt=("response_time_sec", "mean"),
    ).reset_index()
    summ["rg_rate"] = (summ["n_rg"] / summ["n_items"]).replace([np.inf, -np.inf], np.nan)
    # Flag participants with RG rate >= 10% (tunable) and at least 20 items (tunable)
    summ["flag_rg_participant"] = (summ["rg_rate"] >= 0.10) & (summ["n_items"] >= 20)
    return summ


def plot_rt_hist_with_thresholds(df: pd.DataFrame, group_col: str, rt_col: str, thr_df: pd.DataFrame, outdir: Path):
    # One plot per group
    for g, sub in df.groupby(group_col, dropna=False):
        rts = pd.to_numeric(sub[rt_col], errors="coerce").dropna()
        if rts.empty:
            continue
        thr = thr_df.loc[thr_df[group_col] == g, "rg_threshold_sec"].values
        thr_val = thr[0] if len(thr) else None
        plt.figure(figsize=(7, 4))
        sns.histplot(rts, bins=30, kde=True, color="#5B9BD5")
        if thr_val is not None and not np.isnan(thr_val):
            plt.axvline(thr_val, color="red", linestyle="--", label=f"threshold={thr_val:.2f}s")
            plt.legend()
        plt.title(f"RT distribution - {group_col}={g}")
        plt.xlabel(rt_col)
        plt.ylabel("Count")
        plt.tight_layout()
        safe_g = str(g).replace("/", "_")
        plt.savefig(outdir / f"rt_hist_{safe_g}.png", dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Detect rapid guessing per item using response time thresholds.")
    base_dir = Path(__file__).parent
    # Use the full itemwise export by default (contains item/item_index)
    default_csv = base_dir / "EQTd_DAi_25_itemwise.csv"

    parser.add_argument("--csv", type=Path, default=default_csv, help="Path to itemwise CSV")
    parser.add_argument("--group-col", default="group", help="Grouping column (for unit-by group)")
    # Default to item-level thresholds per recent analysis
    parser.add_argument("--unit-by", choices=["group", "item", "item_in_group"], default="item", help="Unit to compute thresholds over")
    parser.add_argument("--item-col", default="item", help="Item column name (if available)")
    parser.add_argument("--item-index-col", default="item_index", help="Item index column name (if available)")
    parser.add_argument("--id-col", default="IDCode", help="Participant ID column")
    parser.add_argument("--rt-col", default="response_time_sec", help="Response time column (seconds)")
    parser.add_argument("--resp-col", default="response", help="Response/correctness column (0/1 if available)")
    # Default to ln-quantile(q=0.10) per item (robust, adaptive)
    parser.add_argument("--method", choices=["quantile", "sdlog", "ln_quantile", "mixture"], default="ln_quantile", help="Thresholding method")
    parser.add_argument("--q", type=float, default=0.10, help="Quantile for method=quantile")
    parser.add_argument("--sd-k", type=float, default=2.0, help="k in exp(mu - k*sd) for method=sdlog")
    parser.add_argument("--min-floor", type=float, default=0.5, help="Minimum threshold in seconds")
    parser.add_argument("--max-cap", type=float, default=None, help="Optional cap on threshold")
    parser.add_argument("--no-plots", action="store_true", help="Disable plots")

    args = parser.parse_args()

    csv_path: Path = args.csv
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    outdir = ensure_outdir(csv_path.parent)

    df = load_csv(csv_path)

    # Decide unit columns
    unit_cols: list[str]
    if args.unit_by == "group":
        unit_cols = [args.group_col]
    elif args.unit_by == "item":
        # Prefer item_col if present, else item_index_col
        if args.item_col in df.columns:
            unit_cols = [args.item_col]
        elif args.item_index_col in df.columns:
            unit_cols = [args.item_index_col]
        else:
            print("ERROR: unit-by=item requires an item or item_index column.")
            sys.exit(2)
    else:  # item_in_group
        cols = []
        if args.group_col in df.columns:
            cols.append(args.group_col)
        else:
            print(f"ERROR: Missing required column: {args.group_col}")
            sys.exit(2)
        if args.item_col in df.columns:
            cols.append(args.item_col)
        elif args.item_index_col in df.columns:
            cols.append(args.item_index_col)
        else:
            print("ERROR: unit-by=item_in_group requires an item or item_index column.")
            sys.exit(2)
        unit_cols = cols

    # Validate other required columns
    for col in [args.id_col, args.rt_col] + unit_cols:
        if col not in df.columns:
            print(f"ERROR: Required column missing: {col}")
            sys.exit(2)

    thr_df = compute_thresholds_by_unit(
        df,
        unit_cols=unit_cols,
        rt_col=args.rt_col,
        method=args.method,
        q=args.q,
        sd_k=args.sd_k,
        min_floor=args.min_floor,
        max_cap=args.max_cap,
    )

    # Thresholds filename by unit
    unit_tag = "_".join(unit_cols)
    thr_out = outdir / f"rg_thresholds_by_{unit_tag}.csv"
    thr_df.to_csv(thr_out, index=False)
    print(f"Saved thresholds: {thr_out}")

    df_rg = add_rapid_guessing_flags(
        df=df,
        unit_cols=unit_cols,
        id_col=args.id_col,
        rt_col=args.rt_col,
        resp_col=args.resp_col if args.resp_col in df.columns else None,
        thr_df=thr_df,
    )

    # Save augmented rows
    out_items = csv_path.with_name(csv_path.stem + "_with_rg.csv")
    df_rg.to_csv(out_items, index=False)
    print(f"Saved augmented itemwise file: {out_items}")

    # Participant-level summary
    part_summary = summarize_by_participant(df_rg, id_col=args.id_col)
    out_part = outdir / "rg_participant_summary.csv"
    part_summary.to_csv(out_part, index=False)
    print(f"Saved participant summary: {out_part}")

    # Flag list
    out_flags = outdir / "rg_flagged_participants.csv"
    part_summary.loc[part_summary["flag_rg_participant"]].to_csv(out_flags, index=False)
    print(f"Saved flagged participants: {out_flags}")

    if not args.no_plots and args.unit_by == "group":
        plot_rt_hist_with_thresholds(df, args.group_col, args.rt_col, thr_df, outdir)
        print(f"Saved RT histograms with thresholds to: {outdir}")

    print("Done.")


if __name__ == "__main__":
    main()
