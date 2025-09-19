import sys
from pathlib import Path
from typing import List, Tuple, Dict, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

# Local imports from detect_rapid_guessing.py
sys.path.append(str(Path(__file__).parent))
from detect_rapid_guessing import (
    compute_thresholds_by_unit,
    add_rapid_guessing_flags,
    summarize_by_participant,
)


DEFAULT_CSV = Path(r"c:\Users\pdaadh\Desktop\Study 2\EQTd_DAi_25_itemwise.csv")
ID_COL = "IDCode"
GROUP_COL = "group"
ITEM_COL = "item"
ITEM_INDEX_COL = "item_index"
RT_COL = "response_time_sec"
RESP_COL = "response"


def ensure_outdir(base: Path) -> Path:
    out = base / "eda_output" / "qsweep"
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_csv(path: Path) -> pd.DataFrame:
    print(f"Reading: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def has_item_columns(df: pd.DataFrame) -> Tuple[bool, str]:
    if ITEM_COL in df.columns:
        return True, ITEM_COL
    if ITEM_INDEX_COL in df.columns:
        return True, ITEM_INDEX_COL
    return False, ""


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def compute_metrics_for_q(df: pd.DataFrame, unit_cols: List[str], q: float, outdir: Path) -> Dict:
    thr_df = compute_thresholds_by_unit(
        df, unit_cols=unit_cols, rt_col=RT_COL, method="ln_quantile", q=q, sd_k=2.0, min_floor=0.5, max_cap=None
    )
    df_rg = add_rapid_guessing_flags(
        df=df, unit_cols=unit_cols, id_col=ID_COL, rt_col=RT_COL, resp_col=RESP_COL, thr_df=thr_df
    )

    # Response-level metrics
    y_true = pd.to_numeric(df_rg[RESP_COL], errors="coerce")  # correctness 0/1
    rg_flag = df_rg["is_rapid_guess"].astype(bool)
    # AUC for predicting correctness using non-rapid as score
    valid_mask = y_true.isin([0, 1]) & rg_flag.notna()
    auc = np.nan
    if valid_mask.sum() > 0:
        try:
            auc = roc_auc_score(y_true[valid_mask].values, (~rg_flag[valid_mask]).astype(int).values)
        except Exception:
            auc = np.nan

    # Correctness separation
    mean_corr_rg = y_true[rg_flag].mean(skipna=True)
    mean_corr_non = y_true[~rg_flag].mean(skipna=True)
    delta_correct = (mean_corr_non - mean_corr_rg) if pd.notna(mean_corr_rg) and pd.notna(mean_corr_non) else np.nan

    # Within-item separation (average difference)
    item_col = unit_cols[-1] if len(unit_cols) == 2 else (ITEM_COL if ITEM_COL in df.columns else ITEM_INDEX_COL)
    diffs = []
    for _, sub in df_rg.groupby(item_col, dropna=False):
        y = pd.to_numeric(sub[RESP_COL], errors="coerce")
        rg = sub["is_rapid_guess"].astype(bool)
        if rg.any() and (~rg).any():
            diffs.append(y[~rg].mean(skipna=True) - y[rg].mean(skipna=True))
    item_mean_delta = float(np.nanmean(diffs)) if diffs else np.nan

    # Prevalence
    resp_flag_rate = float(rg_flag.mean())

    # Participant-level summary
    part = summarize_by_participant(df_rg, id_col=ID_COL)
    part_flagged = set(part.loc[part["flag_rg_participant"], ID_COL].astype(str).tolist())

    return {
        "q": q,
        "auc_correct": auc,
        "mean_correct_rg": float(mean_corr_rg) if pd.notna(mean_corr_rg) else np.nan,
        "mean_correct_nonrg": float(mean_corr_non) if pd.notna(mean_corr_non) else np.nan,
        "delta_correct": float(delta_correct) if pd.notna(delta_correct) else np.nan,
        "item_mean_delta": float(item_mean_delta) if pd.notna(item_mean_delta) else np.nan,
        "resp_flag_rate": resp_flag_rate,
        "n_participants_flagged": int(len(part_flagged)),
        "flagged_participants": part_flagged,
    }


def recommend_q(rows: List[Dict]) -> Tuple[float, pd.DataFrame]:
    # Choose q with highest AUC, with soft constraints: prevalence 0.1-0.35 and stability (jaccard with neighbors)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "flagged_participants"} for r in rows])
    df = df.sort_values("q").reset_index(drop=True)

    # Normalize AUC to handle NaNs
    df["auc_norm"] = (df["auc_correct"] - df["auc_correct"].min()) / (df["auc_correct"].max() - df["auc_correct"].min() + 1e-9)

    # Stability: Jaccard to neighbors
    jaccards = []
    for i in range(len(rows)):
        if i == 0 or i == len(rows) - 1:
            jaccards.append(np.nan)
            continue
        a = rows[i - 1]["flagged_participants"]
        b = rows[i]["flagged_participants"]
        c = rows[i + 1]["flagged_participants"]
        j_prev = jaccard(a, b)
        j_next = jaccard(b, c)
        jaccards.append(np.nanmean([j_prev, j_next]))
    df["stability_jaccard"] = jaccards

    # Score combining AUC (weight 0.6), stability (0.3), and prevalence closeness to 0.2 (0.1)
    target_prev = 0.20
    df["prev_score"] = 1.0 - np.minimum(1.0, np.abs(df["resp_flag_rate"] - target_prev) / 0.2)
    stab_norm = (df["stability_jaccard"] - np.nanmin(df["stability_jaccard"])) / (
        (np.nanmax(df["stability_jaccard"]) - np.nanmin(df["stability_jaccard"])) + 1e-9
    )
    df["stab_norm"] = stab_norm.fillna(stab_norm.mean())

    df["selection_score"] = 0.6 * df["auc_norm"].fillna(0) + 0.3 * df["stab_norm"] + 0.1 * df["prev_score"]

    # Soft filter: keep qs with prevalence 0.05-0.40
    cand = df[(df["resp_flag_rate"] >= 0.05) & (df["resp_flag_rate"] <= 0.40)]
    if cand.empty:
        cand = df

    best_row = cand.sort_values(["selection_score", "auc_correct"], ascending=[False, False]).iloc[0]
    # Return both the recommended q and the scoring dataframe (with per-q scores only)
    # Note: exclude metrics already present in the summary (e.g., auc_correct, resp_flag_rate) to avoid duplicate columns
    return float(best_row["q"]), df[[
        "q", "auc_norm", "stability_jaccard", "stab_norm", "prev_score", "selection_score"
    ]]


def main():
    base_dir = Path(__file__).parent
    outdir = ensure_outdir(base_dir)

    csv_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    df = load_csv(csv_path)

    # Ensure required columns
    for col in [ID_COL, GROUP_COL, RT_COL, RESP_COL]:
        if col not in df.columns:
            print(f"ERROR: Required column missing: {col}")
            sys.exit(2)
    has_item, item_col = has_item_columns(df)
    if not has_item:
        print("ERROR: Need 'item' or 'item_index' column for item-level thresholds.")
        sys.exit(2)

    unit_cols = [GROUP_COL, item_col]

    qs = [round(q, 2) for q in np.arange(0.05, 0.205, 0.01)]
    results: List[Dict] = []

    for q in qs:
        print(f"Computing metrics for q={q:.2f}")
        res = compute_metrics_for_q(df, unit_cols=unit_cols, q=q, outdir=outdir)
        results.append(res)

    # Build summary table
    rows = []
    for r in results:
        rows.append({k: v for k, v in r.items() if k != "flagged_participants"})
    summ = pd.DataFrame(rows).sort_values("q")

    # Stability to previous
    prev_sets: List[Set[str]] = [r["flagged_participants"] for r in results]
    j_prev = [np.nan]
    for i in range(1, len(prev_sets)):
        j_prev.append(jaccard(prev_sets[i - 1], prev_sets[i]))
    summ["jaccard_to_prev_q"] = j_prev

    # Recommend q and get per-q scores
    q_rec, score_df = recommend_q(results)
    summ = summ.merge(score_df, on="q", how="left")
    summ["recommended"] = summ["q"].apply(lambda x: x == q_rec)

    out_csv = outdir / "rg_quantile_sweep_summary.csv"
    summ.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Plots
    sns.set(context="notebook", style="whitegrid")

    plt.figure(figsize=(8,5))
    plt.plot(summ["q"], summ["auc_correct"], marker="o", label="AUC (correctness)")
    plt.axvline(q_rec, color="red", linestyle=":", label=f"recommended q={q_rec:.2f}")
    plt.xlabel("q (ln-quantile)")
    plt.ylabel("ROC-AUC predicting correctness")
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "qsweep_auc_vs_q.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(summ["q"], summ["resp_flag_rate"], marker="o", label="response flagged rate")
    plt.plot(summ["q"], summ["n_participants_flagged"] / df[ID_COL].nunique(), marker="s", label="participants flagged rate")
    plt.axvline(q_rec, color="red", linestyle=":")
    plt.xlabel("q (ln-quantile)")
    plt.ylabel("Prevalence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "qsweep_prevalence_vs_q.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(summ["q"], summ["delta_correct"], marker="o", label="nonRG - RG correctness")
    plt.plot(summ["q"], summ["item_mean_delta"], marker="s", label="avg within-item delta")
    plt.axvline(q_rec, color="red", linestyle=":")
    plt.xlabel("q (ln-quantile)")
    plt.ylabel("Correctness gap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "qsweep_correctness_gap_vs_q.png", dpi=150)
    plt.close()

    # Stability vs q plot (average Jaccard to neighbors and Jaccard to previous q)
    plt.figure(figsize=(8,5))
    if "stability_jaccard" in summ.columns:
        plt.plot(summ["q"], summ["stability_jaccard"], marker="o", label="avg Jaccard to neighbors")
    if "jaccard_to_prev_q" in summ.columns:
        plt.plot(summ["q"], summ["jaccard_to_prev_q"], marker="s", label="Jaccard to previous q")
    plt.axvline(q_rec, color="red", linestyle=":")
    plt.xlabel("q (ln-quantile)")
    plt.ylabel("Stability (Jaccard)")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "qsweep_stability_vs_q.png", dpi=150)
    plt.close()

    # One-pager summary (markdown)
    rec_row = summ.loc[summ["q"] == q_rec].iloc[0]
    top5 = summ.sort_values("selection_score", ascending=False).head(5)[["q", "selection_score"]]

    md_lines = []
    md_lines.append("# Quantile Sweep Summary (Item-level ln-quantile)\n")
    md_lines.append(f"- **Recommended q**: {q_rec:.2f}\n")
    md_lines.append("- **Selection score formula**: 0.6×auc_norm + 0.3×stab_norm + 0.1×prev_score\n")
    md_lines.append("- **Key metrics at recommended q**:\n")
    md_lines.append(f"  - **AUC (correctness)**: {rec_row['auc_correct']:.3f}\n")
    md_lines.append(f"  - **Correctness gap (nonRG − RG)**: {rec_row['delta_correct']:.3f}\n")
    md_lines.append(f"  - **Avg within-item gap**: {rec_row['item_mean_delta']:.3f}\n")
    md_lines.append(f"  - **Response-level prevalence**: {rec_row['resp_flag_rate']:.3%}\n")
    md_lines.append(f"  - **Participants flagged**: {int(rec_row['n_participants_flagged'])}\n")
    if 'stability_jaccard' in rec_row.index and pd.notna(rec_row['stability_jaccard']):
        md_lines.append(f"  - **Stability (avg Jaccard to neighbors)**: {rec_row['stability_jaccard']:.3f}\n")
    md_lines.append("\n- **Top-5 q by selection_score**:\n")
    for _, r in top5.iterrows():
        md_lines.append(f"  - q={r['q']:.2f}: score={r['selection_score']:.4f}\n")
    md_lines.append("\n- **Figures**: `qsweep_auc_vs_q.png`, `qsweep_prevalence_vs_q.png`, `qsweep_correctness_gap_vs_q.png`, `qsweep_stability_vs_q.png`\n")

    (outdir / "qsweep_summary.md").write_text("".join(md_lines), encoding="utf-8")

    # Combined scores plot: selection score, auc_norm, stability (norm), prevalence score
    plt.figure(figsize=(10,6))
    plt.plot(summ["q"], summ["selection_score"], color="#f5a623", marker="o", linewidth=2, label="Selection Score")
    if "auc_norm" in summ.columns:
        plt.plot(summ["q"], summ["auc_norm"], color="#82aaff", marker="s", linestyle="--", linewidth=2, label="AUC norm")
    if "stab_norm" in summ.columns:
        plt.plot(summ["q"], summ["stab_norm"], color="#00bfa5", marker="^", linestyle="-.", linewidth=2, label="Stability norm")
    if "prev_score" in summ.columns:
        plt.plot(summ["q"], summ["prev_score"], color="#ffd54f", marker="d", linestyle=":", linewidth=2, label="Prevalence score")
    plt.axvline(q_rec, color="red", linestyle=":", linewidth=2, label=f"Recommended q = {q_rec:.2f}")
    plt.ylim(0, 1.05)
    plt.xlabel("Quantile threshold (q)")
    plt.ylabel("Score")
    plt.title("Quantile Sweep: Selection, AUC norm, Stability norm, Prevalence score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "qsweep_scores_vs_q.png", dpi=150)
    plt.close()

    print(f"Recommended q: {q_rec:.2f}")


if __name__ == "__main__":
    main()
