# Quantile Sweep Summary (Item-level ln-quantile)
- **Recommended q**: 0.10
- **Selection score formula**: 0.6×auc_norm + 0.3×stab_norm + 0.1×prev_score
- **Key metrics at recommended q**:
  - **AUC (correctness)**: 0.527
  - **Correctness gap (nonRG − RG)**: 0.168
  - **Avg within-item gap**: 0.163
  - **Response-level prevalence**: 7.420%
  - **Participants flagged**: 136
  - **Stability (avg Jaccard to neighbors)**: 0.912

- **Top-5 q by selection_score**:
  - q=0.10: score=0.7911
  - q=0.05: score=0.7756
  - q=0.13: score=0.7168
  - q=0.06: score=0.7167
  - q=0.12: score=0.7016

- **Figures**: `qsweep_auc_vs_q.png`, `qsweep_prevalence_vs_q.png`, `qsweep_correctness_gap_vs_q.png`, `qsweep_stability_vs_q.png`
