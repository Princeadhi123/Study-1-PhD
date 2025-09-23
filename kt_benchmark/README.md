# KT Benchmark: 9 Model Families on EQTd_DAi_25_itemwise.csv

This folder contains a lightweight, end-to-end benchmarking suite that fits and evaluates one representative (basic) model from each of the 9 families you outlined:

1. Psychometric Models — Rasch (1PL) via simple Joint MLE (alternating Newton updates)
2. Bayesian Models — Standard BKT (no forgetting) with EM per skill (group)
3. Machine Learning — Logistic Regression on tabular features
4. Deep Learning — Minimal DKT (LSTM-based) for sequence modeling (PyTorch, optional)
5. Graph Models — GKT-lite (KC graph + neighbor smoothing of student skill)
6. Temporal & Sequential — TIRT-lite (logistic with time-progress feature)
7. Multi-task — FKT-lite (predict correctness + response time with shared MLP, PyTorch optional)
8. Contrastive & Self-supervised — CLKT-lite (simple contrastive pretrain of student embeddings, PyTorch optional)
9. Domain Adaptive — AdaptKT-lite (CORAL feature alignment from source KCs to target KCs)

Why these choices:

- Rasch (1PL) is the simplest useful IRT model and robust on binary items.
- Standard BKT is the canonical Bayesian KT baseline and interpretable via (L0, learn, guess, slip).
- Logistic Regression is a strong ML baseline, fast and reasonably competitive.
- DKT (LSTM) is the classic deep sequential KT model; we include a small version to keep it runnable.
- GKT-lite reflects graph message passing by smoothing group mastery over a KC co-occurrence graph.
- TIRT-lite adds a simple time-progress covariate to emulate temporal ability drift (temporal IRT idea).
- FKT-lite implements hard parameter sharing for a primary (correctness) and auxiliary (RT) task.
- CLKT-lite provides a minimal contrastive objective over student sequence views before supervised finetuning.
- AdaptKT-lite demonstrates domain adaptation via CORAL alignment between group domains.

Data expectations

- Input file: `c:/Users/pdaadh/Desktop/Study 2/EQTd_DAi_25_itemwise.csv` (from your pipeline).
- Required columns: `IDCode`, `item`, `item_index`, `group`, `response`, `response_time_sec`, `orig_order`.
  - If some are missing, the runner will degrade gracefully and skip models that need them.
- Sex will be normalized to M/F/U according to your preference (non-destructive).

Outputs

- All metrics and artifacts are written to `kt_benchmark/output/`.
- Main summary: `metrics_summary.csv` and `metrics_summary.md` comparison tables.
- Per-model predictions and any learned parameters are saved in dedicated files.

Install

- Minimal set (non-deep models):
  - numpy, pandas, scikit-learn, scipy
- Optional (deep models: DKT/MTL/Contrastive):
  - torch

Example install (PowerShell):

```powershell
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn scipy matplotlib seaborn
# Optional deep models
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Run

```powershell
# Run from the project root directory containing the kt_benchmark/ folder
python -m kt_benchmark.run_benchmark
```

This will:

- Load and preprocess the itemwise CSV
- Create train/test split by students
- Train each model (skipping deep ones if PyTorch isn't present)
- Save per-model metrics and an overall comparison table

Notes

- The implementations favor clarity and minimal dependencies over state-of-the-art performance.
- DKT/MTL/Contrastive are intentionally small to fit typical CPU constraints. You can tune epochs/hidden sizes.
- Domain adaptation (AdaptKT-lite) uses a simple alphabetical split of `group` labels into source/target for demonstration. You can customize this in `config.py`.
