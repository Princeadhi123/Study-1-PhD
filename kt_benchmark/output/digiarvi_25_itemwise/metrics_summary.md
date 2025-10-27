# KT Benchmark Summary

Run time: 2025-10-27T15:46:21.619158

## Models

- **Psychometric (IRT) — Rasch1PL**: Rasch (1PL) is the simplest IRT model: a single ability per student and difficulty per item, with logistic link. It provides an interpretable baseline in the psychometric family.
- **Bayesian — BKT**: Standard BKT is the canonical Bayesian model for KT with interpretable parameters (L0, learn, guess, slip). We fit per-skill (group) using a coarse grid over parameters for robustness and speed.
- **Machine Learning — LogisticRegression**: Logistic Regression is a strong, transparent ML baseline for binary correctness using tabular features. It is fast, robust to high-dimensional one-hot features, and provides calibrated probabilities.
- **Deep Learning — DKT**: DKT (LSTM-based) is the classic deep learning KT model that captures sequential dependencies. Here we use a minimal version predicting current response from the running hidden state over item embeddings.
- **Graph — GKT**: GKT-lite builds a knowledge-concept (KC) graph over groups (skills) via co-occurrence of student interactions, then smooths group difficulty over the graph to obtain predictions: p(g) blended with neighbors.
- **Temporal/Sequential — TIRT**: TIRT-lite augments item effects with a simple temporal progress covariate (z-scored within-student time index). This emulates temporal Item Response Theory by allowing ability to drift over time.
- **Multi-task — FKT**: FKT-lite: a small shared MLP predicts correctness (primary) and response time (aux) jointly. Multi-task learning can improve representation quality and calibration by leveraging correlated auxiliary signals.
- **Contrastive/Self-supervised — CLKT**: CLKT-lite: learn item embeddings via contrastive pretraining from train sequences (positive = co-occurring items), then fine-tune a logistic regression using learned embeddings + metadata to predict correctness.
- **Domain Adaptive — AdaptKT**: AdaptKT-lite performs unsupervised CORAL feature alignment from a source domain (groups in train) to a target domain (unseen or held-out groups in test), then trains a logistic classifier on aligned source and evaluates on aligned target.

## Metrics (higher is better unless noted)

| category                    | model              |    n |   roc_auc |   accuracy |   avg_precision |       f1 |   precision |   recall |    brier |   log_loss |   runtime_sec |
|:----------------------------|:-------------------|-----:|----------:|-----------:|----------------:|---------:|------------:|---------:|---------:|-----------:|--------------:|
| Psychometric (IRT)          | Rasch1PL           | 6104 |  0.331029 |   0.373362 |        0.614674 | 0.492907 |    0.568502 | 0.435057 | 0.626638 |   8.65733  |      4.54501  |
| Bayesian                    | BKT                | 4469 |  0.694664 |   0.758783 |        0.793545 | 0.831878 |    0.820111 | 0.843987 | 0.191695 |   0.57265  |    102.188    |
| Machine Learning            | LogisticRegression | 6104 |  0.798145 |   0.761959 |        0.88472  | 0.82458  |    0.851621 | 0.799204 | 0.178697 |   0.541098 |      0.305043 |
| Deep Learning               | DKT                | 6104 |  0.741397 |   0.74443  |        0.850553 | 0.830028 |    0.776555 | 0.891411 | 0.176101 |   0.53125  |     24.1699   |
| Graph                       | GKT                | 6104 |  0.711818 |   0.731324 |        0.825243 | 0.833198 |    0.736823 | 0.958577 | 0.190129 |   0.566927 |      0.266088 |
| Temporal/Sequential         | TIRT               | 6104 |  0.799119 |   0.762123 |        0.885533 | 0.824807 |    0.851308 | 0.799906 | 0.178823 |   0.541425 |      0.209214 |
| Multi-task                  | FKT                | 6104 |  0.792162 |   0.775393 |        0.879544 | 0.845972 |    0.813526 | 0.881114 | 0.160768 |   0.494005 |      2.66689  |
| Contrastive/Self-supervised | CLKT               | 6104 |  0.79798  |   0.761632 |        0.884654 | 0.824381 |    0.851196 | 0.799204 | 0.178735 |   0.541147 |      1.65393  |
| Domain Adaptive             | AdaptKT            | 3052 |  0.470842 |   0.637287 |        0.772311 | 0.766308 |    0.762285 | 0.770374 | 0.226723 |   0.645763 |      0.16319  |
## Best by metric

- **roc_auc**: TIRT (Temporal/Sequential) = 0.7991 (n=6104)
- **accuracy**: FKT (Multi-task) = 0.7754 (n=6104)
- **avg_precision**: TIRT (Temporal/Sequential) = 0.8855 (n=6104)
- **f1**: FKT (Multi-task) = 0.8460 (n=6104)

Notes: Brier and log_loss are lower-is-better. Metrics skip models that failed.
