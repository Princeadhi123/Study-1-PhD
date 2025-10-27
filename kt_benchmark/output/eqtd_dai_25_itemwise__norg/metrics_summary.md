# KT Benchmark Summary

Run time: 2025-10-27T15:48:36.357764

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
| Psychometric (IRT)          | Rasch1PL           | 5646 |  0.322391 |   0.358307 |        0.626919 | 0.471172 |    0.574377 | 0.399406 | 0.641693 |   8.86532  |      4.26367  |
| Bayesian                    | BKT                | 4134 |  0.679687 |   0.757378 |        0.796645 | 0.834461 |    0.822917 | 0.846334 | 0.191825 |   0.572899 |     96.8681   |
| Machine Learning            | LogisticRegression | 5646 |  0.793574 |   0.761424 |        0.89056  | 0.828254 |    0.854287 | 0.803761 | 0.180537 |   0.545034 |      0.22967  |
| Deep Learning               | DKT                | 5646 |  0.744854 |   0.757705 |        0.86246  | 0.842759 |    0.786864 | 0.907201 | 0.170144 |   0.516953 |     15.6842   |
| Graph                       | GKT                | 5646 |  0.716786 |   0.725469 |        0.840007 | 0.83834  |    0.724536 | 0.994556 | 0.184478 |   0.554244 |      0.253533 |
| Temporal/Sequential         | TIRT               | 5646 |  0.795085 |   0.757882 |        0.891221 | 0.824721 |    0.855774 | 0.795843 | 0.18049  |   0.544994 |      0.15331  |
| Multi-task                  | FKT                | 5646 |  0.777918 |   0.769571 |        0.877207 | 0.851127 |    0.791613 | 0.920317 | 0.160143 |   0.493003 |      2.81041  |
| Contrastive/Self-supervised | CLKT               | 5646 |  0.793461 |   0.759122 |        0.889505 | 0.826309 |    0.853787 | 0.800544 | 0.180279 |   0.544545 |      1.31933  |
| Domain Adaptive             | AdaptKT            | 2904 |  0.524612 |   0.547176 |        0.794219 | 0.662041 |    0.784887 | 0.572444 | 0.247762 |   0.688668 |      0.162641 |
## Best by metric

- **roc_auc**: TIRT (Temporal/Sequential) = 0.7951 (n=5646)
- **accuracy**: FKT (Multi-task) = 0.7696 (n=5646)
- **avg_precision**: TIRT (Temporal/Sequential) = 0.8912 (n=5646)
- **f1**: FKT (Multi-task) = 0.8511 (n=5646)

Notes: Brier and log_loss are lower-is-better. Metrics skip models that failed.
