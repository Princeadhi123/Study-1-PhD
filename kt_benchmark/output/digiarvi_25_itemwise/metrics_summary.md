# KT Benchmark Summary

Run time: 2025-10-08T02:39:08.797086

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

```text
category, model, n, roc_auc, accuracy, avg_precision, f1, precision, recall, brier, log_loss, runtime_sec
Psychometric (IRT), Rasch1PL, 6104, 0.33102893033786507, 0.3733617300131062, 0.6146739665241557, 0.49290733130054354, 0.5685015290519878, 0.4350573367657384, 0.6266382699868939, 8.657328008383669, 5.040183799999795
Bayesian, BKT, 4469, 0.694663768844707, 0.7587827254419334, 0.7935454426969721, 0.8318777292576419, 0.8201107011070111, 0.8439873417721518, 0.19169469668239952, 0.5726504001628928, 101.26117660000091
Machine Learning, LogisticRegression, 6104, 0.7981449189486063, 0.7619593709043251, 0.8847196116249127, 0.824580466014729, 0.8516209476309227, 0.7992043061081208, 0.17869723318972588, 0.5410979223292347, 0.2935001000005286
Deep Learning, DKT, 6104, 0.7412188838173676, 0.7442660550458715, 0.8503267815668176, 0.8278751791818282, 0.7827356130108424, 0.8785396676807863, 0.17612186420306192, 0.53132646801282, 13.544338600000629
Graph, GKT, 6104, 0.7118181261609514, 0.7313237221494102, 0.8252431147677118, 0.8331977217249796, 0.7368231696348264, 0.9585771120992277, 0.1901286256400014, 0.5669265104213157, 0.268931499998871
Temporal/Sequential, TIRT, 6104, 0.7991189262899925, 0.7621231979030144, 0.8855329986268918, 0.8248069498069498, 0.851307596513076, 0.7999063889538965, 0.1788231321858179, 0.5414245272651791, 0.20647599999938393
Multi-task, FKT, 6104, 0.7855145853141856, 0.7708060288335518, 0.8717721550983175, 0.8463481603514552, 0.7973923841059603, 0.9017084015913878, 0.1626570240307659, 0.4987999140710676, 2.9132146000010835
Contrastive/Self-supervised, CLKT, 6104, 0.7970974824073478, 0.7554062909567497, 0.8839490749982921, 0.8186125622646094, 0.8511874684183931, 0.7884390358062251, 0.17912038630911226, 0.5417634546889808, 1.1830282000009902
Domain Adaptive, AdaptKT, 3052, 0.4708420540366489, 0.6372870249017037, 0.7723111824309268, 0.766307789740342, 0.7622847543049139, 0.7703735144312394, 0.2267232974754026, 0.6457627893216308, 0.13651450000179466
```
## Best by metric

- **roc_auc**: TIRT (Temporal/Sequential) = 0.7991 (n=6104)
- **accuracy**: FKT (Multi-task) = 0.7708 (n=6104)
- **avg_precision**: TIRT (Temporal/Sequential) = 0.8855 (n=6104)
- **f1**: FKT (Multi-task) = 0.8463 (n=6104)

Notes: Brier and log_loss are lower-is-better. Metrics skip models that failed.
