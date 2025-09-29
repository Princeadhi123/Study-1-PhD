# KT Benchmark Summary

Run time: 2025-09-29T03:44:13.631162

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
Psychometric (IRT), Rasch1PL, 6213, 0.33229222881813275, 0.37035248672139065, 0.6164822616095943, 0.4872870249017038, 0.5685015290519878, 0.42637614678899083, 0.6296475132786093, 8.698902237842718, 4.307282399997348
Bayesian, BKT, 6213, 0.6815938200884261, 0.6333494286174151, 0.7852102805867278, 0.7007356805044667, 0.8201107011070111, 0.611697247706422, 0.21404210517843927, 0.6184785453404718, 102.7986447000003
Machine Learning, LogisticRegression, 6213, 0.7375524193348748, 0.6978915177852889, 0.8502565559908538, 0.7720150613385157, 0.8205525432481281, 0.7288990825688073, 0.20438914134400077, 0.5988471529494774, 0.13309920000028796
Deep Learning, DKT, 6213, 0.7396272842947464, 0.7452116529856752, 0.8501068915398554, 0.8291419320021587, 0.7830784913353721, 0.8809633027522936, 0.17557906569961723, 0.5297760197620419, 14.99196669999219
Graph, GKT, 6213, 0.7094846195358877, 0.7324963785610816, 0.8247858134271331, 0.8342640606302354, 0.7380028228652082, 0.9594036697247706, 0.18989403074150257, 0.5664835035995843, 0.26093290001153946
Temporal/Sequential, TIRT, 6213, 0.7408490199379136, 0.6967648478995654, 0.8523516641831027, 0.7701878506952915, 0.8225638353309015, 0.7240825688073395, 0.2041267598048022, 0.5986292686123131, 0.17372360000445042
Multi-task, FKT, 6213, 0.7245091644098092, 0.738129727989699, 0.8365748642250075, 0.8254104517652109, 0.7755595886267392, 0.8821100917431193, 0.18154296039307213, 0.5456648691663599, 1.855875000008382
Contrastive/Self-supervised, CLKT, 6213, 0.7364331706085346, 0.6982134234669242, 0.8475941993166216, 0.7723132969034608, 0.8206451612903226, 0.7293577981651376, 0.20448499113248708, 0.5996397615784008, 4.57979350000096
Domain Adaptive, AdaptKT, 3161, 0.47472209771402957, 0.6159443214172731, 0.7756464291683085, 0.7469779074614422, 0.7551622418879056, 0.7389690721649484, 0.22696918511603065, 0.6461877576523162, 0.1500705999933416
```
## Best by metric

- **roc_auc**: TIRT (Temporal/Sequential) = 0.7408 (n=6213)
- **accuracy**: DKT (Deep Learning) = 0.7452 (n=6213)
- **avg_precision**: TIRT (Temporal/Sequential) = 0.8524 (n=6213)
- **f1**: GKT (Graph) = 0.8343 (n=6213)

Notes: Brier and log_loss are lower-is-better. Metrics skip models that failed.
