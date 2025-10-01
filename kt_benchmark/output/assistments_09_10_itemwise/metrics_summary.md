# KT Benchmark Summary

Run time: 2025-10-01T20:25:00.121116

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
Psychometric (IRT), Rasch1PL, 10497, 0.39909957958039827, 0.42259693245689245, 0.5380178904269473, 0.4954632481478398, 0.5222885222885223, 0.4712589073634204, 0.5634705153853482, 7.318856881384439, 69.80718280002475
Bayesian, BKT, 10497, 0.6874329450993266, 0.6805754024959513, 0.724748900994272, 0.7694740460639395, 0.679951397326853, 0.8861441013460016, 0.21576759020488112, 0.6233150312969084, 182.4645945999655
Machine Learning, LogisticRegression, 10497, 0.7205451255295003, 0.6826712394017338, 0.777939626476151, 0.7510277300246655, 0.711211778029445, 0.7955661124307205, 0.203781822711706, 0.5903195324359798, 1.032501800043974
Deep Learning, DKT, 10497, 0.6260994693920671, 0.6599980946937221, 0.6804408794335925, 0.7722254132363265, 0.6467821252939919, 0.9580364212193191, 0.22254706836356214, 0.6361019431609369, 125.80081939999945
Graph, GKT, 10497, 0.5666027687942101, 0.6245593979232161, 0.6359032470749522, 0.7615128593040847, 0.616258570029383, 0.9963578780680918, 0.2356328271876234, 0.6641638093970206, 0.1474877999862656
Temporal/Sequential, TIRT, 10497, 0.7248506493727784, 0.6868629132132991, 0.7821126357411115, 0.7543898976313234, 0.7142048670062252, 0.7993665874901029, 0.2027799627250273, 0.5881336120658299, 1.0220025000162423
Multi-task, FKT, 10497, 0.5391701909893207, 0.5427264932837954, 0.637585777117172, 0.6623047699451245, 0.5958982149639195, 0.7453681710213776, 0.24349647116203663, 0.6800845643604917, 52.62122420000378
Contrastive/Self-supervised, CLKT, 10497, 0.652408296613356, 0.6403734400304849, 0.7065074416512557, 0.7172072814443029, 0.6805516064827979, 0.7580364212193191, 0.2237578933965789, 0.6372486698568557, 9.064116800029296
Domain Adaptive, AdaptKT, 1525, 0.629888640088818, 0.6439344262295082, 0.7183424860379096, 0.7834064619066613, 0.6439344262295082, 1.0, 0.22612004145563683, 0.6445716487522521, 329.2788621999789
```
## Best by metric

- **roc_auc**: TIRT (Temporal/Sequential) = 0.7249 (n=10497)
- **accuracy**: TIRT (Temporal/Sequential) = 0.6869 (n=10497)
- **avg_precision**: TIRT (Temporal/Sequential) = 0.7821 (n=10497)
- **f1**: AdaptKT (Domain Adaptive) = 0.7834 (n=1525)

Notes: Brier and log_loss are lower-is-better. Metrics skip models that failed.
