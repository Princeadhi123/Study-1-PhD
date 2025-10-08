# KT Benchmark Summary

Run time: 2025-10-08T02:27:30.499631

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
Psychometric (IRT), Rasch1PL, 10423, 0.39858319943723675, 0.42233522018612685, 0.5386060722359015, 0.49568640589664126, 0.522791519434629, 0.47125338429686253, 0.563753238031277, 7.321310057851325, 80.64030310000089
Bayesian, BKT, 10255, 0.6885579845618868, 0.6853242320819113, 0.7261967994986391, 0.7761980719883487, 0.679951397326853, 0.9041848440782032, 0.21462626956417719, 0.6209877600146901, 147.06683260000136
Machine Learning, LogisticRegression, 10423, 0.7445627577615156, 0.6954811474623429, 0.7966673370758981, 0.7534948741845293, 0.7353342428376535, 0.7725752508361204, 0.1970225937473087, 0.5746290587558862, 0.3983150000021851
Deep Learning, DKT, 10423, 0.6386551920325213, 0.6636285138635709, 0.696801915693621, 0.7740397009538541, 0.6501028472447764, 0.9563624781016086, 0.21823751954513002, 0.6251674733781245, 131.74916539999685
Graph, GKT, 10423, 0.5670374404846454, 0.6255396718794972, 0.6369248203231083, 0.7622296679865976, 0.6172059984214681, 0.9963369963369964, 0.23547777382865856, 0.6638506924787969, 0.15845830000034766
Temporal/Sequential, TIRT, 10423, 0.7468292489643422, 0.6959608557996738, 0.799671215200782, 0.7535961433792084, 0.7362503798237617, 0.7717789456919891, 0.1963812727223177, 0.5732283693186331, 0.6906779999990249
Multi-task, FKT, 10423, 0.5676186048856857, 0.6024177300201478, 0.6432544533614515, 0.7518860016764459, 0.6024177300201478, 1.0, 0.23816522951857821, 0.6694107417802001, 44.647491899999295
Contrastive/Self-supervised, CLKT, 10423, 0.6844068425978365, 0.6581598388179987, 0.7275626739116416, 0.7141137767792667, 0.7195989650711514, 0.7087115782767957, 0.21910392218909958, 0.6278034922808196, 2.348526199999469
Domain Adaptive, AdaptKT, 1515, 0.7053893833257867, 0.7161716171617162, 0.771106054133551, 0.7934678194044188, 0.7475113122171946, 0.8454452405322416, 0.2068417718726254, 0.6039676941595206, 413.47461439999825
```
## Best by metric

- **roc_auc**: TIRT (Temporal/Sequential) = 0.7468 (n=10423)
- **accuracy**: AdaptKT (Domain Adaptive) = 0.7162 (n=1515)
- **avg_precision**: TIRT (Temporal/Sequential) = 0.7997 (n=10423)
- **f1**: AdaptKT (Domain Adaptive) = 0.7935 (n=1515)

Notes: Brier and log_loss are lower-is-better. Metrics skip models that failed.
