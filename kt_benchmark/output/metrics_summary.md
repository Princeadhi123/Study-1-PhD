# KT Benchmark Summary

Run time: 2025-10-07T22:10:58.486296

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
Psychometric (IRT), Rasch1PL, 6104, 0.33102893033786507, 0.3733617300131062, 0.6146739665241557, 0.49290733130054354, 0.5685015290519878, 0.4350573367657384, 0.6266382699868939, 8.657328008383669, 4.678750000000036
Bayesian, BKT, 4469, 0.694663768844707, 0.7587827254419334, 0.7935454426969721, 0.8318777292576419, 0.8201107011070111, 0.8439873417721518, 0.19169469668239952, 0.5726504001628928, 104.22914409999976
Machine Learning, LogisticRegression, 6104, 0.7981449189486063, 0.7619593709043251, 0.8847196116249127, 0.824580466014729, 0.8516209476309227, 0.7992043061081208, 0.17869723318972588, 0.5410979223292347, 0.23185319999993226
Deep Learning, DKT, 6104, 0.7407656550223337, 0.744429882044561, 0.8493627397784433, 0.830028328611898, 0.7765545361875638, 0.8914111865200094, 0.17626754662590105, 0.5316971067397884, 13.143782100000408
Graph, GKT, 6104, 0.7118181261609514, 0.7313237221494102, 0.8252431147677118, 0.8331977217249796, 0.7368231696348264, 0.9585771120992277, 0.1901286256400014, 0.5669265104213157, 0.2639717000001838
Temporal/Sequential, TIRT, 6104, 0.7991189262899925, 0.7621231979030144, 0.8855329986268918, 0.8248069498069498, 0.851307596513076, 0.7999063889538965, 0.1788231321858179, 0.5414245272651791, 0.20540340000025026
Multi-task, FKT, 6104, 0.792933299062113, 0.7693315858453473, 0.8805275338423321, 0.8400363553737786, 0.8162949878560388, 0.8652000936110461, 0.16009905182770143, 0.4915955866845567, 2.984245000000101
Contrastive/Self-supervised, CLKT, 6104, 0.7981480503940317, 0.7619593709043251, 0.8848080410294712, 0.824580466014729, 0.8516209476309227, 0.7992043061081208, 0.17872996226096421, 0.5411964174395599, 4.647929100000056
Domain Adaptive, AdaptKT, 3052, 0.4708420540366489, 0.6372870249017037, 0.7723111824309268, 0.766307789740342, 0.7622847543049139, 0.7703735144312394, 0.2267232974754026, 0.6457627893216308, 0.13645190000033836
```
## Best by metric

- **roc_auc**: TIRT (Temporal/Sequential) = 0.7991 (n=6104)
- **accuracy**: FKT (Multi-task) = 0.7693 (n=6104)
- **avg_precision**: TIRT (Temporal/Sequential) = 0.8855 (n=6104)
- **f1**: FKT (Multi-task) = 0.8400 (n=6104)

Notes: Brier and log_loss are lower-is-better. Metrics skip models that failed.
