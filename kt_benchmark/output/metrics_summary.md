# KT Benchmark Summary

Run time: 2025-10-08T01:21:56.120651

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
Psychometric (IRT), Rasch1PL, 6104, 0.33102893033786507, 0.3733617300131062, 0.6146739665241557, 0.49290733130054354, 0.5685015290519878, 0.4350573367657384, 0.6266382699868939, 8.657328008383669, 4.674296500001219
Bayesian, BKT, 4469, 0.694663768844707, 0.7587827254419334, 0.7935454426969721, 0.8318777292576419, 0.8201107011070111, 0.8439873417721518, 0.19169469668239952, 0.5726504001628928, 109.87767239999994
Machine Learning, LogisticRegression, 6104, 0.7981449189486063, 0.7619593709043251, 0.8847196116249127, 0.824580466014729, 0.8516209476309227, 0.7992043061081208, 0.17869723318972588, 0.5410979223292347, 0.24411460000010266
Deep Learning, DKT, 6104, 0.7404383231148092, 0.744429882044561, 0.8495837714381981, 0.830028328611898, 0.7765545361875638, 0.8914111865200094, 0.17655841235219435, 0.5324152991683209, 16.63808340000105
Graph, GKT, 6104, 0.7118181261609514, 0.7313237221494102, 0.8252431147677118, 0.8331977217249796, 0.7368231696348264, 0.9585771120992277, 0.1901286256400014, 0.5669265104213157, 0.2845885000006092
Temporal/Sequential, TIRT, 6104, 0.7991189262899925, 0.7621231979030144, 0.8855329986268918, 0.8248069498069498, 0.851307596513076, 0.7999063889538965, 0.1788231321858179, 0.5414245272651791, 0.2192672000001039
Multi-task, FKT, 6104, 0.7735320774405176, 0.754259501965924, 0.8700561213902007, 0.8347653668208856, 0.7885535900104058, 0.8867306342148373, 0.16827814382197978, 0.5130393494795071, 2.1325368999987404
Contrastive/Self-supervised, CLKT, 6104, 0.7982149610748552, 0.7619593709043251, 0.884847941825011, 0.824580466014729, 0.8516209476309227, 0.7992043061081208, 0.17873742076335744, 0.5411999126820959, 4.69546629999968
Domain Adaptive, AdaptKT, 3052, 0.4708420540366489, 0.6372870249017037, 0.7723111824309268, 0.766307789740342, 0.7622847543049139, 0.7703735144312394, 0.2267232974754026, 0.6457627893216308, 0.16643830000066373
```
## Best by metric

- **roc_auc**: TIRT (Temporal/Sequential) = 0.7991 (n=6104)
- **accuracy**: TIRT (Temporal/Sequential) = 0.7621 (n=6104)
- **avg_precision**: TIRT (Temporal/Sequential) = 0.8855 (n=6104)
- **f1**: FKT (Multi-task) = 0.8348 (n=6104)

Notes: Brier and log_loss are lower-is-better. Metrics skip models that failed.
