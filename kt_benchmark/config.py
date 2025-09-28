from pathlib import Path

# Paths
BASE_DIR = Path(r"c:/Users/pdaadh/Desktop/Study 2")
INPUT_CSV = BASE_DIR / "assistments_09_10_itemwise.csv"
OUTPUT_DIR = BASE_DIR / "kt_benchmark" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns expected (some may be missing; code will degrade gracefully)
COL_ID = "IDCode"
COL_ITEM = "item"
COL_ITEM_INDEX = "item_index"
COL_GROUP = "group"
COL_RESP = "response"
COL_RT = "response_time_sec"
COL_ORDER = "orig_order"
COL_SEX = "sex"

# General
RANDOM_STATE = 42
TEST_SIZE = 0.2  # by students
MIN_STUDENT_EVENTS = 5  # drop students with fewer events

# Deep learning toggles (optional; runner will skip if torch is not available)
USE_TORCH = True

# Training durations (set higher for longer training)
EPOCHS_DKT = 20          # DKT epochs
EPOCHS_MTL = 20          # FKT-lite epochs
EPOCHS_CLKT = 10         # If used by any CLKT fine-tuning variants

# Fairness: per-model wall-clock budget (seconds). Models with iterative loops
# (e.g., scikit-learn solvers) may ignore it but typically finish under budget.
TRAIN_TIME_BUDGET_S = 120

# Rasch (1PL)
RASCH_MAX_ITER = 100
RASCH_INNER_ITER = 5

# BKT coarse grid (denser for longer fitting)
BKT_GRID_L0 = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
BKT_GRID_T  = [0.05, 0.10, 0.15, 0.20, 0.30]
BKT_GRID_G  = [0.05, 0.10, 0.15, 0.20, 0.25]
BKT_GRID_S  = [0.05, 0.10, 0.15, 0.20, 0.25]
# Control whether BKT fits per-skill (group) or a single global model across all skills.
# Setting this to False will generally reduce BKT performance.
BKT_PER_GROUP = False
# Graph smoothing
GKT_SMOOTH_ALPHA = 0.7

# Logistic Regression max iterations
LOGREG_MAX_ITER = 500   # plain LogisticRegression
TIRT_MAX_ITER   = 500   # TIRT-lite's logistic head
CLKT_SUP_MAX_ITER = 5000 # supervised LR after contrastive pretraining
ADAPT_MAX_ITER  = 5000   # AdaptKT-lite classifier

# Linear model feature toggles and solver tolerances
# If Assistments runs are slow due to very high KC/skill cardinality, you can
# set LINEAR_USE_GROUP = False to drop group one-hots for linear baselines.
LINEAR_USE_GROUP = True
LOGREG_TOL = 1e-3
TIRT_TOL = 1e-3

# DKT architecture & training
DKT_EMB_DIM = 64
DKT_HID_DIM = 128
DKT_LR = 1e-3
MTL_HID_DIM = 128
MTL_LR = 1e-3
MTL_BATCH = 256

# CLKT contrastive pretraining
CLKT_PRE_STEPS = 2000
CLKT_PRE_DIM = 64
CLKT_PRE_LR = 1e-2
CLKT_PRE_BATCH = 512

# Logging
VERBOSE = True

# ---------------------------------------------------------------------------
# New: Lightweight tuning/regularization controls to improve model performance
# while respecting the global TRAIN_TIME_BUDGET_S.
# ---------------------------------------------------------------------------

# Logistic/TIRT C grids
LOGREG_C_GRID = [0.01, 0.1, 1.0, 10.0]
TIRT_C_GRID   = [0.01, 0.1, 1.0, 10.0]

# GKT smoothing alpha grid (will pick best on a small validation split of train)
GKT_ALPHA_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]

# DKT training controls
DKT_BATCH = 64
DKT_DROPOUT = 0.2
DKT_WEIGHT_DECAY = 1e-4
DKT_PATIENCE = 3

# MTL training controls
MTL_DROPOUT = 0.2
MTL_WEIGHT_DECAY = 1e-4
MTL_PATIENCE = 3
