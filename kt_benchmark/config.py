from pathlib import Path

# Paths
BASE_DIR = Path(r"c:/Users/pdaadh/Desktop/Study 2")
INPUT_CSV = BASE_DIR / "EQTd_DAi_25_itemwise.csv"
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

# Rasch (1PL)
RASCH_MAX_ITER = 100
RASCH_INNER_ITER = 5

# BKT coarse grid (denser for longer fitting)
BKT_GRID_L0 = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
BKT_GRID_T  = [0.05, 0.10, 0.15, 0.20, 0.30]
BKT_GRID_G  = [0.05, 0.10, 0.15, 0.20, 0.25]
BKT_GRID_S  = [0.05, 0.10, 0.15, 0.20, 0.25]

# Graph smoothing
GKT_SMOOTH_ALPHA = 0.7

# Logistic Regression max iterations
LOGREG_MAX_ITER = 5000   # plain LogisticRegression
TIRT_MAX_ITER   = 5000   # TIRT-lite's logistic head
CLKT_SUP_MAX_ITER = 5000 # supervised LR after contrastive pretraining
ADAPT_MAX_ITER  = 5000   # AdaptKT-lite classifier

# DKT architecture & training
DKT_EMB_DIM = 64
DKT_HID_DIM = 128
DKT_LR = 1e-3
DKT_BATCH = 64

# FKT-lite (MTL) architecture & training
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
