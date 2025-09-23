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

# Training epochs (kept small for CPU run-time)
EPOCHS_DKT = 5
EPOCHS_MTL = 5
EPOCHS_CLKT = 3

# Logging
VERBOSE = True
