# Study 1: Knowledge Tracing Benchmark

This repository contains a compact but fairly complete **knowledge tracing (KT) benchmarking suite** built on top of standardized **itemwise response data**. It is centered around your DigiArvi/EQTd dataset, with an optional ASSISTments baseline.

At a high level, the project provides:

- **Standardized itemwise data** under `data/` (students × items × skills, with response times and sex).
- A **KT benchmark** (`kt_benchmark/`) that trains and evaluates 9 representative model families on this itemwise data.
- Scripts to run the benchmark on **one dataset** or to **compare two datasets** (e.g., DigiArvi vs. ASSISTments) using consistent metrics and plots.

If you are new to this repo, you only need to know how to:

1. Install Python dependencies.
2. Place/inspect your data in `data/`.
3. Run `python -m kt_benchmark.run_benchmark` (single dataset) or
   `python -m kt_benchmark.run_multi_benchmark` (two datasets).

The rest of this README explains those pieces in more detail.

---

## Repository layout

From the project root (`Study 1`):

- **`.git/`**  
  Local Git repository metadata (branches, history).

- **`data/`**  
  Folder for all input data files used by the benchmark. Currently contains:
  - `DigiArvi_25_itemwise.csv` – main DigiArvi/EQTd itemwise dataset.
  - `assistments_09_10_itemwise.csv` – ASSISTments 2009–2010 itemwise dataset.
  - `EQTd_DAi_25_cleaned 3_1 for Prince.xlsx` – original EQTd Excel export used upstream (not read directly by `kt_benchmark`).

  You can add more datasets here as long as they follow the expected **itemwise format** described below.

- **`kt_benchmark/`**  
  Python package implementing the KT benchmark:
  - `config.py` – central configuration (paths, hyperparameters).
  - `run_benchmark.py` – run all 9 model families on a **single** dataset.
  - `run_multi_benchmark.py` – run the benchmark on **two** datasets and create comparative plots.
  - `utils.py` – data loading, preprocessing, and train/test splitting.
  - `metrics.py` – safe wrappers for ROC AUC, accuracy, etc.
  - `models/` – implementations of each KT model family.
  - `plot_results.py` – per-dataset plots (metric curves, trajectories, etc.).
  - `output/` – automatically created; contains metrics, predictions, and plots from runs.
  - `README.md` – model-family–level details and extra notes.

---

## Data format: itemwise responses

All datasets are expected to be **row-wise event logs** in CSV format, with at least the following columns when possible:

- **`IDCode`** – student identifier.
- **`item`** – item/question identifier.
- **`group`** – skill/KC or answer type, depending on how the dataset was constructed.
- **`response`** – binary correctness (0 / 1).
- **`response_time_sec`** – response time in seconds (float). Missing values are allowed; the benchmark will coerce to numeric.
- **`orig_order`** – original sequence index within each student: 1, 2, 3, ... in the order items were seen.
- **`sex`** – standardized to `M`, `F`, or `U` (unknown/other).

The benchmark is fairly robust:

- If some columns are missing (e.g., `sex`, `item_index`), only the models that strictly need them will be affected or skipped.
- If `orig_order` is missing, the code will **reconstruct** a sequence index per student based on row order.

### Current datasets

Under `data/`:

- **`DigiArvi_25_itemwise.csv`**  
  - Main dataset used for most runs.  
  - Columns include at least: `IDCode`, `orig_order`, `item`, `group`, `response`, `response_time_sec`, `sex`.

- **`assistments_09_10_itemwise.csv`**  
  - ASSISTments dataset converted into the same itemwise schema.  
  - Same core columns as above.

- **`EQTd_DAi_25_cleaned 3_1 for Prince.xlsx`**  
  - Original Excel file from earlier preprocessing; **not** used directly by `kt_benchmark` in this repo state.

If you create additional datasets, place their CSVs in `data/` so the runners can find them easily.

---

## Installation

You need **Python 3.8+** (any reasonably recent 3.x is fine) and the packages below.

### Required packages (non-deep models)

- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `seaborn`

### Optional packages (for deep models)

Deep-learning-based models (DKT, FKT-lite, CLKT-lite) require **PyTorch**:

- `torch` (CPU build is enough; GPU is not required).

### Example install commands (PowerShell, from project root)

```powershell
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn scipy matplotlib seaborn

# Optional: for DKT / FKT-lite / CLKT-lite
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If you do **not** install `torch`, the benchmark will automatically skip the deep models and run only the classical/linear ones.

---

## Configuration (`kt_benchmark/config.py`)

Key settings in `kt_benchmark/config.py`:

- **Paths**
  - `BASE_DIR` – automatically set to the project root (parent of `kt_benchmark/`).
  - `DATA_DIR` – `BASE_DIR / "data"`.
  - `INPUT_CSV` – default dataset for single-dataset runs; currently:
    ```python
    INPUT_CSV = DATA_DIR / "DigiArvi_25_itemwise.csv"
    ```
  - `OUTPUT_DIR` – `BASE_DIR / "kt_benchmark" / "output"`.

- **Columns** (names expected in the CSV):
  - `COL_ID = "IDCode"`
  - `COL_ITEM = "item"`
  - `COL_ITEM_INDEX = "item_index"` (optional)
  - `COL_GROUP = "group"`
  - `COL_RESP = "response"`
  - `COL_RT = "response_time_sec"`
  - `COL_ORDER = "orig_order"`
  - `COL_SEX = "sex"`

- **General settings**
  - `RANDOM_STATE` – for reproducible train/test splits.
  - `TEST_SIZE` – fraction of students held out for test (default 0.2).
  - `MIN_STUDENT_EVENTS` – drop students with fewer than this many events.

- **Deep model toggles**
  - `USE_TORCH = True` – if set to False, deep models will be skipped even if `torch` is installed.

You typically only need to edit `INPUT_CSV` if you want a different default dataset, or adjust training durations/hyperparameters for longer runs.

---

## Running the KT benchmark on a single dataset

Script: `kt_benchmark/run_benchmark.py`  
Entry point: `python -m kt_benchmark.run_benchmark`

From the project root (`Study 1`):

```powershell
# Uses config.INPUT_CSV, which default points to data/DigiArvi_25_itemwise.csv
python -m kt_benchmark.run_benchmark
```

This will:

- Load `DATA_DIR / "DigiArvi_25_itemwise.csv"`.
- Clean and standardize types (`response` to numeric 0/1, `response_time_sec` to float, etc.).
- Create a train/test split **by student** using `TEST_SIZE`.
- Train one representative model for each of the 9 families:
  - Rasch (1PL)
  - BKT
  - Logistic Regression
  - DKT (minimal)
  - GKT-lite
  - TIRT-lite
  - FKT-lite (multi-task)
  - CLKT-lite (contrastive)
  - AdaptKT-lite (domain adaptation)
- Compute metrics on the test set (ROC AUC, accuracy, F1, etc.).
- Save metrics, predictions, and a Markdown summary.

### Single-run outputs

All written under:

- `kt_benchmark/output/` (or a subdirectory if you customize it).

Important files include:

- `metrics_summary.csv` – one row per model with key metrics.
- `metrics_summary.md` – human-readable Markdown report including:
  - a list of models with descriptions,
  - a tabular summary of metrics,
  - a "best by metric" section.
- `details.json` – per-model details (e.g., hyperparameters, runtimes) for reproducibility.
- `preds_<model>.csv` – per-model predictions on the test set (true label + predicted probability).
- `preds_<model>__viz.csv` – optional, denser predictions for visualization (when provided).

Some additional figures (e.g., trajectories, grouped metrics) may also be saved by `plot_results.py` into the same `output/` directory.

---

## Comparing two datasets

Script: `kt_benchmark/run_multi_benchmark.py`  
Entry point: `python -m kt_benchmark.run_multi_benchmark`

This script orchestrates **two separate single-dataset runs** and then builds a comparative report.

### Option 1: Let the script auto-pick datasets from `data/`

If you run without arguments, it will look inside `DATA_DIR` (normally `data/`) for these candidates:

- `assistments_09_10_itemwise.csv`
- `DigiArvi_25_itemwise.csv`
- `EQTd_DAi_25_itemwise_with_10_quartile.csv` (if present)

It picks the **first two that exist** and uses those as the two datasets to compare.

Example:

```powershell
python -m kt_benchmark.run_multi_benchmark
```

### Option 2: Provide the two dataset paths explicitly

You can also specify exactly which two CSVs to compare:

```powershell
python -m kt_benchmark.run_multi_benchmark ^
    --datasets data\assistments_09_10_itemwise.csv data\DigiArvi_25_itemwise.csv
```

This is robust to renaming files, as long as you pass the new names.

### Multi-run outputs

For each dataset separately, `run_multi_benchmark`:

1. Creates a per-dataset output directory under:  
   `kt_benchmark/output/<dataset_tag>/`

2. Runs the full benchmark via `run_benchmark`.

3. Generates per-dataset plots via `plot_results.py`.

Then it creates a comparative directory:

- `kt_benchmark/output/compare_<tag1>_vs_<tag2>/`

Key outputs there:

- `metrics_summary_both.csv` – combined metrics for both datasets.
- `comparative_grouped_bars.png` / `.pdf` – grouped bar plots showing model performance by metric across datasets.

(Additional radar plots may exist in the code, but in your version they are typically disabled or de-emphasized in favor of the grouped bars.)

---

## Typical usage patterns

1. **Benchmark DigiArvi only**

   ```powershell
   python -m kt_benchmark.run_benchmark
   ```

2. **Compare DigiArvi vs ASSISTments**

   ```powershell
   python -m kt_benchmark.run_multi_benchmark ^
       --datasets data\assistments_09_10_itemwise.csv data\DigiArvi_25_itemwise.csv
   ```

3. **Switch default single-run dataset**

   Edit `kt_benchmark/config.py`:

   ```python
   INPUT_CSV = DATA_DIR / "assistments_09_10_itemwise.csv"
   ```

   Then:

   ```powershell
   python -m kt_benchmark.run_benchmark
   ```

---

## Notes and tips

- Always run commands from the **project root** (the directory that has `data/` and `kt_benchmark/`).
- If you move or rename CSV files in `data/`, either:
  - update `INPUT_CSV` in `config.py`, or
  - pass explicit paths via `--datasets` to `run_multi_benchmark`.
- If you do not care about deep models, you can skip installing `torch`. The benchmark will still run and provide useful results from Rasch, BKT, Logistic Regression, GKT-lite, TIRT-lite, and AdaptKT-lite.
- For more implementation details about each model family (what exactly Rasch, BKT, DKT, etc. are doing), see `kt_benchmark/README.md`.

This README is meant as the main entry point: someone with only this document and the repository should be able to install dependencies, locate the data, and run the full benchmark and comparative analysis.
