from pathlib import Path
import sys
import argparse
from datetime import datetime
import shutil

try:
    import pandas as pd
except ImportError as e:
    sys.stderr.write("pandas is required to run this script. Install with: pip install pandas\n")
    raise


def main():
    base_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Trim EQTd_DAi_25_itemwise.csv to minimal columns required by KT benchmark models.")
    parser.add_argument("--input", type=str, default=str((base_dir / "EQTd_DAi_25_itemwise.csv").resolve()), help="Path to the input itemwise CSV")
    parser.add_argument("--output", type=str, default=None, help="Path to write the trimmed CSV (default saves alongside as *_minimal.csv). If --overwrite is set, this is ignored.")
    parser.add_argument("--overwrite", action="store_true", help="If set, back up the input CSV and overwrite it in place with the minimal columns.")
    args = parser.parse_args()

    input_csv = Path(args.input)
    # Decide output path
    if args.overwrite:
        output_csv = input_csv
    else:
        if args.output:
            output_csv = Path(args.output)
        else:
            output_csv = input_csv.with_name(input_csv.stem + "_minimal" + input_csv.suffix)

    # Minimal columns needed across all 9 models
    desired_columns = [
        "IDCode",            # student id (all models)
        "orig_order",        # sequencing/time index (BKT/TIRT/DKT/etc.)
        "item",              # item identifier (Rasch/LogReg/TIRT/Contrastive/etc.)
        "group",             # skill/KC (BKT/GKT/Adapt/LogReg/TIRT)
        "response",          # binary label (all predictive models)
        "response_time_sec", # auxiliary target (MTL) and optional feature
        "sex",               # standardized covariate (LogReg/MTL/Adapt)
    ]

    if not input_csv.exists():
        sys.stderr.write(f"Input file not found: {input_csv}\n")
        sys.exit(1)

    print(f"Reading: {input_csv}")
    df = pd.read_csv(input_csv)

    missing = [c for c in desired_columns if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns in input: {missing}")

    present_cols = [c for c in desired_columns if c in df.columns]
    if not present_cols:
        sys.stderr.write("None of the desired columns are present. Exiting.\n")
        sys.exit(2)

    # Keep only the requested columns, in the requested order
    trimmed = df.loc[:, present_cols]

    # Normalize sex to 'M', 'F', or 'U' (unknown)
    if "sex" in trimmed.columns:
        # Work on a cleaned string version, strip non-letters for robust mapping
        sex_raw = trimmed["sex"]
        sex_norm = sex_raw.astype(str).str.strip().str.lower().str.replace(r"[^a-z]", "", regex=True)
        mapping = {
            "boy": "M", "b": "M", "male": "M", "m": "M", "man": "M",
            "girl": "F", "gir": "F", "g": "F", "female": "F", "f": "F", "woman": "F",
        }
        mapped = sex_norm.map(mapping)
        trimmed["sex"] = mapped.fillna("U")
        print("Normalized 'sex' to M/F/U. Counts:", trimmed["sex"].value_counts(dropna=False).to_dict())

    # Impute missing response_time_sec where possible
    if "response_time_sec" in trimmed.columns:
        # Ensure numeric dtype for safe operations
        trimmed["response_time_sec"] = pd.to_numeric(trimmed["response_time_sec"], errors="coerce")
        before_na = int(trimmed["response_time_sec"].isna().sum())

        if before_na > 0:
            # Prefer imputation by group median if 'group' is available
            if "group" in trimmed.columns:
                group_median = trimmed.groupby("group")["response_time_sec"].transform(lambda s: s.median(skipna=True))
                trimmed["response_time_sec"] = trimmed["response_time_sec"].fillna(group_median)

            # Fallback to overall median
            overall_median = trimmed["response_time_sec"].median(skipna=True)
            trimmed["response_time_sec"] = trimmed["response_time_sec"].fillna(overall_median)

            after_na = int(trimmed["response_time_sec"].isna().sum())
            filled = before_na - after_na
            print(f"Imputed response_time_sec: filled {filled} missing values (remaining {after_na}).")

    # If overwriting, create a timestamped backup first
    if args.overwrite:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_csv = input_csv.with_name(f"{input_csv.stem}_backup_{ts}{input_csv.suffix}")
        try:
            shutil.copy2(input_csv, backup_csv)
            print("Backed up original to:", backup_csv)
        except Exception as e:
            print("Warning: failed to back up original:", e)

    trimmed.to_csv(output_csv, index=False)
    print("Saved:", output_csv)
    print(f"Rows: {len(trimmed):,} | Columns: {len(trimmed.columns)} -> {list(trimmed.columns)}")


if __name__ == "__main__":
    main()
