from pathlib import Path
import sys

try:
    import pandas as pd
except ImportError as e:
    sys.stderr.write("pandas is required to run this script. Install with: pip install pandas\n")
    raise


def main():
    base_dir = Path(__file__).parent
    input_csv = base_dir / "EQTd_DAi_25_itemwise.csv"
    output_csv = base_dir / "EQTd_DAi_25_itemwise_minimal.csv"

    desired_columns = [
        "IDCode",
        "sex",
        "group",
        "response",
        "response_time_sec",
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

    trimmed.to_csv(output_csv, index=False)
    print("Saved:", output_csv)
    print(f"Rows: {len(trimmed):,} | Columns: {len(trimmed.columns)} -> {list(trimmed.columns)}")


if __name__ == "__main__":
    main()
