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

    trimmed.to_csv(output_csv, index=False)
    print("Saved:", output_csv)
    print(f"Rows: {len(trimmed):,} | Columns: {len(trimmed.columns)} -> {list(trimmed.columns)}")


if __name__ == "__main__":
    main()
