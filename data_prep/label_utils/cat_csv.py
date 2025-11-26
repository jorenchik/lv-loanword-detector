import argparse
import pandas as pd
import os

def main():

    parser = argparse.ArgumentParser("Combine CSV files only if columns match exactly")
    parser.add_argument("files", nargs="+", help="List of CSV files, last is output")
    args = parser.parse_args()

    if len(args.files) < 2:
        raise ValueError("Provide at least one input file and one output file.")

    *input_files, output_file = args.files

    combined = []
    ref_columns = None

    for path in input_files:
        fname = os.path.basename(path)
        if not path.lower().endswith(".csv"):
            print(f"Skipped {fname} (not a CSV file)")
            continue

        df = pd.read_csv(path)

        if ref_columns is None:
            ref_columns = list(df.columns)
            combined.append(df)
            print(f"Using columns from {fname}: {ref_columns}")
        else:
            if list(df.columns) != ref_columns:
                raise ValueError(
                    f"Column mismatch in '{fname}'. Expected {ref_columns}, got {list(df.columns)}"
                )
            combined.append(df)
            print(f"Added {fname}")

    if not combined:
        raise ValueError("No CSV files found.")

    result = pd.concat(combined, ignore_index=True)
    result.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")


if __name__ == "__main__":
    main()
