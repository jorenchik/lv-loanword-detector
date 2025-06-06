import pandas as pd
import argparse

def normalize_loanword_status(value: str) -> int:
    """
    Converts Latvian loanword labels to binary:
    'ir' -> 1 (loanword), 'nav' -> 0 (not a loanword)
    """
    value = value.strip().lower()
    if value == "ir":
        return 1
    elif value == "nav":
        return 0
    else:
        raise ValueError(f"Unrecognized loanword status: '{value}'")

def convert_format(input_file, output_file):
    df = pd.read_csv(input_file)

    # Validate required columns
    required = ["vārds", "ir/nav aizguvums"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df_out = pd.DataFrame()
    df_out["word"] = df["vārds"].astype(str).str.strip()
    df_out["is_loanword"] = df["ir/nav aizguvums"].apply(normalize_loanword_status)
    df_out["source"] = "manual_collection"

    df_out.to_csv(output_file, index=False)
    print(f"[I] Converted file saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert manual loanword CSV format to classifier-ready format.")
    parser.add_argument("--input", required=True, help="Path to input CSV file (manual format)")
    parser.add_argument("--output", required=True, help="Path to output CSV file (standard format)")
    args = parser.parse_args()

    convert_format(args.input, args.output)

