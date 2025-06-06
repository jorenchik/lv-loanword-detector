import argparse
import re
from tqdm import tqdm

# --- Normalization and filtering helpers ---
def normalize_token(token):
    return token.lower()

def is_shorthand(token):
    return re.fullmatch(r"\w+\.", token) is not None

def is_valid_token(token):
    allowed = set("aābcčdeēfgģhiījkķlļlņoprsštuūvzž")
    return all(c in allowed for c in token.lower())

def is_clean(token):
    return (
        token.isalpha() and
        len(token) > 1 and
        is_valid_token(token)
    )

# --- Main cleaning logic ---
def clean_tokens(input_file, output_file):
    with open(input_file, encoding='utf-8') as f:
        raw_tokens = [line.strip('<>\n ') for line in f if line.strip()]

    filtered = [normalize_token(t) for t in tqdm(raw_tokens, desc="Filtering tokens") if not is_shorthand(t) and is_clean(t)]

    with open(output_file, "w", encoding="utf-8") as out_f:
        for token in filtered:
            out_f.write(f"<{token}>\n")

    print(f"Cleaned tokens saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize and filter tokens from <token> format file.")
    parser.add_argument("input_file", help="Path to raw token file")
    parser.add_argument("--output", default="cleaned_tokens.txt", help="Output path for cleaned token file")
    args = parser.parse_args()
    clean_tokens(args.input_file, args.output)
