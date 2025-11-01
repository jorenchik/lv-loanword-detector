import argparse
import re
from tqdm import tqdm

# --- Helpers ---

def is_shorthand(token):
    return re.fullmatch(r"\w+\.", token) is not None

def is_all_lv_chars(token):
    allowed = set("aābcčdeēfgģhiījkķlļlņoprsštuūvzž")
    return all(c in allowed for c in token.lower())

# --- Cleaning func/filter ---

def transform_token(token):
    return token.lower()

def is_clean(token, only_lv_chars=False):
    return (
        token.isalpha() and
        len(token) > 1 and
        (is_all_lv_chars(token) if only_lv_chars else True)
    )

def clean_tokens(input_file, output_file, only_lv_chars=False):
    with open(input_file, encoding='utf-8') as f:
        raw_tokens = [line.strip('<>\n ') for line in f if line.strip()]

    filtered = [
        transform_token(t)
        for t in tqdm(raw_tokens, desc="Filtering tokens")
        if not is_shorthand(t) and is_clean(t, only_lv_chars)
    ]

    with open(output_file, "w", encoding="utf-8") as out_f:
        for token in filtered:
            out_f.write(f"<{token}>\n")

    print(f"Cleaned tokens saved to {output_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Normalize and filter tokens from <token> format file.")
    parser.add_argument("input_file", help="Path to raw token file")
    parser.add_argument("--output", default="cleaned_tokens.txt", help="Output path for cleaned token file")
    parser.add_argument("--only-lv-chars", action="store_true", help="Skip the is_valid_token check", default=False)

    args = parser.parse_args()
    clean_tokens(args.input_file, args.output, args.only_lv_chars)
