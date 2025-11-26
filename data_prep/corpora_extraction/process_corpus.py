
import argparse
import re
import random
from collections import Counter

from tqdm import tqdm

EXCLUDE_SYMBOLS  = set(".;, 0123456789+=/()")

# Final alphabet - LV_ALPHABET + "-" ..

# --- Checks ---

# def is_shorthand(token):
#     return re.fullmatch(r"\w+\.", token) is not None

# def is_all_lv_chars(token):
#     global LV_ALPHABET

# --- Cleaning logic ---

ACTIONS = ["transform", "sample", "stats"]

def output_tokens(output_file, tokens):
    with open(output_file, "w", encoding="utf-8") as out_f:
        for t in tokens:
            out_f.write(f"{t}\n")

def sample_tokens(tokens, p):
    # p in [0,1]
    return [t for t in tokens if random.random() < p]

def transform_tokens(tokens):
    out = []
    for t in tokens:
        low = t.lower()

        if len(low) < 2:
            continue

        if any(c in EXCLUDE_SYMBOLS for c in low):
            continue

        if t.upper() == t:
            continue

        out.append(t)

    return out

def compute_stats(tokens, top_k, char_range):
    # character frequency
    char_freq = Counter()

    for t in tokens:
        char_freq.update(t.lower())
    top = char_freq.most_common(top_k)

    lengths = Counter(min(len(t), char_range) for t in tokens)
    length_dist = {i: lengths.get(i, 0) for i in range(1, char_range + 1)}

    return top, length_dist, sum(char_freq.values())

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help="Path to raw token file"
    )
    parser.add_argument(
        "output",
        help="Output path for cleaned token file"
    )
    parser.add_argument(
        "--transform",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--stats",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--sample",
        default=1.0,
        type=float,
        help="Sampling percentage"
    )
    args = parser.parse_args()

    tokens = []
    print("[I] Reading the tokens")
    with open(args.input, "r", encoding='utf-8') as in_f :
        tokens = [line.strip('\n') for line in in_f if line.strip()]
    print(f"[I] Read {len(tokens)} tokens")

    if args.sample > 1 or args.sample < 0:
        print("[E] Sample is between 0 and 1")
        exit(1)

    if args.transform:
        tokens = transform_tokens(tokens)
        print(f"[I] Filtered to {len(tokens)} tokens")

    if args.sample != 1:
        tokens = sample_tokens(tokens, args.sample)
        print(f"[I] Sampled down to {len(tokens)} tokens")

    if args.stats:
        print(f"[I] Computing stats")
        # Reserve 1 for <UKN>.
        stats = compute_stats(tokens, 63, 30)

        for i, s in enumerate(stats[0]):
            if i % 10 == 0:
                print()
            print(s, end=" ")
        print()

        for i, s in enumerate(stats[1]):
            if i % 10 == 0:
                print()
            print(f"{s}:{stats[1][s]}", end=" ")
        print("\n")

        print("Total char amount: ", stats[2])

    output_tokens(args.output, tokens)
    print(f"[I] Wrote theoutput to {args.output}")

if __name__ == "__main__":
    main()
