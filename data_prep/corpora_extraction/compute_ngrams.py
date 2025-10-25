import argparse
import csv
from collections import Counter
from math import log
from itertools import islice, product
from statistics import mean, stdev

# --- Helper ---
def extract_ngrams(seq, n):
    return zip(*(islice(seq, i, None) for i in range(n)))

def is_valid_ngram(ngram):
    allowed = set("aābcčdeēfgģhiīijkķlļlņoprsštuūvzž")
    return all(c in allowed for c in ngram.lower())

def print_summary(summary_stats, use_surprise):
    print("\nSummary Report:")
    header = [
        "ngram_size", "mode", "count", 
        "top_ngram", 
        "top_score" if not use_surprise else "lowest_surprisal", 
        "mean", "min", "max", "stdev"
    ]
    print(" | ".join(header))
    print("-" * 80)
    for row in summary_stats:
        print(f"{row['ngram_size']:>10} | {row['mode']:<6} | {row['count']:>5} | "
              f"{row['top_ngram']:<10} | {row['top_score']:.4f} | "
              f"{row['mean']:.4f} | {row['min']:.4f} | {row['max']:.4f} | {row['stdev']:.4f}")
    print()

# --- Main computation logic ---
def compute_char_ngrams(tokens, n, top_k, use_surprise, mode):
    if mode == "prefix":
        char_stream = ''.join(token[:n] for token in tokens if len(token) >= n)
    elif mode == "suffix":
        char_stream = ''.join(token[-n:] for token in tokens if len(token) >= n)
    else:
        char_stream = ''.join(tokens)

    ngrams = extract_ngrams(char_stream, n)
    ngram_counts = Counter(ngrams)
    total_count = sum(ngram_counts.values())

    most_common = ngram_counts.most_common(top_k)

    if use_surprise:
        scores = [
            (''.join(ng), -log(count / total_count))
            for ng, count in most_common
        ]
    else:
        scores = [
            (''.join(ng), log(count / total_count))
            for ng, count in most_common
        ]

    is_prefix = mode == "prefix"
    is_suffix = mode == "suffix"
    extended_scores = [
        (ngram, score, len(ngram), is_prefix, is_suffix, n, mode)
        for ngram, score in scores
    ]

    return extended_scores

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute character n-gram log probabilities or surprise from cleaned token file.")
    parser.add_argument("input_file", help="Path to cleaned <token> file")
    parser.add_argument("--ngram", type=int, nargs="+", default=[3], help="One or more n-gram sizes (e.g. --ngram 2 3 4)")
    parser.add_argument("--top_k", type=int, default=30000, help="Default number of top n-grams to keep")
    parser.add_argument("--top_k_map", nargs="+", help="Override top_k for specific n, mode, or both. E.g. 3:1000 prefix:500 4,suffix:200")
    parser.add_argument("--output", default="char_ngrams_logprob.csv", help="Output CSV file")
    parser.add_argument("--surprise", action="store_true", help="Output negative log-probabilities (surprisal values)")
    parser.add_argument("--mode", choices=["full", "prefix", "suffix"], nargs="+", default=["full"], help="One or more modes: full, prefix, suffix")
    args = parser.parse_args()

    # Parse input
    with open(args.input_file, encoding='utf-8') as f:
        tokens = [line.strip('<>\n ') for line in f if line.strip()]

    # Parse --top_k_map into a dispatchable override dictionary
    top_k_overrides = {}
    if args.top_k_map:
        for entry in args.top_k_map:
            try:
                key, value = entry.split(":")
                value = int(value)
                if "," in key:
                    n_str, mode = key.split(",")
                    top_k_overrides[(int(n_str), mode)] = value
                elif key.isdigit():
                    top_k_overrides[(int(key), None)] = value
                else:
                    top_k_overrides[(None, key)] = value
            except Exception as e:
                raise ValueError(f"Invalid --top_k_map entry: '{entry}' ({e})")

    # --- Process combinations ---
    all_results = []
    summary_stats = []

    for n, mode in product(args.ngram, args.mode):
        # Determine the top_k using most specific applicable rule
        key_exact = (n, mode)
        key_n_only = (n, None)
        key_mode_only = (None, mode)

        top_k = (
            top_k_overrides.get(key_exact) or
            top_k_overrides.get(key_n_only) or
            top_k_overrides.get(key_mode_only) or
            args.top_k
        )

        results = compute_char_ngrams(tokens, n, top_k, args.surprise, mode)

        if not results:
            print(f"[INFO] Skipping n={n}, mode='{mode}' — no valid n-grams.")
            continue

        all_results.extend(results)

        scores_only = [score for _, score, *_ in results]

        top_result = min(results, key=lambda x: x[1]) if args.surprise else max(results, key=lambda x: x[1])
        top_ngram = top_result[0]
        top_score = top_result[1]

        stat = {
            "ngram_size": n,
            "mode": mode,
            "count": len(scores_only),
            "top_ngram": top_ngram,
            "top_score": top_score,
            "mean": mean(scores_only),
            "min": min(scores_only),
            "max": max(scores_only),
            "stdev": stdev(scores_only) if len(scores_only) > 1 else 0.0
        }
        summary_stats.append(stat)

    # --- Output detailed data ---
    with open(args.output, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "ngram",
            "log_probability" if not args.surprise else "surprisal",
            "length",
            "is_prefix",
            "is_suffix",
            "ngram_size",
            "mode"
        ])
        for row in all_results:
            writer.writerow(row)

    # --- Print summary ---
    print_summary(summary_stats, args.surprise)

    print(f"Saved {len(all_results)} n-gram entries to {args.output}")
