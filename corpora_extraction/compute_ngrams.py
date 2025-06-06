import argparse
import csv
from collections import Counter
from math import log
from itertools import islice

# --- Helper ---
def extract_ngrams(seq, n):
    return zip(*(islice(seq, i, None) for i in range(n)))

def is_valid_ngram(ngram):
    allowed = set("aābcčdeēfgģhiījkķlļlņoprsštuūvzž")
    return all(c in allowed for c in ngram.lower())

# --- Main computation logic ---
def compute_char_ngrams(input_file, n, top_k, use_surprise, mode):
    with open(input_file, encoding='utf-8') as f:
        tokens = [line.strip('<>\n ') for line in f if line.strip()]

    if mode == "prefix":
        char_stream = ''.join(token[:n] for token in tokens if len(token) >= n)
    elif mode == "suffix":
        char_stream = ''.join(token[-n:] for token in tokens if len(token) >= n)
    else:
        char_stream = ''.join(tokens)

    ngrams = extract_ngrams(char_stream, n)
    # ngrams = (ng for ng in ngrams if is_valid_ngram(''.join(ng)))

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

    return scores

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute character n-gram log probabilities or surprise from cleaned token file.")
    parser.add_argument("input_file", help="Path to cleaned <token> file")
    parser.add_argument("--ngram", type=int, default=3, help="Size of character n-gram (default: 3)")
    parser.add_argument("--top_k", type=int, default=30000, help="Number of top n-grams to keep (default: 30000)")
    parser.add_argument("--output", default="char_ngrams_logprob.csv", help="Output CSV file")
    parser.add_argument("--surprise", action="store_true", help="Output negative log-probabilities (surprisal values)")
    parser.add_argument("--mode", choices=["full", "prefix", "suffix"], default="full", help="Use full tokens, prefixes, or suffixes (default: full)")
    args = parser.parse_args()

    result = compute_char_ngrams(args.input_file, args.ngram, args.top_k, args.surprise, args.mode)

    with open(args.output, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ngram", "log_probability" if not args.surprise else "surprisal"])
        for ngram, score in result:
            writer.writerow([ngram, score])

    print(f"Saved {len(result)} {'surprisal' if args.surprise else 'log-prob'} n-grams to {args.output}")
