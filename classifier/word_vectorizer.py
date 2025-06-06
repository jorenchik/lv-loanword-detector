import pandas as pd
import os
import argparse
from collections import defaultdict
import numpy as np

# Define corpora with probability files
CORPORA_WITH_PROBS = [
    "rainis",
    "lv_disertacijas",
    "vikipedija"
]

def load_ngram_surprisal(prob_dir):
    corpus_ngrams = defaultdict(lambda: defaultdict(dict))  # corpus -> ngram_size -> ngram -> surprisal
    for corpus_key in CORPORA_WITH_PROBS:
        fname = f"{corpus_key}_ngram_probs.csv"
        fpath = os.path.join(prob_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Expected ngram file for corpus '{corpus_key}' not found: {fname}")
        df = pd.read_csv(fpath)
        for _, row in df.iterrows():
            if row["mode"] != "full":
                continue
            corpus_ngrams[corpus_key][row["ngram_size"]][row["ngram"]] = row["surprisal"]
    return corpus_ngrams


def get_ngrams(word, n):
    return [word[i:i+n] for i in range(len(word) - n + 1)]


def compute_aggregated_surprisal(word, ngram_dict, n):
    ngrams = get_ngrams(word, n)
    surprisals = [ngram_dict.get(ng, np.nan) for ng in ngrams]
    surprisals = [s for s in surprisals if not np.isnan(s)]
    if not surprisals:
        return np.nan
    return float(np.mean(surprisals))


def main(word_file, prob_dir, output_file):
    words_df = pd.read_csv(word_file)
    corpus_ngrams = load_ngram_surprisal(prob_dir)

    for corpus_key in CORPORA_WITH_PROBS:
        for ngram_size in [2, 3]:
            col_name = f"{ngram_size}_grams_{corpus_key}_surprisal"
            words_df[col_name] = words_df["word"].apply(
                lambda w: compute_aggregated_surprisal(w, corpus_ngrams[corpus_key][ngram_size], ngram_size)
            )

    words_df.to_csv(output_file, index=False)
    print(f"Saved feature-enhanced word vectors to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute n-gram surprisal features for words.")
    parser.add_argument("--word_file", required=True, help="Path to input CSV file with words")
    parser.add_argument("--prob_dir", required=True, help="Directory containing *_ngram_probs.csv files")
    parser.add_argument("--output_file", default="word_vectors.csv", help="Output CSV file with features")
    args = parser.parse_args()

    main(args.word_file, args.prob_dir, args.output_file)
