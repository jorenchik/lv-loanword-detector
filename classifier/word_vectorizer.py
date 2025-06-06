import pandas as pd
import os
import argparse
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ProbabilityFeature:
    corpus: str
    ngram_size: int
    mode: str = "full"  # full, prefix, suffix
    method: str = "max"  # max, mean, etc.
    compare_to: Optional[str] = None  # other corpus to compare to (for difference)
    as_suprisal: bool = True  # if False, treat as raw probability (negate surprise)

FEATURES = [

    # Raw surprisal scores (for older corpora)

    # 2-grams
    ProbabilityFeature("rainis", 2, mode="full"),
    ProbabilityFeature("lv_avizes", 2, mode="full"),
    ProbabilityFeature("lv_disertacijas", 2, mode="full", as_suprisal=False),
    ProbabilityFeature("vikipedija", 2, mode="full", as_suprisal=False),

    # 3-grams
    ProbabilityFeature("vikipedija", 3, mode="full"),
    ProbabilityFeature("lv_disertacijas", 3, mode="full"),
    ProbabilityFeature("rainis", 3, mode="full", as_suprisal=False),
    ProbabilityFeature("lv_avizes", 3, mode="full", as_suprisal=False),

    # Suffix and Prefix based surprisal
    ProbabilityFeature("vikipedija", 3, mode="suffix"),
    ProbabilityFeature("lv_disertacijas", 3, mode="suffix"),
    ProbabilityFeature("rainis", 3, mode="suffix", as_suprisal=False),
    ProbabilityFeature("lv_avizes", 3, mode="suffix", as_suprisal=False),

    # Contrasts: modern vs traditional sources
    ProbabilityFeature("rainis", 3, mode="full", compare_to="vikipedija"),
    ProbabilityFeature("rainis", 3, mode="prefix", compare_to="vikipedija"),
    ProbabilityFeature("rainis", 3, mode="suffix", compare_to="vikipedija"),

    ProbabilityFeature("lv_avizes", 3, mode="full", compare_to="lv_disertacijas"),
    ProbabilityFeature("lv_avizes", 3, mode="prefix", compare_to="lv_disertacijas"),
    ProbabilityFeature("lv_avizes", 3, mode="suffix", compare_to="lv_disertacijas"),
]

CORPORA_WITH_PROBS = ["rainis", "lv_disertacijas", "vikipedija", "lv_avizes"]

def load_ngram_surprisal(prob_dir):
    print(f"[I] Loading surprisal data from directory: {prob_dir}")
    corpus_ngrams = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for corpus_key in CORPORA_WITH_PROBS:
        fname = f"{corpus_key}_ngram_probs.csv"
        fpath = os.path.join(prob_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Expected ngram file for corpus '{corpus_key}' not found: {fname}")
        print(f"[I] Reading {fname}")
        df = pd.read_csv(fpath)
        for _, row in df.iterrows():
            corpus_ngrams[corpus_key][row["mode"]][row["ngram_size"]][row["ngram"]] = row["surprisal"]
    print("[I] Surprisal loading complete.")
    return corpus_ngrams

def get_ngrams(word, n):
    return [word[i:i+n] for i in range(len(word) - n + 1)]

# def compute_aggregated_surprisal(word, ngram_dict, n, method="max"):
#     ngrams = get_ngrams(word, n)
#     values = [ngram_dict.get(ng) for ng in ngrams if ng in ngram_dict and not np.isnan(ngram_dict[ng])]
#
#     if values:
#         return float(np.mean(values))
#     else:
#         return 20.0
def compute_aggregated_surprisal(word, ngram_dict, n, method="max"):
    ngrams = get_ngrams(word, n)
    surprisals = [ngram_dict.get(ng, np.nan) for ng in ngrams]
    surprisals = [s for s in surprisals if not np.isnan(s)]

    if not surprisals:
        return np.nan

    return float(np.mean(surprisals))

def compute_feature_column(words_df, corpus_ngrams, feature: ProbabilityFeature):
    base_dict = corpus_ngrams[feature.corpus][feature.mode][feature.ngram_size]

    if feature.compare_to:
        compare_dict = corpus_ngrams[feature.compare_to][feature.mode][feature.ngram_size]

        def compute_both(w):
            left = compute_aggregated_surprisal(w, base_dict, feature.ngram_size, method=feature.method)
            right = compute_aggregated_surprisal(w, compare_dict, feature.ngram_size, method=feature.method)
            if np.isnan(left) or np.isnan(right):
                return pd.Series([np.nan, np.nan, np.nan])
            diff = left - right
            if not feature.as_suprisal:
                diff = -diff
            return pd.Series([diff, left, right])

        return words_df["word"].apply(compute_both).rename(columns={0: "diff", 1: "left", 2: "right"})

    else:
        return words_df["word"].apply(
            lambda w: compute_aggregated_surprisal(w, base_dict, feature.ngram_size, method=feature.method)
            if feature.as_suprisal else
            -compute_aggregated_surprisal(w, base_dict, feature.ngram_size, method=feature.method)
        )

def vectorize_words(words_df, corpus_ngrams, features: List[ProbabilityFeature]):
    print(f"[I] Computing surprisal features for {len(words_df)} words...")
    for feat in features:
        base = f"{feat.ngram_size}g_{feat.mode}_{feat.corpus}"
        if feat.compare_to:
            col_name = f"{base}_diff_{feat.compare_to}"
            results = compute_feature_column(words_df, corpus_ngrams, feat)
            words_df[f"{col_name}"] = results["diff"]
            words_df[f"{base}_vs_{feat.compare_to}_left"] = results["left"]
            words_df[f"{base}_vs_{feat.compare_to}_right"] = results["right"]
        else:
            col_name = f"{base}_prob" if not feat.as_suprisal else base
            words_df[col_name] = compute_feature_column(words_df, corpus_ngrams, feat)
    print("[I] Feature computation complete.")
    return words_df

def main(word_file, prob_dir, output_file):
    print(f"[I] Reading input words from: {word_file}")
    words_df = pd.read_csv(word_file)
    corpus_ngrams = load_ngram_surprisal(prob_dir)
    words_df = vectorize_words(words_df, corpus_ngrams, FEATURES)
    words_df.to_csv(output_file, index=False)
    print(f"[I] Saved feature-enhanced word vectors to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute n-gram surprisal features for words.")
    parser.add_argument("--word_file", required=True, help="Path to input CSV file with words")
    parser.add_argument("--prob_dir", required=True, help="Directory containing *_ngram_probs.csv files")
    parser.add_argument("--output_file", default="word_vectors.csv", help="Output CSV file with features")
    args = parser.parse_args()

    main(args.word_file, args.prob_dir, args.output_file)
