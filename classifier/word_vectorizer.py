import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Union

# --- Feature classes ---
class ProbabilityFeature:
    def __init__(self, corpus, ngram_size, mode="full", as_suprisal=False, compare_to=None, method="mean"):
        self.corpus = corpus
        self.ngram_size = ngram_size
        self.mode = mode
        self.as_suprisal = as_suprisal
        self.compare_to = compare_to
        self.method = method

class LengthFeature:
    def __init__(self, name="length", transform=None):
        self.name = name
        self.transform = transform

# --- Constants ---

FEATURES = [
    LengthFeature(name="word_length_log", transform="log"),

    ProbabilityFeature("rainis", 2, mode="full"),
    ProbabilityFeature("lv_avizes", 2, mode="full"),
    ProbabilityFeature("lava", 2, mode="full", as_suprisal=True),
    ProbabilityFeature("lv_disertacijas", 2, mode="full", as_suprisal=True),
    ProbabilityFeature("vikipedija", 2, mode="full", as_suprisal=True),

    ProbabilityFeature("lava", 3, mode="full"),
    ProbabilityFeature("vikipedija", 3, mode="full"),
    ProbabilityFeature("lv_disertacijas", 3, mode="full"),
    ProbabilityFeature("rainis", 3, mode="full", as_suprisal=True),
    ProbabilityFeature("lv_avizes", 3, mode="full", as_suprisal=True),

    ProbabilityFeature("lava", 3, mode="suffix"),
    ProbabilityFeature("vikipedija", 3, mode="suffix"),
    ProbabilityFeature("lv_disertacijas", 3, mode="suffix"),
    ProbabilityFeature("rainis", 3, mode="suffix", as_suprisal=True),
    ProbabilityFeature("lv_avizes", 3, mode="suffix", as_suprisal=True),

    ProbabilityFeature("rainis", 3, mode="full", compare_to="vikipedija"),
    ProbabilityFeature("rainis", 3, mode="prefix", compare_to="vikipedija"),
    ProbabilityFeature("rainis", 3, mode="suffix", compare_to="vikipedija"),

    ProbabilityFeature("lv_avizes", 3, mode="full", compare_to="lv_disertacijas"),
    ProbabilityFeature("lv_avizes", 3, mode="prefix", compare_to="lv_disertacijas"),
    ProbabilityFeature("lv_avizes", 3, mode="suffix", compare_to="lv_disertacijas"),

    ProbabilityFeature("lv_avizes", 3, mode="full", compare_to="lava"),
    ProbabilityFeature("lv_avizes", 3, mode="prefix", compare_to="lava"),
    ProbabilityFeature("lv_avizes", 3, mode="suffix", compare_to="lava"),
]
CORPORA_WITH_PROBS = ["rainis", "lv_disertacijas", "vikipedija", "lv_avizes", "lava"]

# --- Helper functions ---
def nested_dict():
    return defaultdict(dict)

def nested_defaultdict():
    return defaultdict(nested_dict)

def load_ngram_surprisal(prob_dir):
    print(f"[I] Loading surprisal data from directory: {prob_dir}")
    corpus_ngrams = defaultdict(nested_defaultdict)
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

    def compute_value(w):
        val = compute_aggregated_surprisal(w, base_dict, feature.ngram_size, method=feature.method)
        return val if feature.as_suprisal else -val

    return words_df["word"].apply(compute_value)

def vectorize_words(words_df, corpus_ngrams, features: List[Union[ProbabilityFeature, LengthFeature]]):
    print(f"[I] Computing features for {len(words_df)} words...")
    for feat in features:
        if isinstance(feat, ProbabilityFeature):
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

        elif isinstance(feat, LengthFeature):
            def transform_length(w):
                if feat.transform == "log":
                    return np.log(len(w)) if len(w) > 0 else 0.0
                elif feat.transform == "sqrt":
                    return np.sqrt(len(w))
                else:
                    return len(w)
            words_df[feat.name] = words_df["word"].apply(transform_length)

    print("[I] Feature computation complete.")
    return words_df

# --- Main routine ---
def main(word_file, prob_dir, output_file):
    print(f"[I] Reading input words from: {word_file}")
    words_df = pd.read_csv(word_file)

    # Add source if it's not in the CSV already
    if "source" not in words_df.columns:
        print("[W] 'source' column missing — trying to infer from 'source'")
        words_df["source"] = words_df["source"].map({
            "etym_dict": "etym_dict",
            "manual_collection": "manual_collection"
        }).fillna("unknown")

    corpus_ngrams = load_ngram_surprisal(prob_dir)

    words_df = vectorize_words(words_df, corpus_ngrams, FEATURES)
    words_df.to_csv(output_file, index=False)
    print(f"[I] Saved feature-enhanced word vectors to {output_file}")

# --- Entry point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute n-gram surprisal features for words.")
    parser.add_argument("--word_file", required=True, help="Path to input CSV file with words")
    parser.add_argument("--prob_dir", required=True, help="Directory containing *_ngram_probs.csv files")
    parser.add_argument("--output_file", default="word_vectors.csv", help="Output CSV file with features")
    args = parser.parse_args()

    main(args.word_file, args.prob_dir, args.output_file)
