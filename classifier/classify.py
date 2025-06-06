import argparse
import pandas as pd
import joblib
import tempfile
from word_vectorizer import load_ngram_surprisal, vectorize_words, FEATURES
from train import LoanwordClassifier
from sklearn.metrics import classification_report

def load_vectors(vector_file):
    df = pd.read_csv(vector_file)
    X = df.drop(columns=["word", "is_loanword", "source"], errors="ignore")
    words = df["word"] if "word" in df.columns else None
    return X, words

def interactive_mode(model, corpus_ngrams):
    print("[I] Enter words to classify (type 'exit' to quit):")
    while True:
        word = input("> ").strip()
        if word.lower() == 'exit':
            break
        df_word = pd.DataFrame({"word": [word]})
        df_vec = vectorize_words(df_word, corpus_ngrams, FEATURES)
        X, _ = load_vectors_from_df(df_vec)
        prob = model.predict_proba(X)[0]
        pred = model.predict(X)[0]
        print(f"[I] Word: {word} | Prediction: {pred} | Probability: {prob:.4f}")

def load_vectors_from_df(df):
    X = df.drop(columns=["word", "is_loanword", "source"], errors="ignore")
    words = df["word"] if "word" in df.columns else None
    return X, words

def main():
    parser = argparse.ArgumentParser(description="Classify words using a saved loanword classifier.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--vector_file", help="CSV file with precomputed features")
    group.add_argument("--word_file", help="CSV file with raw words to be vectorized")
    group.add_argument("--interactive", action="store_true", help="Run in interactive classification mode")

    parser.add_argument("--model", required=True, help="Path to trained model (.pkl)")
    parser.add_argument("--prob_dir", help="Directory with ngram surprisal files (required if using --word_file or --interactive)")
    parser.add_argument("--output_file", help="Optional path to save predictions as CSV")
    args = parser.parse_args()

    model = joblib.load(args.model)

    if args.interactive:
        if not args.prob_dir:
            raise ValueError("--prob_dir is required for interactive mode")
        corpus_ngrams = load_ngram_surprisal(args.prob_dir)
        interactive_mode(model, corpus_ngrams)
        return

    if args.vector_file:
        X, words = load_vectors(args.vector_file)
    elif args.word_file:
        if not args.prob_dir:
            raise ValueError("--prob_dir is required when using --word_file")
        corpus_ngrams = load_ngram_surprisal(args.prob_dir)
        df_words = pd.read_csv(args.word_file)
        df_vec = vectorize_words(df_words, corpus_ngrams, FEATURES)
        X, words = load_vectors_from_df(df_vec)

    probs = model.predict_proba(X)
    preds = model.predict(X)

    df_out = pd.DataFrame({
        "word": words if words is not None else range(len(preds)),
        "pred": preds,
        "prob": probs
    })

    if args.output_file:
        df_out.to_csv(args.output_file, index=False)
        print(f"Saved predictions to {args.output_file}")

    if "is_loanword" in df_vec.columns:
        y_true = df_vec["is_loanword"]
        print("\nClassification Report:")
        print(classification_report(y_true, preds))

if __name__ == "__main__":
    main()
