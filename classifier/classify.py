import argparse
import pandas as pd
import joblib
import tempfile
from sklearn.metrics import classification_report

from classifier.model import LoanwordClassifier
from classifier.word_vectorizer import FEATURES

def load_vectors(vector_file):
    df = pd.read_csv(vector_file)
    X = df.drop(columns=["word", "is_loanword", "source"], errors="ignore")
    words = df["word"] if "word" in df.columns else None
    return X, words, df

def interactive_mode(model):
    print("[I] Enter words to classify (type 'exit' to quit):")
    while True:
        word = input("> ").strip()
        if word.lower() == 'exit':
            break
        df_word = pd.DataFrame({"word": [word]})
        X, _ = model.vectorize_words(df_word)
        prob = model.predict_proba(X)[0]
        pred = model.predict(X)[0]
        print(f"[I] Word: {word} | Prediction: {pred} | Probability: {prob:.4f}")

def main():

    parser = argparse.ArgumentParser(description="Classify words using a saved loanword classifier.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--vector_file", help="CSV file with precomputed features")
    group.add_argument("--word_file", help="CSV file with raw words to be vectorized")
    group.add_argument("--interactive", action="store_true", help="Run in interactive classification mode")

    parser.add_argument("--threshold", type=float, default=None, help="Path to trained model (.pkl)")
    parser.add_argument("--model", required=True, help="Path to trained model (.pkl)")
    parser.add_argument("--output_file", help="Optional path to save predictions as CSV")
    parser.add_argument("--filter_source", help="Optional source filter to evaluate only rows from this source")
    args = parser.parse_args()

    model = joblib.load(args.model)
    print("[I] Base threshold:", "%.2f" % model.threshold)

    if args.interactive:
        interactive_mode(model)
        return

    if args.vector_file:
        X, words, df_vec = load_vectors(args.vector_file)
    elif args.word_file:
        df_words = pd.read_csv(args.word_file)
        X, df_vec = model.vectorize_words(df_words)
        words = df_words["word"]
    else:
        raise ValueError("Either --vector_file or --word_file must be provided")

    probs = model.predict_proba(X)
    if args.threshold:
        preds = model.predict(X, args.threshold)
    else: 
        preds = model.predict(X)

    df_out = pd.DataFrame({
        "word": words if words is not None else range(len(preds)),
        "pred": preds,
        "prob": probs
    })

    if args.output_file:
        df_eval = df_vec.copy()
        df_out["truth"] = df_eval["is_loanword"]
        df_out.to_csv(args.output_file, index=False)
        print(f"Saved predictions to {args.output_file}")

    if "is_loanword" in df_vec.columns:
        df_eval = df_vec.copy()
        if args.filter_source and "source" in df_eval.columns:
            df_eval = df_eval[df_eval["source"] == args.filter_source]
        if not df_eval.empty:
            y_true = df_eval["is_loanword"]
            y_pred = preds[df_eval.index]
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred))
        else:
            print("[W] No data matches the specified source filter.")

if __name__ == "__main__":
    main()
