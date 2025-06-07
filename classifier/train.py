import pandas as pd
import argparse
import os
import tempfile
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from word_vectorizer import load_ngram_surprisal, vectorize_words, FEATURES

# -- Classifier wrapper
class LoanwordClassifier:
    def __init__(self, classifier, threshold, imputer, scaler=None, corpus_ngrams=None):
        self.classifier = classifier
        self.threshold = threshold
        self.imputer = imputer
        self.scaler = scaler
        self.corpus_ngrams = corpus_ngrams

    def _preprocess(self, X):
        X_imputed = self.imputer.transform(X)
        if self.scaler:
            X_imputed = self.scaler.transform(X_imputed)
        return X_imputed

    def predict_proba(self, X):
        X_proc = self._preprocess(X)
        return self.classifier.predict_proba(X_proc)[:, 1]

    def predict(self, X, threshold=None):
        probs = self.predict_proba(X)
        if threshold is not None:
            return (probs >= threshold).astype(int)
        else:
            return (probs >= self.threshold).astype(int)

    def vectorize_words(self, df_words):
        if self.corpus_ngrams is None:
            raise ValueError("corpus_ngrams not set in this model.")
        df_vec = vectorize_words(df_words, self.corpus_ngrams, FEATURES)
        X = df_vec.drop(columns=["word", "is_loanword", "source"], errors="ignore")
        return X, df_vec

# -- Helpers
def load_data(input_file):
    print(f"[I] Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    X = df.drop(columns=["word", "is_loanword", "source"], errors="ignore")
    y = df["is_loanword"].astype(int) if "is_loanword" in df.columns else None
    return X, y

def output_threshold_metrics(y_true, y_probs):
    print("threshold,precision,recall,f1")
    for threshold in np.arange(0.0, 1.01, 0.01):
        preds = (y_probs >= threshold).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        print(f"{threshold:.2f},{precision:.3f},{recall:.3f},{f1:.3f}")

def find_best_threshold(y_true, y_probs, min_precision=0.0, min_recall=0.0):
    print("[I] Searching for best threshold (loanword class = 1)...")
    best_threshold = 0.5
    best_f1 = 0

    for threshold in np.arange(0.0, 1.01, 0.01):
        preds = (y_probs >= threshold).astype(int)
        precision = precision_score(y_true, preds, pos_label=1, zero_division=0)
        recall = recall_score(y_true, preds, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)

        if precision >= min_precision and recall >= min_recall and f1 > best_f1:
            best_threshold = threshold
            best_f1 = f1

    print(f"[I] Chosen threshold: {best_threshold:.2f} (F1: {best_f1:.4f}) with min precision: {min_precision}")
    return best_threshold, best_f1

def train_model(X_train_raw, y_train, X_tune_raw=None, y_tune=None, classifier_type="lr", threshold=None,
                auto_threshold=False, min_precision=0.0, min_recall=0.0, dump_metrics=False,
                corpus_ngrams=None):
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train_raw)
    scaler = None

    if classifier_type == "lr":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        clf = LogisticRegression(max_iter=1000)
    elif classifier_type == "rf":
        clf = RandomForestClassifier(
            # combined: 300 - OK, 325 - High precision; 350 - a bit worse than
            # 300; 600 - higher pres, 400 - 66 / 58; 900 - higher precision (72/53);
            # for now -> favor higher precision.
            n_estimators=900,
            max_depth=None,
            # Try
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError("Unsupported classifier type.")

    clf.fit(X_train, y_train)

    if X_tune_raw is None or y_tune is None:
        X_tune_raw, y_tune = X_train_raw, y_train

    X_tune = imputer.transform(X_tune_raw)
    if scaler:
        X_tune = scaler.transform(X_tune)

    y_probs = clf.predict_proba(X_tune)[:, 1]

    if dump_metrics:
        output_threshold_metrics(y_tune, y_probs)
        exit(0)

    if auto_threshold:
        threshold, _ = find_best_threshold(y_tune, y_probs, min_precision=min_precision, min_recall=min_recall)
    elif threshold is None:
        threshold = 0.5

    return LoanwordClassifier(clf, threshold, imputer, scaler, corpus_ngrams=corpus_ngrams)

# -- Main script
def main():
    parser = argparse.ArgumentParser(description="Train classifier on word surprisal features.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train_vectors")
    group.add_argument("--train_words")

    parser.add_argument("--tune_vectors")
    parser.add_argument("--tune_words")

    parser.add_argument("--eval_vectors")
    parser.add_argument("--eval_words")
    parser.add_argument("--prob_dir", help="Required if using raw word files")
    parser.add_argument("--classifier", choices=["lr", "rf"], default="lr")
    parser.add_argument("--min_precision", type=float, default=0.0)
    parser.add_argument("--min_recall", type=float, default=0.0)
    parser.add_argument("--model_out", required=True)
    parser.add_argument("--threshold", type=float, help="Manual threshold override")
    parser.add_argument("--auto_threshold", action="store_true", help="Automatically select best threshold")
    parser.add_argument("--dump_threshold_metrics", action="store_true", help="Output metrics across thresholds")
    args = parser.parse_args()

    corpus_ngrams = None
    if args.train_words or args.eval_words or args.tune_words:
        if not args.prob_dir:
            raise ValueError("--prob_dir is required for raw word input")
        corpus_ngrams = load_ngram_surprisal(args.prob_dir)

    if args.train_vectors:
        X_train_raw, y_train = load_data(args.train_vectors)
    else:
        df = pd.read_csv(args.train_words)
        df = vectorize_words(df, corpus_ngrams, FEATURES)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
            df.to_csv(f.name, index=False)
            X_train_raw, y_train = load_data(f.name)

    if args.tune_vectors:
        X_tune_raw, y_tune = load_data(args.tune_vectors)
    elif args.tune_words:
        df = pd.read_csv(args.tune_words)
        df = vectorize_words(df, corpus_ngrams, FEATURES)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
            df.to_csv(f.name, index=False)
            X_tune_raw, y_tune = load_data(f.name)
    else:
        X_tune_raw, y_tune = X_train_raw, y_train

    model = train_model(
        X_train_raw,
        y_train,
        X_tune_raw=X_tune_raw,
        y_tune=y_tune,
        classifier_type=args.classifier,
        threshold=args.threshold,
        auto_threshold=args.auto_threshold,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        dump_metrics=args.dump_threshold_metrics,
        corpus_ngrams=corpus_ngrams  # Inject into model
    )
    joblib.dump(model, args.model_out)
    print(f"[I] Model saved to {args.model_out}")

    if args.eval_vectors:
        X_eval_raw, y_eval = load_data(args.eval_vectors)
    elif args.eval_words:
        df_eval = pd.read_csv(args.eval_words)
        df_eval = vectorize_words(df_eval, corpus_ngrams, FEATURES)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
            df_eval.to_csv(f.name, index=False)
            X_eval_raw, y_eval = load_data(f.name)
    else:
        return

    print("[I] Running evaluation...")
    probs = model.predict_proba(X_eval_raw)
    preds = model.predict(X_eval_raw)
    print(classification_report(y_eval, preds))

if __name__ == "__main__":
    main()
