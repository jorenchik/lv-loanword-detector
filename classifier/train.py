import pandas as pd
import argparse
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from word_vectorizer import main as vectorize_main  # assumes the vectorizer is importable
import tempfile
import numpy as np


def load_data(input_file):
    df = pd.read_csv(input_file)
    X = df.drop(columns=["word", "is_loanword", "source"])
    y = df["is_loanword"].astype(int)
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    return X_imputed, y


def find_best_threshold(y_true, y_probs, min_precision=0.0):
    best_threshold = 0.5
    best_f1 = 0
    for threshold in np.arange(0.0, 1.01, 0.01):
        preds = (y_probs >= threshold).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        if precision >= min_precision and f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1


def train_and_evaluate(X_train, y_train, X_eval, y_eval, classifier_type="lr", min_precision=0.0):
    if classifier_type == "lr":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_eval = scaler.transform(X_eval)
        clf = LogisticRegression(max_iter=1000)
    elif classifier_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError("Unsupported classifier type. Use 'lr' or 'rf'.")

    clf.fit(X_train, y_train)
    y_probs = clf.predict_proba(X_eval)[:, 1]
    best_threshold, best_f1 = find_best_threshold(y_eval, y_probs, min_precision=min_precision)
    y_pred = (y_probs >= best_threshold).astype(int)

    print(f"Best threshold: {best_threshold:.2f} with F1 score: {best_f1:.4f}")
    print(classification_report(y_eval, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classifier on word surprisal features.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train_vectors", help="CSV file with training features")
    parser.add_argument("--eval_vectors", help="CSV file with evaluation features")

    group.add_argument("--train_words", help="CSV file with words for training")
    parser.add_argument("--eval_words", help="CSV file with words for evaluation")
    parser.add_argument("--prob_dir", help="Directory with ngram surprisal files")

    parser.add_argument("--classifier", choices=["lr", "rf"], default="lr", help="Classifier type: 'lr' or 'rf'")
    parser.add_argument("--min_precision", type=float, default=0.0, help="Minimum precision required when tuning threshold")
    args = parser.parse_args()

    if args.train_vectors and args.eval_vectors:
        X_train, y_train = load_data(args.train_vectors)
        X_eval, y_eval = load_data(args.eval_vectors)

    elif args.train_words and args.eval_words and args.prob_dir:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as train_out, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as eval_out:

            vectorize_main(args.train_words, args.prob_dir, train_out.name)
            vectorize_main(args.eval_words, args.prob_dir, eval_out.name)

            X_train, y_train = load_data(train_out.name)
            X_eval, y_eval = load_data(eval_out.name)
    else:
        raise ValueError("You must provide either --train_vectors/--eval_vectors or --train_words/--eval_words with --prob_dir")

    train_and_evaluate(X_train, y_train, X_eval, y_eval, classifier_type=args.classifier, min_precision=args.min_precision)
