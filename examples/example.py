import argparse
import pandas as pd
import joblib

from classifier.train import LoanwordClassifier

def main():
    parser = argparse.ArgumentParser(description="Classification example.")
    parser.add_argument("--model", required=True, help="Path to the .pkl model file")
    args = parser.parse_args()

    # Load the model
    model: LoanwordClassifier = joblib.load(args.model)

    # Example Latvian words
    words = [
        "maize",       # native (bread)
        "televizors",  # clear loanword (television)
        "internets",   # clear loanword (internet)
        "saule",       # native (sun)
        "kafija",      # loanword (coffee)
        "students",    # loanword (student)
        "logs",        # native (window)
        "motocikls"    # clear loanword (motorcycle)
    ]
    df_words = pd.DataFrame({"word": words})

    # Vectorize the input words
    X, _ = model.vectorize_words(df_words)

    # Predict using the model
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Show results
    for word, pred, prob in zip(words, predictions, probabilities):
        print(f"Word: {word} | Prediction: {pred} | Probability: {prob:.4f}")

if __name__ == "__main__":
    main()
