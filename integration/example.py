import pandas as pd
import joblib

from classifier.train import LoanwordClassifier

# Load the model.
model_path = "pretrained_models/rf_v0_1_0.pkl" 
model: LoanwordClassifier = joblib.load(model_path)

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
