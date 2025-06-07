
from classifier.word_vectorizer import vectorize_words, FEATURES

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


