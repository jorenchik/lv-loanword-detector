
# Classifier module 

Module that involves scripts to produce, test and use the models.

## Project Structure

```
├── classify.py        # Run classification using a trained model
├── train.py           # Train classifier using features or raw word lists
├── word_vectorizer.py # Extract surprisal and length-based features
├── model.py           # Classifier wrapper: includes model, preprocessing, and corpus config
```

## Training

Train a loanword classifier using precomputed features or raw words:

### Option A: From features

```
python -m classifier.train \
    --train_vectors word_vectors.csv \
    --model_out model.pkl \
    --classifier rf \
    --auto_threshold \
    --min_precision 0.75
```

### Option B: From raw words

```
python -m classifier.train \
    --train_words words.csv \
    --prob_dir path/to/ngram_probs \
    --model_out model.pkl \
    --classifier lr
```

### Arguments

```
Other arguments:
    --tune_vectors, --tune_words: For tuning threshold separately from training data.
    --eval_vectors, --eval_words: For post-training evaluation.
    --threshold: Manually set classification threshold (overrides auto).
    --dump_threshold_metrics: Print F1/precision/recall vs. threshold.
```

## Classification

Classify new words with a trained model:

```
python -m classifier.classify \
    --vector_file word_vectors.csv \
    --model model.pkl \
    --output_file predictions.csv
```

Or classify raw words directly (feature extraction is done on the fly):

```
python -m classifier.classify \
    --word_file words.csv \
    --model model.pkl \
    --output_file predictions.csv
```

Interactive mode:

```
python classify.py --interactive --model model.pkl
```

```
Optional:
    --filter_source: Limit evaluation to a single source (if column exists).
    --threshold: Override model’s default decision threshold.
```

## Model Format

Trained models are serialized using joblib and wrapped in a custom
LoanwordClassifier, which includes:

- The scikit-learn classifier.
- F1 score optimal threshold value.
- Vectorizer,
- Corpus aggregated n-gram probability data.

## Vectorizer 

Compute word-level features such as n-gram surprisals and word length.
Vectorizer can be called as a standalone CLI tool.

```
python -m classifier.word_vectorizer \
    --word_file path/to/words.csv \
    --prob_dir path/to/ngram_probs/ \
    --output_file word_vectors.csv
```

```
Inputs:
    --word_file: CSV with at least a word column.
    --prob_dir: Directory with precomputed *_ngram_probs.csv files (one per corpus).
    --output_file: Where to save the feature-enhanced output.
```

