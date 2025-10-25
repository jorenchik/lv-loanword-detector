# Corpus Processing Tools

Scripts for extracting, cleaning, and analyzing Latvian text corpora to compute character n-gram statistics.

## Description

A utility that parses corpora that is used in training into a homogenous format for easily using it. What particular corpora is parsed is determined by CLI option.

It produces either a frequency list or a simple list. There is an option to avoid duplicates if such behaviour is desired.

| Name         | Key               | Description                     | Format (option in parsing) | Used in current. model? |
| ------------ | ----------------- | ------------------------------- | -------------------------- | ----------------------- |
| LAvīzes      | `lv_avizes`       | LA newspaper (1822–1915)        | `vert`                     | ✓                       |
| Senie        | `senie`           | \~1600–1800 year Latv. language | `senie_xml`                | x                       |
| LaVA         | `lava`            | Latv. learner essays            | `lava_csv`                 | ✓                       |
| Likumi       | `likumi`          | Latv. law documents             | `vert`                     | x                       |
| Disertacijas | `lv_disertacijas` | Latv. dissertations             | `lv_disertacijas_txt`      | ✓                       |
| Vikipēdija   | `vikipedija`      | Latvian Wikipedia               | `vert`                     | ✓                       |
| Rainis       | `rainis`          | Raiņa darbu korpuss             | `rainis_txt`               | ✓                       |

The main output of this utility are surprise log probabilities of ngram occurance. Probabilities are stored in separate file for each corpus with a name: `<key>_ngram_probs.csv`, e.g., `lv_disertacijas_ngram_probs.csv`.

## Requirements

Install pytorch.

CPU version:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

GPU version (CUDA):

```bash
pip install torch
```

## Project Structure

```
├── parse_corpus.py   # Parse various formats into tokenized text
├── clean_tokens.py   # Normalize and filter raw token sequences
├── compute_ngrams.py # Compute n-gram log-probabilities or surprisals
```

## 1. Corpus Parsing

Extract Latvian text from structured corpora and tokenize using Stanza.

```bash
python parse_corpus.py input.vert.gz --format vert --output_file parsed.txt
```

### Supported formats (via `--format`):

- `vert`: Verticalized corpora (.vert or .vert.gz)
- `lv_disertacijas_txt`: Custom `.txt` corpus with <doc> markers
- `rainis_txt`: Plaintext file per line
- `lava_csv`: Structured CSV with essay tokens
- `senie_xml`: TEI XML format (e.g. ancient corpora)

### Options:

- `--use_gpu`: Use GPU acceleration for Stanza (optional)
- `--output_file`: Where to save `<token>` format output (default: `parsed_latvian_tokens.txt`)

## 2. Token Cleaning

Normalize, lowercase, and filter invalid or shorthand tokens.

```bash
python -m corpora_extraction.clean_tokens parsed.txt --output cleaned.txt
```

### Options:

- `--output`: Path for cleaned token output
- `--skip-valid-check`: Skip alphabetic validation (only applies basic filtering)

Tokens are expected in `<token>` format, one per line.

## 3. N-gram Computation

Compute character-level n-gram statistics (log-probabilities or surprisals).

```bash
python -m corpora_extraction.compute_ngrams cleaned.txt --ngram 2 3 --mode full suffix --surprise --output char_ngrams.csv
```

### Options:

- `--ngram`: One or more n-gram sizes (e.g. 2 3 4)
- `--mode`: One or more modes: `full`, `prefix`, `suffix`
- `--surprise`: Output negative log-probabilities (surprisal)
- `--top_k`: Number of top n-grams to keep (default: 30,000)
- `--top_k_map`: Fine-tuned override per n/mode (e.g. `3:1000`, `4,suffix:500`)
- `--output`: Output CSV file path

### Output CSV columns:

- `ngram`: The n-gram
- `log_probability` or `surprisal`: The score
- `length`: Length of n-gram
- `is_prefix`, `is_suffix`: Boolean flags
- `ngram_size`: The n value
- `mode`: Computation mode

## Example Workflow

```bash
python -m corpora_extraction.parse_corpus rainis.txt --format rainis_txt --output_file parsed.txt
python -m corpora_extraction.clean_tokens parsed.txt --output cleaned.txt
python -m corpora_extraction.compute_ngrams cleaned.txt --ngram 3 --mode full suffix --surprise --output rainis_3grams.csv
```

Generates tokenized, cleaned text and character-level n-gram surprisal stats.
