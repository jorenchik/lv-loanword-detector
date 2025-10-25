# Dictionary Processing Tools

Scripts for parsing, labeling, and validating headword entries from Latvian
etymological sources.

## Project Structure

```
├── extract_entries.py # Extract headwords and text entries from scanned PDF
├── process_entries.py # Analyze ordering and detect disordered pages
├── label_data.py      # Heuristically label entries as loanwords or not
```

## 1. Entry Extraction

Extract headwords and entry text from a scanned dictionary PDF. Designed
specifically for [Latviešu etimoloģijas vārdnīca](Latviešu etimoloģijas
vārdnīca).

- Loads `data/Latviesu etimologijas vardnica (2001).pdf`
- Extracts headwords based on text indentation heuristics
- Skips pages with no headwords
- Outputs:
  - `entry_raw_data.csv`: raw entries
  - `temp_results.txt`: readable plaintext preview

PDF path and constants are hardcoded. Adjust as needed.

### Example:

```bash
python -m etymological_dict.extract_entries
```

## 2. Entry Labeling

Label entries as loanwords (1) or native (0) using cue-based heuristics.

```bash
python -m etymological_dict.label_data input_file.csv --output labeled.csv
```

### Options:

```
--debug           Output full data with explanations and unlabeled rows
--make-full       Skip labeling; only merge and save input files
--split           Split into train/test/dev sets
--no-ocr          Disable defined OCR corrections
--no-multi-ref    Disable reference-based labeling
--source          Source label to attach to each row (default: etym_dict)
--output-dir      Directory to save output files
```

Supports multiple input files (merged before processing).

### Output columns (non-debug):

* `word`.
* `is_loanword` (0 or 1).
* `source`.

With `--debug`, also includes:

- `explanation`: heuristic that triggered label.
- `flag_manual`: 1 if unresolved.
- `reference_row`: row index of referenced entry (if resolved).
- `text`: full entry text.

### Example:

```bash
python -m etymological_dict.label_data entry_raw_data.csv --debug --split train_dev_test --output-dir labeled_data
```

## Labeling heuristic 

- Recognizes common etymological markers like `aizg.`, `no fr.`, etc.
- Uses proto-root and derivation cues to exclude native words.
- Optionally resolves references (`sk. vārds`) to infer labels.
- Applies minimal OCR corrections for noisy entries.
