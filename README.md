
# Latvian loanword detector

A modular system for detecting potential loanwords in Latvian text using
linguistic features and machine learning. Includes CLI tools, corpus processing
pipelines, a GUI interface, and manually curated data. Built for research and
prototyping, not production deployment.

## Features

- Dictionary processing & labeling: `etymological_dict/`.
- Corpus parsing & n-gram probability modeling: `corpora_extraction/`.
- Classifier training & usage: `classifier/`.
- Manual annotation tools: `manual_collection/`.
- Interactive GUI interface: `gui_integration/`.
- Shell and integration scripts: `scripts/`, `shell_scripts/`.

## Installation

The project uses `pyproject.toml` and requires Python 3.8+.

```bash
https://github.com/jorenchik/lv-loanword-detector
cd lv-loanword-detector
pip install .
```

Some dependencies are optional:

- `stanza` — for corpus parsing (`pip install stanza` or `pip install .[stanza]`)
- `tkinter` — GUI support; install via system package manager:
  - Debian/Ubuntu: `sudo apt install python3-tk`

## Usage

Refer to READMEs of modules.

## Project Layout

```
classifier/          # Model training, classification CLI, feature engineering
corpora_extraction/  # Corpus parsing, cleaning, n-gram calculation
etymological_dict/   # Dictionary parsing, labeling heuristics
manual_collection/   # Manually prepared word lists and datasets
integration/         # pyproject config logic, shared glue
scripts/             # Model runner scripts, exports
shell_scripts/       # Bash utilities (e.g. packaging, fetch)
gui_integration/     # Tkinter GUI for word-by-word inspection
docs/                # Removed / legacy
```

## Data

The data used to train and evaluate models is hosted separately (~522MB).

See `data/README.md` for layout and descriptions.

## CLI Entry Points

Installed with package:

```bash
lv-loanword-classify   # Run classification
lv-loanword-gui        # Launch GUI
lv-loanword-download   # Download pretrained models
lv-loanword-example    # Run a sample integration script
```
