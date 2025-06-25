
# GUI Integration

Graphical interface for interactive evaluation of Latvian loanword detection
models. Provides a text editor with real-time highlighting based on model
predictions.

## Project Structure

```
├── main.py       # Main GUI logic and event loop
├── samples.py    # Example texts for demonstration
├── utils.py      # Logging and color utility functions
```

## Overview

The GUI offers a text area to input Latvian text. Each word is evaluated using
a selected model and is visually highlighted based on its predicted probability
of being a loanword.

### Features

- Supports multiple pre-trained models (LogReg, RandomForest).
- Adjustable classification threshold via slider.
- Real-time prediction with tooltip feedback.
- Optional full highlighting mode or underline-only.
- Preloaded sample texts.

## Launching the App

```bash
python -m gui_integration.main
```

Note: Linux users _may_ need to install Tkinter separately:

```bash
sudo apt-get install python3-tk
```

## Model Loading

Models are loaded from the following paths:

- `gui_integration/packaged_models/`.
- `~/.lv_loanword_detection/pretrained_models/`.

Expected model filenames:

- `rf_v0_2_1.pkl`.
- `lr_v0_2_1.pkl`.

## Text Tagging and Highlighting

- Words are tokenized using a regex-based word pattern.
- Probabilities are fetched asynchronously and cached per input state.
- Colors are assigned using a probability-to-color mapping (`green → red`).
- Hover tooltips show individual word scores.

## Parameters Panel

- Model threshold: Adjusts highlighting sensitivity.
- Model type: Chooses among available models.
- Highlight full output: Toggles continuous coloring.
- Show tokenization: Debug view for token boundaries.
- Reset parameters: Reverts to model defaults.

## Sample Texts

Defined in `samples.py`. Three examples are available:

- "Sample LV 1".
- "Sample LV 2".
- "Valus Ta'aurc" (English test case).

These are required for GUI logic, model execution, and visual debugging.
