## Description

A utility that parses corpora that is used in training into a homogenous format
for easily using it. What particular corpora is parsed is determined by CLI
option.

It produces either a frequency list or a simple list. There is an option to
avoid duplicates if such behaviour is desired.

| Name            | Key               |  Description                   | Format (option in parsing) | Prob. computed?  |
|-----------------|-------------------|--------------------------------|----------------------------|------------------|
| LAvīzes         | `lv_avizes`       | LA newspaper (1822–1915)       | `vert`                     |        ✓         |
| Senie           | `senie`           | ~1600–1800 year Latv. language | `senie_xml`                |        x         |
| LaVA            | `lava`            | Latv. learner essays           | `lava_csv`                 |        x         |
| Likumi          | `likumi`          | Latv. law documents            | `vert`                     |        x         |
| Disertacijas    | `lv_disertacijas` | Latv. dissertations            | `lv_disertacijas_txt`      |        ✓         |
| Vikipēdija      | `vikipedija`      | Latvian Wikipedia              | `vert`                     |        ✓         |
| Rainis          | `rainis`          | Raiņa darbu korpuss            | `rainis_txt`               |        ✓         |


The main output of this utility are surprise log probabilities of ngram
occurance. Probabilities are stored in separate file for each corpus with a
name: `<key>_ngram_probs.csv`, e.g., `lv_disertacijas_ngram_probs.csv`. 

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

## TODOs

TODO(Jorens): describe the formats with examples.
TODO(Jorens): add source links to the corpora.


## Notes

For now the filtering out non latvian symbol tokens while cleaning is only on the new corpora.


