# Data Directory

This directory contains raw, intermediate, and processed data used for
training, evaluating, and analyzing Latvian loanword detection models.

Available at:
[lv-loanword-detector-data](https://nextcloud.jorensh.xyz/s/lv-loanword-detector-data).

## Structure

```
data/
├── cleaned_corpora/
│   └── base/             
│   └── lv_chars/          
│
├── extracted_dict_entries/     # Raw headword extraction from dictionary PDF
│   ├── v1_entry_raw_data.csv
│   ├── v2_entry_raw_data.csv
│   └── v3_entry_raw_data.csv
│
├── labeled_data/
│   ├── dict/
│   ├── dict_foreign/
│   ├── dict_man/             
│   └── dict_man_foreign/    
│
├── manual_label/                 # Auxiliary labeling data
│   ├── additional_native.csv
│   ├── foreign_lang_words.csv
│   └── manual_modern.csv
│
├── manual_label_source/
│   └── latviskoti_vardi_500_bez_indoeiro_piesu.csv
│
├── ngram_probabilities/
│   ├── base_ngram_probs/   
│   └── cleaned_ngram_probs/ 
│
└── tokenized_corpora/        
    ├── lava_raw_tokens.txt
    ├── lv_avizes_raw_tokens.txt
    ├── lv_disertacijas_raw_tokens.txt
    ├── rainis_raw_tokens.txt
    └── vikipedija_raw_tokens.txt
```

## Key Inputs to Current Model

- `cleaned_corpora/base/`: Final tokenized inputs used to compute n-gram features.
- `ngram_probabilities/base_ngram_probs/`: Character n-gram surprisals.
- `labeled_data/dict_man/` and `dict_man_foreign/`: Primary training labels.
