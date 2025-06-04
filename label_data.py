import pandas as pd
import re

# Define key abbreviation sets
loanword_cues = ['aizg.', 'aizgūts', 'no v.', 'no lat.', 'no fr.', 'no gr.', 'no it.', 'no d.',
                 'no h.', 'no bv.', 'no vlv.', 'no vv.', 'no vav.', 'no jsļ.', 'no norv.', 'no zv.']
non_loanword_roots = ['ide.', 'pirmside.', 'b.', 'b-sl.', 'lš.', 'pr.', 'sl.', 'skr.', 's.', 'ssl.', 's-u.']
derivation_cues = ['atv.', 'pried.', 'pied.']

# OCR correction toggle and dictionary
USE_OCR_CORRECTION = True
ocr_corrections = {
    'laitit': 'laitīt',
}

# Toggle for extended reference analysis.
USE_MULTI_REF_RESOLUTION = True

def label_entry(text):
    if pd.isnull(text):
        return pd.NA, 'Missing text'

    text_lower = text.lower()
    tokens = re.findall(r'\b\w+\b', text_lower)

    if any(cue in text_lower for cue in loanword_cues):
        return True, 'Explicit loanword marker'
    elif re.search(r'\bide\.\s*\*?\(?[a-z\-]+\)?', text_lower):
        return False, 'Indo-European proto-root'
    elif any(cue.rstrip('.') in tokens for cue in non_loanword_roots):
        return False, 'Proto/Baltic/Slavic roots'
    elif any(cue in text_lower for cue in derivation_cues):
        return False, 'Latvian derivation'
    elif any(lang in text_lower for lang in ['lš.', 'la.', 'pr.']) and not any(cue in text_lower for cue in loanword_cues):
        return False, 'Baltic/Latvian derivation'
    else:
        return pd.NA, 'No clear indicator'

# Load data.
input_file = 'v3_entry_raw_data.csv'
df = pd.read_csv(input_file)

# Apply initial labeling.
df[['is_loanword', 'explanation']] = df['text'].apply(lambda txt: pd.Series(label_entry(txt)))
df['flag_manual'] = df['is_loanword'].isna()
df['reference_row'] = pd.NA

# Build word -> index map.
headword_map = df.set_index('headword').to_dict('index')
headword_to_index = {headword: i for i, headword in enumerate(df['headword'])}

# Function to try resolving reference.
reference_pattern = re.compile(r'sk\.?\s*(\w+)', re.IGNORECASE)

def try_reference_label(row):
    if not pd.isna(row['is_loanword']) or pd.isnull(row['text']):
        return row

    if USE_MULTI_REF_RESOLUTION:
        matches = reference_pattern.findall(row['text'])
        for ref_word in matches:
            if USE_OCR_CORRECTION:
                ref_word = ocr_corrections.get(ref_word, ref_word)
            ref_entry = headword_map.get(ref_word)
            if ref_entry:
                ref_label = ref_entry.get('is_loanword')
                if ref_label in [True, False]:
                    row['is_loanword'] = ref_label
                    row['explanation'] = 'Derived from a reference'
                    row['flag_manual'] = False
                    ref_index = headword_to_index.get(ref_word)
                    row['reference_row'] = int(ref_index) + 1 if ref_index is not None else pd.NA
                    break
    else:
        match = reference_pattern.search(row['text'])
        if match:
            ref_word = match.group(1)
            if USE_OCR_CORRECTION:
                ref_word = ocr_corrections.get(ref_word, ref_word)
            ref_entry = headword_map.get(ref_word)
            if ref_entry:
                ref_label = ref_entry.get('is_loanword')
                if ref_label in [True, False]:
                    row['is_loanword'] = ref_label
                    row['explanation'] = 'Derived from a reference'
                    row['flag_manual'] = False
                    ref_index = headword_to_index.get(ref_word)
                    row['reference_row'] = int(ref_index) + 1 if ref_index is not None else pd.NA
    return row

# Apply reference resolution.
df = df.apply(try_reference_label, axis=1)

# Reorder columns to put is_loanword and metadata before the text column.
txt_col = df.pop('text')
df.insert(len(df.columns), 'text', txt_col)

# Save output.
output_file = 'entries_labeled.csv'
df.to_csv(output_file, index=False)

# Print summary report.
print(f"Labeled data saved to {output_file}\n")
print("Loanword Classification Summary:")
print(df['is_loanword'].value_counts(dropna=False))
print("\nTop Explanations:")
print(df['explanation'].value_counts().head(10))
