import pandas as pd
import re
import argparse
import os
from sklearn.model_selection import train_test_split

# === ARGUMENT PARSING ===
parser = argparse.ArgumentParser(description="Label Latvian dictionary entries as loanwords or not.")
parser.add_argument('input', nargs='+', help='Path(s) to input CSV file(s).')
parser.add_argument('--debug', action='store_true', help='Output all columns instead of a minimal set.')
parser.add_argument('--no-ocr', action='store_true', help='Disable OCR correction.')
parser.add_argument('--no-multi-ref', action='store_true', help='Disable multi-reference resolution.')
parser.add_argument('--make-full', action='store_true', help='Only merge inputs and write full CSV, no processing.')
parser.add_argument('--output', type=str, default='entries_labeled.csv', help='Base output file name.')
parser.add_argument('--source', type=str, default='etym_dict', help='Source label to attach to each row.')
parser.add_argument('--output-dir', type=str, default='.', help='Directory to save output files.')
parser.add_argument('--split', choices=['none', 'train_test', 'train_dev_test'], default='none', help='Split dataset into train/test or train/dev/test.')
parser.add_argument('--test-size', type=float, default=0.15, help='Proportion of test data.')
parser.add_argument('--dev-size', type=float, default=0.15, help='Proportion of dev data (only used in train_dev_test).')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
args = parser.parse_args()

USE_OCR_CORRECTION = not args.no_ocr
USE_MULTI_REF_RESOLUTION = not args.no_multi_ref
DEBUG_OUTPUT = args.debug
output_file = args.output
source_label = args.source
output_dir = args.output_dir
split_type = args.split

# === UTILS ===
def save_df(df_subset, suffix):
    filename = os.path.splitext(os.path.basename(output_file))[0] + f'_{suffix}.csv'
    path = os.path.join(output_dir, filename)
    df_subset.to_csv(path, index=False)
    print(f"[I] Saved {suffix} set to {path}")

# === MAIN ===
df_list = [pd.read_csv(path) for path in args.input]
df = pd.concat(df_list, ignore_index=True)

if args.make_full:
    os.makedirs(output_dir, exist_ok=True)
    save_df(df, 'full')
    exit(0)

# === Handle already-labeled format ===
if 'is_loanword' in df.columns and 'text' not in df.columns:
    os.makedirs(output_dir, exist_ok=True)
    df = df[~df['is_loanword'].isna()].copy()
    if split_type == 'none':
        save_df(df[['word', 'is_loanword', 'source']], 'full')
    elif split_type == 'train_test':
        train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=df['is_loanword'])
        save_df(train_df[['word', 'is_loanword', 'source']], 'train')
        save_df(test_df[['word', 'is_loanword', 'source']], 'test')
    elif split_type == 'train_dev_test':
        temp_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=df['is_loanword'])
        dev_ratio = args.dev_size / (1 - args.test_size)
        train_df, dev_df = train_test_split(temp_df, test_size=dev_ratio, random_state=args.seed, stratify=temp_df['is_loanword'])
        save_df(train_df[['word', 'is_loanword', 'source']], 'train')
        save_df(dev_df[['word', 'is_loanword', 'source']], 'dev')
        save_df(test_df[['word', 'is_loanword', 'source']], 'test')
    exit(0)

# === CUES for labeling ===
loanword_cues = ['aizg.', 'aizgūts', 'no v.', 'no lat.', 'no fr.', 'no gr.', 'no it.', 'no d.',
                 'no h.', 'no bv.', 'no vlv.', 'no vv.', 'no vav.', 'no jsļ.', 'no norv.', 'no zv.']
non_loanword_roots = ['ide.', 'pirmside.', 'b.', 'b-sl.', 'lš.', 'pr.', 'sl.', 'skr.', 's.', 'ssl.', 's-u.']
derivation_cues = ['atv.', 'pried.', 'pied.']

ocr_corrections = {
    'laitit': 'laitīt',
}

reference_pattern = re.compile(r'sk\.?\??\s*(\w+)', re.IGNORECASE)

def label_entry(text):
    if pd.isnull(text):
        return pd.NA, 'Missing text'

    text_lower = text.lower()
    tokens = re.findall(r'\b\w+\b', text_lower)

    if any(cue in text_lower for cue in loanword_cues):
        return 1, 'Explicit loanword marker'
    elif re.search(r'\bide\.\s*\*?\(?[a-z\-]+\)?', text_lower):
        return 0, 'Indo-European proto-root'
    elif any(cue.rstrip('.') in tokens for cue in non_loanword_roots):
        return 0, 'Proto/Baltic/Slavic roots'
    elif any(cue in text_lower for cue in derivation_cues):
        return 0, 'Latvian derivation'
    elif any(lang in text_lower for lang in ['lš.', 'la.', 'pr.']) and not any(cue in text_lower for cue in loanword_cues):
        return 0, 'Baltic/Latvian derivation'
    else:
        return pd.NA, 'No clear indicator'

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
                if ref_label in [0, 1]:
                    row['is_loanword'] = ref_label
                    row['explanation'] = 'Derived from a reference'
                    row['flag_manual'] = 0
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
                if ref_label in [0, 1]:
                    row['is_loanword'] = ref_label
                    row['explanation'] = 'Derived from a reference'
                    row['flag_manual'] = 0
                    ref_index = headword_to_index.get(ref_word)
                    row['reference_row'] = int(ref_index) + 1 if ref_index is not None else pd.NA
    return row

# === LABELING PROCESS ===
df[['is_loanword', 'explanation']] = df['text'].apply(lambda txt: pd.Series(label_entry(txt)))
df['flag_manual'] = df['is_loanword'].isna().astype(int)
df['reference_row'] = pd.NA
df['source'] = source_label

headword_map = df.set_index('headword').to_dict('index')
headword_to_index = {headword: i for i, headword in enumerate(df['headword'])}

df = df.apply(try_reference_label, axis=1)

# move 'text' column to end
txt_col = df.pop('text')
df.insert(len(df.columns), 'text', txt_col)

# === OUTPUT ===
os.makedirs(output_dir, exist_ok=True)

if DEBUG_OUTPUT:
    save_df(df, 'full')
else:
    df.rename(columns={'headword': 'word'}, inplace=True)
    df = df[~df['is_loanword'].isna()].copy()

    if split_type == 'none':
        save_df(df[['word', 'is_loanword', 'source']], 'full')
    elif split_type == 'train_test':
        train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=df['is_loanword'])
        save_df(train_df[['word', 'is_loanword', 'source']], 'train')
        save_df(test_df[['word', 'is_loanword', 'source']], 'test')
    elif split_type == 'train_dev_test':
        temp_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=df['is_loanword'])
        dev_ratio = args.dev_size / (1 - args.test_size)
        train_df, dev_df = train_test_split(temp_df, test_size=dev_ratio, random_state=args.seed, stratify=temp_df['is_loanword'])
        save_df(train_df[['word', 'is_loanword', 'source']], 'train')
        save_df(dev_df[['word', 'is_loanword', 'source']], 'dev')
        save_df(test_df[['word', 'is_loanword', 'source']], 'test')

if not DEBUG_OUTPUT:
    print("[I] Use --debug to see unlabeled rows, they are filtered by default")

# === SUMMARY ===
print("\nLoanword Classification Summary:")
print(df['is_loanword'].value_counts(dropna=False))
print("\nTop Explanations:")
print(df['explanation'].value_counts().head(10))
