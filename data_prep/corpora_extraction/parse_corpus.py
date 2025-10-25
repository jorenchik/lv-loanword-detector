import stanza
from lxml import etree as ET
import argparse
import os
import gzip
from tqdm import tqdm
import re
import csv

# --- Step 1: Argument parsing ---
parser = argparse.ArgumentParser(description="Parse Latvian TEI XML, VERT, TXT, or CSV format and tokenize text.")
parser.add_argument("input_file", help="Path to the input file (.xml, .vert, .vert.gz, .txt, .txt.gz, .csv, .csv.gz)")
parser.add_argument("--output_file", default="parsed_latvian_tokens.txt", help="Path to the output file")
parser.add_argument("--use_gpu", action="store_true", help="Enable GPU acceleration for Stanza")
parser.add_argument("--format", choices=["lv_disertacijas_txt", "rainis_txt", "lava_csv", "senie_xml", "vert"],
                    required=True,
                    help="Specify the input format explicitly")
args = parser.parse_args()

# --- Step 2: Load Stanza Latvian model ---
if not os.path.exists(os.path.expanduser("~/.stanza_resources/lv")):
    stanza.download('lv')
nlp = stanza.Pipeline(lang='lv', processors='tokenize', use_gpu=args.use_gpu)

# --- Step 3: Determine file type and extract text ---
input_path = args.input_file
latvian_texts = []

if args.format == "vert":
    open_func = gzip.open if input_path.endswith(".gz") else open
    with open_func(input_path, mode='rt', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    with open_func(input_path, mode='rt', encoding='utf-8') as f, tqdm(total=total_lines, desc="Reading VERT file") as pbar:
        for line in f:
            if line.strip() and not line.startswith('<'):
                parts = line.strip().split('\t')
                if parts:
                    latvian_texts.append(parts[0])
            pbar.update(1)

elif args.format == "lv_disertacijas_txt":
    open_func = gzip.open if input_path.endswith(".gz") else open
    with open_func(input_path, mode="rt", encoding="utf-8") as f:
        lines = f.readlines()
    inside_doc = False
    for line in tqdm(lines, desc="Parsing TXT lv_disertacijas_txt"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("<doc"):
            inside_doc = True
            continue
        elif line.startswith("</doc>"):
            inside_doc = False
            continue
        elif line.startswith("<section") or line.startswith("</section>"):
            continue
        elif inside_doc and not line.startswith("<"):
            latvian_texts.append(line)

elif args.format == "rainis_txt":
    open_func = gzip.open if input_path.endswith(".gz") else open
    with open_func(input_path, mode="rt", encoding="utf-8") as f:
        lines = f.readlines()
    for line in tqdm(lines, desc="Parsing TXT rainis_txt"):
        line = line.strip()
        if line:
            latvian_texts.append(line)

elif args.format == "lava_csv":
    open_func = gzip.open if input_path.endswith(".gz") else open
    with open_func(input_path, mode='rt', encoding='utf-8', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        current_essay_id = None
        current_tokens = []

        for row in tqdm(reader, desc="Reading CSV lava_csv"):
            token = row['corrected_token'] if row['corrected_token'] else row['original_token']
            essay_id = row['essay_id']

            if essay_id != current_essay_id:
                if current_tokens:
                    latvian_texts.append(" ".join(current_tokens))
                current_tokens = [token]
                current_essay_id = essay_id
            else:
                current_tokens.append(token)

        if current_tokens:
            latvian_texts.append(" ".join(current_tokens))

elif args.format == "senie_xml":
    parser = ET.XMLParser(recover=True)
    context = ET.iterparse(input_path, events=("end",), tag="{http://www.tei-c.org/ns/1.0}div", recover=True)
    for event, elem in tqdm(context, desc="Parsing XML senie_xml"):
        if elem.get("type") == "Language" and elem.get("lang") == "Latvian" and elem.text:
            latvian_texts.append(elem.text.strip())
        elem.clear()

else:
    raise ValueError("Unsupported format.")

# --- Step 4: Tokenize and write in chunks ---
BATCH_SIZE = 1000

with open(args.output_file, "w", encoding="utf-8") as out_f:
    for i in tqdm(range(0, len(latvian_texts), BATCH_SIZE), desc="Tokenizing"):
        batch_text = ' '.join(latvian_texts[i:i + BATCH_SIZE])
        doc = nlp(batch_text)
        for sentence in doc.sentences:
            for word in sentence.words:
                out_f.write(f"<{word.text}>\n")

print(f"Tokenized words saved to {args.output_file}")
