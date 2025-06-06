import stanza
from lxml import etree as ET
import argparse
import os
import gzip
from tqdm import tqdm
import re

# --- Step 1: Argument parsing ---
parser = argparse.ArgumentParser(description="Parse Latvian TEI XML, VERT, or custom TXT format and tokenize text.")
parser.add_argument("input_file", help="Path to the input file (.xml, .vert, .vert.gz, .txt, or .txt.gz)")
parser.add_argument("--output_file", default="parsed_latvian_tokens.txt", help="Path to the output file")
parser.add_argument("--use_gpu", action="store_true", help="Enable GPU acceleration for Stanza")
parser.add_argument("--txt_format", choices=["format_1"], help="Specify custom TXT input format, e.g., 'format_1'")
args = parser.parse_args()

# --- Step 2: Load Stanza Latvian model ---
if not os.path.exists(os.path.expanduser("~/.stanza_resources/lv")):
    stanza.download('lv')
nlp = stanza.Pipeline(lang='lv', processors='tokenize', use_gpu=args.use_gpu)

# --- Step 3: Determine file type and extract text ---
input_path = args.input_file
latvian_texts = []

if input_path.endswith(".xml"):
    parser = ET.XMLParser(recover=True)
    context = ET.iterparse(input_path, events=("end",), tag="{http://www.tei-c.org/ns/1.0}div", recover=True)
    for event, elem in tqdm(context, desc="Parsing XML"):
        if elem.get("type") == "Language" and elem.get("lang") == "Latvian" and elem.text:
            latvian_texts.append(elem.text.strip())
        elem.clear()

elif input_path.endswith(".vert") or input_path.endswith(".vert.gz"):
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

elif input_path.endswith(".txt") or input_path.endswith(".txt.gz"):
    if args.txt_format == "format_1":
        open_func = gzip.open if input_path.endswith(".gz") else open
        with open_func(input_path, mode="rt", encoding="utf-8") as f:
            lines = f.readlines()
        for line in tqdm(lines, desc="Parsing TXT format_1"):
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
    else:
        raise ValueError("TXT file given but --txt_format not specified or unsupported format.")

else:
    raise ValueError("Unsupported file format. Use .xml, .vert, .vert.gz, .txt, or .txt.gz")

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
