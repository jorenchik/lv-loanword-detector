import re
import csv
import argparse
from tqdm import tqdm
import subprocess

# LANG_HEADER = re.compile(r"^==[A-Z][A-Za-z ]+==")
LANG_HEADER = re.compile(r"(^|[^=])==([A-Za-z ]+)==")
HEADER = re.compile(r"===([A-Za-z ]+)===")
SECTION_HEADER = re.compile(r"^===")
TITLE_TAG = re.compile(r"<title>(.*?)</title>")

# TITLE_TAG = re.compile(r"<title>(.*?)</title>")
# TEMPLATE_RE = re.compile(r"\{\{([^{}]+)\}\}")
# TITLE_TAG = re.compile(r"<title>")

# Detect common etymological templates

# ETYMOLOGY_MARKER = re.compile(
#     r"\{\{(der|inh|bor|af|suf|prefix|compound|com|blend|root|etyl|unk|cog|ncog)",
#     re.IGNORECASE,
# )

SPECIFIER = re.compile(
    r"\{\{[^}]+\}\}", # root
    re.IGNORECASE,
)

SPECIFIER = re.compile(
    r"\{\{[^}]+\}\}", # root
    re.IGNORECASE,
)

ETYMOLOGY_MARKER = re.compile(
    r"\{\{(der\+?\|lv|inh\+?\|lv|bor\?\|lv)", # root
    re.IGNORECASE,
)

LEXICAL_HEADERS = (
    "Noun",
    "Verb",
    "Adjective",
    "Adverb",
    "Pronoun",
    "Participle",
    "Determiner",
    "Interjection",
    "Proper noun",
    "Preposition",
    "Suffix",
)

def valid_word(word: str, allow_cyrillic: bool = False) -> bool:
    """
    Accepts only proper Latvian-like words:
    - rejects words starting or ending with '-'
    - rejects words containing spaces
    - rejects non-Latvian alphabet characters
    """
    if not word or word.startswith("-") or word.endswith("-") or " " in word:
        return False

    latin_lv = r"A-Za-zĀĒĪŪĶĻŅŠŽāēīūķļņšž\-"
    cyr = r"\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F"
    charset = latin_lv + (cyr if allow_cyrillic else "")

    # word must contain only allowed characters and be at least 2 chars long
    return bool(re.fullmatch(fr"[{charset}]+", word)) and len(word) > 2

# @dataclass
# class WictEntry:
#     line: str
#     word: str
#     etymology_text: str
#
#     def labels():
#         return ["line", "word", "etymology_text"]
#
#     def __list__():
#         return [line, word, etymology_text]

def extract_latvian_data(
    filename: str,
    csv_out: str,
    allow_cyrillic: bool = False,
    only_latvian: bool = False,
    range_start: int = None,
    range_end: int = None,
    total_lines: int = None,
):

    word_content = None
    buffer = []
    debug_buffer = []

    inside_lv = False
    inside_etym = False
    lv_line = None
    prev_title = None
    word = None
    title_line = None
    lang = None

    if not total_lines:
        print("Counting lines int the input file...")
        line_count = 0
        with open(filename, "r") as f:
            for _ in f:
                line_count += 1
        total_lines = line_count

    with (
        open(filename, "r", encoding="utf-8", errors="ignore") as f,
        open(csv_out, "w", encoding="utf-8", newline="") as out
    ):
        writer = csv.writer(out)
        writer.writerow(["line", "language", "word", "etymology_text"])

        for line_num, line in tqdm(enumerate(f, start=1), total=total_lines):

            if range_start and line_num < range_start:
                continue

            if range_end and line_num > range_end:
                break

            # if line_num > 9389000 and "English" in line:
            #     __import__('pdb').set_trace()

            if LANG_HEADER.search(line) or TITLE_TAG.search(line):

                if inside_lv or (not only_latvian):
                    normalized_text = re.sub(r"\s+", " ", " ".join(buffer)).strip()

                    if re.search(SPECIFIER, normalized_text):
                        specifiers = re.findall(SPECIFIER, normalized_text)
                        parts = [x.strip('{}').split('|') for x in specifiers]

                    if valid_word(word) and normalized_text: 
                        # if ETYMOLOGY_MARKER.search(etymology):
                        writer.writerow([line_num, lang, word, normalized_text])

                if TITLE_TAG.search(line):
                    prev_title = line_num
                    title_line = line
                    res = TITLE_TAG.search(line)
                    word = res.group(1)

                if LANG_HEADER.search(line):
                    res = LANG_HEADER.search(line)
                    lang = res.group(2)

                    if lang == "Latvian":
                        inside_lv = True
                    else:
                        inside_lv = False 

                    lv_line = line_num
                    word_content = None

                buffer.clear()
                debug_buffer.clear()
                inside_etym = False 

                continue

            res = HEADER.search(line)
            if res:

                name = res.group(1)

                if inside_lv or (not only_latvian):
                    if name == "Etymology":
                        inside_etym = True
                        continue
                    else:
                        inside_etym = False 

            if inside_etym:
                buffer.append(line)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("xml")
    parser.add_argument("--out",            default="latvian_extracted.csv")
    parser.add_argument("--allow-cyrillic", action="store_true")
    parser.add_argument("--only-latvian",   action="store_true")
    parser.add_argument("--start",    type=int, default=None)
    parser.add_argument("--end",      type=int, default=None)
    parser.add_argument("--total-lines",      type=int, default=None)

    args = parser.parse_args()
    extract_latvian_data(
        args.xml,
        args.out,
        allow_cyrillic=args.allow_cyrillic,
        only_latvian=args.only_latvian,
        range_start=args.start,
        range_end=args.end,
        total_lines=args.total_lines,
    )

if __name__ == "__main__":
    main()
