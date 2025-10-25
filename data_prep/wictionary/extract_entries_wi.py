import re
import csv
import argparse
from tqdm import tqdm

# LANG_HEADER = re.compile(r"^==[A-Z][A-Za-z ]+==")
LANG_HEADER = re.compile(r"(^|[^=])==([A-Za-z ]+)==")
HEADER = re.compile(r"===([A-Za-z ]+)===")
SECTION_HEADER = re.compile(r"^===")
TITLE_TAG = re.compile(r"<title>(.*?)</title>")
# TITLE_TAG = re.compile(r"<title>(.*?)</title>")
# TEMPLATE_RE = re.compile(r"\{\{([^{}]+)\}\}")
# TITLE_TAG = re.compile(r"<title>")

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
    latin_lv = r"A-Za-zĀĒĪŪĶĻŅŠŽāēīūķļņšž\-"
    cyr = r"\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F"
    charset = latin_lv + (cyr if allow_cyrillic else "")
    return re.fullmatch(fr"[{charset}]+", word) and len(word) > 2


def extract_latvian_data(filename: str, csv_out: str, allow_cyrillic: bool = False):

    word_content = None
    buffer = []
    debug_buffer = []

    inside_lv = False
    inside_etym = False
    lv_line = None
    prev_title = None
    word = None

    with (
        open(filename, "r", encoding="utf-8", errors="ignore") as f,
        open(csv_out, "w", encoding="utf-8", newline="") as out
    ):
        writer = csv.writer(out)
        writer.writerow(["line", "word", "text"])

        for line_num, line in tqdm(enumerate(f, start=1)):

            if TITLE_TAG.search(line):
                prev_title = line_num
                res = TITLE_TAG.search(line)
                word = res.group(1)

            # if line_num < 1000000:
            #     continue

            # if line_num > 2000000:
            #     break

            if LANG_HEADER.match(line):

                res = LANG_HEADER.search(line)
                lang = res.group(2)

                if inside_lv:
                    etymology = re.sub(r"\s+", " ", " ".join(buffer)).strip()
                    writer.writerow([prev_title, line_num, word, word_content, etymology])

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

            if HEADER.match(line):

                res = HEADER.search(line)
                name = res.group(1)

                if inside_lv:
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
    parser.add_argument("--out", default="latvian_extracted.csv")
    parser.add_argument("--allow-cyrillic", action="store_true")
    args = parser.parse_args()
    extract_latvian_data(args.xml, args.out, allow_cyrillic=args.allow_cyrillic)


if __name__ == "__main__":
    main()
