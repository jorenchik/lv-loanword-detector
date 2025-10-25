import re
import csv
import argparse

LANG_HEADER = re.compile(r"^==[A-Z][A-Za-z ]+==")
ETYMOLOGY_HEADER = re.compile(r"^===Etymology")
TITLE_TAG = re.compile(r"<title>(.*?)</title>")
TEMPLATE_RE = re.compile(r"\{\{([^{}]+)\}\}")

def extract_templates(text: str):
    matches = TEMPLATE_RE.findall(text)
    templates = []
    for m in matches:
        parts = [p.strip() for p in m.split("|") if p.strip()]
        templates.append(parts)
    return templates


def valid_word(word: str, allow_cyrillic: bool = False) -> bool:
    latin_lv = r"A-Za-zĀĒĪŪĶĻŅŠŽāēīūķļņšž\-"
    cyr = r"\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F"  # Cyrillic range
    charset = latin_lv + (cyr if allow_cyrillic else "")

    charset_match = re.fullmatch(fr"[{charset}]+", word) is not None
    long_enough = len(word) > 2

    return charset_match and long_enough


def extract_latvian_etymology(
    filename: str, csv_out: str, filename_non_lv: str, allow_cyrillic: bool = False
) -> None:
    inside_latvian = False
    inside_etymology = False
    buffer = []
    current_word = None

    with (
        open(filename, "r", encoding="utf-8", errors="ignore") as f,
        open(csv_out, "w", encoding="utf-8") as out,
        open(filename_non_lv, "w", encoding="utf-8") as out_non_lv,
    ):


        print(out, out_non_lv)
        writer = csv.writer(out, delimiter=",", quotechar='"')
        writer.writerow(["word", "etymology"])

        writer_non_lv = csv.writer(out_non_lv, delimiter=",", quotechar='"')
        writer_non_lv.writerow(["word", "etymology"])

        for line in f:
            # track title
            if "<title>" in line:
                m = TITLE_TAG.search(line)
                if m:
                    current_word = m.group(1).strip()
                continue

            # start Latvian section
            if line.strip() == "==Latvian==":
                inside_latvian = True
                inside_etymology = False
                buffer.clear()
                continue

            # new language section
            if inside_latvian and LANG_HEADER.match(line):
                inside_latvian = False
                inside_etymology = False
                buffer.clear()
                continue

            if not inside_latvian:
                continue

            # etymology section start
            if ETYMOLOGY_HEADER.match(line):
                inside_etymology = True
                buffer.clear()
                continue

            # any new header inside language ends etymology
            if inside_etymology and line.startswith("===") and not ETYMOLOGY_HEADER.match(
                line
            ):
                text = " ".join(line.strip() for line in buffer if line.strip())
                templates = extract_templates(text)
                latvianized = False 

                if len(templates) > 0:
                    if len(templates[0]) > 1:
                        latvianized = templates[0][1] == "lv"
            
                if text and current_word and valid_word(current_word, allow_cyrillic):
                    if latvianized:
                        writer.writerow([current_word, text])
                    else:
                        writer_non_lv.writerow([current_word, text])

                inside_etymology = False
                buffer.clear()
                continue

            # accumulate etymology text
            if inside_etymology:
                buffer.append(line)

        # flush buffer if file ends inside etymology
        if inside_etymology and buffer and current_word:
            text = "".join(buffer).strip()
            if text and valid_word(current_word, allow_cyrillic):
                writer.writerow([current_word, text])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xml", help="Path to Wiktionary XML dump")
    parser.add_argument(
        "--out",
        default="latvian_etymology.csv",
        help="Output CSV (tab separated)",
    )
    parser.add_argument(
        "--out-non-lv",
        default="non_latvian_etymology.csv",
        help="Output CSV (tab separated)",
    )
    parser.add_argument(
        "--allow-cyrillic",
        action="store_true",
        help="Accept words containing Cyrillic characters",
    )

    args = parser.parse_args()
    extract_latvian_etymology(args.xml, args.out, args.out_non_lv, allow_cyrillic=args.allow_cyrillic)


if __name__ == "__main__":
    main()
