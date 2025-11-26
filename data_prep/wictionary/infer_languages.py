import csv
import argparse
import re
import json
from random import randrange
from collections import Counter

# Inverted map: top-level family → list of ISO/etymology codes
# family_to_codes = {
#
#     # Baltic
#     "baltic":          ["lv", "lt", "prg", "xsv", "bat-pro"],
#
#     # Indo-European (including Balto-Slavic)
#     "indoeuropean":    ["ine-pro", "ine-bsl-pro"],
#
#     # Greek
#     "greek":           ["grc", "el"],
#
#     # Romance
#     "romance":         ["la", "VL.", "ML.", "LL.", "NL.", "it", "fr", "es", "pt",
#                         "ro", "ca",],
#
#     # Slavic
#     "slavic":          ["ru", "pl", "cs", "uk", "be", "bg", "sh", "cu",
#                         "orv", "sla-pro"],
#
#     # North Germanic
#     "north-germanic": ["non", "is", "no", "nb", "sv", "da"],
#
#     # West Germanic
#     "west-germanic": ["de", "en", "nl", "dum", "gml", "gmh",
#                       "goh", "ang", "ofs", "osx", "odt",
#                       "gmw-pro", "gem-pro", "gem", "got"],
#     # Uralic
#     "uralic":        ["fi", "et", "liv", "urj-fin-pro",
#                       "urj-fin", "smi", "se", "krl"],
# }

# Inverted map: top-level family → list of ISO/etymology codes
family_to_codes = {

    # Baltic
    "baltic": [
        "lv", "lt", "prg", "xsv", "bat-pro", "bat", "de-bal"
    ],

    # Slavic
    "slavic": [
        "ru", "pl", "cs", "sk", "uk", "be", "bg", "sh", "cu",
        "orv", "sla", "sla-pro", "zlw", "zlw-ocs", "zlw-opl",
        "zle-mbe", "zle-ort", "zls", "csb"
    ],

    # North Germanic
    "north-germanic": [
        "non", "is", "no", "nb", "nn", "sv", "da", "fo",
        "gmq", "gmq-osw", "gmq-oda"
    ],

    # West Germanic
    "west-germanic": [
        "de", "nds", "nds-de", "en", "nl", "dum", "gml", "gmh", "goh",
        "ang", "ofs", "osx", "odt", "gem-pro", "gmw-pro", "gem",
        "stq", "yi"
    ],

    # Romance
    "romance": [
        "la", "la-new", "la-med", "la-ren",
        "VL.", "ML.", "LL.", "NL.",
        "it", "fr", "es", "pt", "pt-BR", "ro", "ca", "vec", "frm", "fro",
        "roa"
    ],

    # Greek
    "greek": ["grc", "grc-koi", "gkm", "el", "EL."],

    # Indo-Iranian
    "indo-iranian": [
        "ira", "ira-mid", "ira-pro", "fa", "fa-cls", "peo", "ps",
        "tg", "kk", "ky", "ur", "hi", "sa", "inc-pro", "iri", "pal"
    ],

    # Indo-European (other / proto-unclassified)
    "ide": [
        "ine", "ine-pro", "ine-bsl-pro", "itc-pro", "osc",
        "cel", "cel-gau", "sq", "alb", "ett", "xcu", "xbc", "pro",
    ],

    # Uralic / Finno-Ugric
    "uralic": [
        "fi", "et", "liv", "krl", "sjd", "se",
        "urj-fin-pro", "urj-fin", "urj-pro", "urj",
        "ural-pro"
    ],

    # Turkic
    "turkic": [
        "trk", "trk-pro", "trk-ogr", "tk", "tt", "uz"
    ],

    # Afroasiatic / Semitic
    "semitic": [
        "ar", "he", "arc", "phn", "syc", "mt", "egy", "afa", "akk"
    ],

    # Caucasian & Kartvelian
    "caucasian": ["ka", "kky"],

    # Altaic (if kept distinct from Turkic)
    "altaic": ["mn", "ja", "ko"],

    # Austronesian
    "austronesian": [
        "poz-pro", "poz-mly-pro", "poz-oce-pro",
        "map-pro", "ms", "mi", "to"
    ],

    # Sino-Tibetan
    "sino-tibetan": [
        "zh", "cmn", "yue", "ltc", "lzh", "nan-hbl"
    ],

    # Indo-European Peripheral / Other
    "celtic": ["cel", "cel-gau"],
    "armenian": ["xno"],
    "albanian": ["sq"],

    # Native American groups (Aztec / Nahuatl)
    "aztec": ["azc-pro", "azc-nah", "nah"],

    # Niger-Congo / Bantu
    "bantu": ["bnt", "bnt-pro", "bnt-cmn"],

    # Austroasiatic
    "austroasiatic": ["km"],

    # Austric / unclear Oceanic
    "austric": ["tpw"],

    # Misc isolates
    "constructed": ["eo"],      # Esperanto
    "creole": ["lg", "wo"],     # Add more if creole-like
    "unknown": ["nocap=1", "mul"],

}

# Templates indicating direct etymological relationship (not just cognates)
origin_templates = {
    "inh",
    "der",
    "bor",
    "uder",
    "dercat",
    "borrowed",
    "calque",
    "cal",
    "clq",
    "bor+",
    "der+",
}

origin_markings = set()

def classify_origin(templates):

    # Build reverse lookup: code → family
    lang_map = {}
    for fam, codes in family_to_codes.items():
        for code in codes:
            lang_map[code] = fam

    origins = set()

    for template in templates:
        if not template or len(template) < 2:
            continue

        template_name = template[0]

        # Direct etymological relationship: check source language
        if template_name in origin_templates and len(template) >= 3:
            source_lang = template[2]  # template[1] is usually the target language ('lv')
            origin_markings.add(source_lang)
            if source_lang in lang_map:
                origins.add(lang_map[source_lang])

    return origins if origins else set()


def extract_templates(text):
    pattern = r"\{\{([^}]+)\}\}"
    matches = re.findall(pattern, text)
    return [[elem.strip() for elem in match.split("|")] for match in matches]


def read_csv(csv_in, csv_out, add_etymology=False):

    cols = None
    new_rows = []
    origin_counter = Counter()

    with open(csv_in, "r", encoding="utf-8", errors="ignore") as in_:
        reader = csv.DictReader(in_)
        cols = next(reader)

        # Headers: line,language,word,etymology_text
        for row in reader:
            add = True

            lang = row["language"].lower().strip()
            # if lang not in {"latvian", "lithuanian"}:
            if lang not in {"latvian"}:
                add = False
                continue

            templates = extract_templates(row["etymology_text"])

            if len(templates) == 1 and templates[0][0] in {"rfe", "suffix", "af"}:
                add = False

            if len(templates) == 0:
                add = False

            # 10% sample
            # if randrange(10) >= 1:
            #     add = False

            origin = classify_origin(templates)
            if len(origin) == 0:
                add = False

            if add:
                # Count each high-level origin family
                for fam in origin:
                    origin_counter[fam] += 1

                new_row = [
                    row["word"],
                    "|".join(sorted(origin)),
                ]

                if add_etymology:
                    new_row.append(row["etymology_text"])
            
                new_rows.append(new_row)

    with open(csv_out, "w", encoding="utf-8") as out:
        writer = csv.writer(out, delimiter=",", quotechar='"')
        writer.writerows([["word", "origin"]])
        writer.writerows(new_rows)


    # Print summary counter of families
    print("\n=== High-level origin summary ===")
    for fam, count in origin_counter.most_common():
        print(f"{fam:20s}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_in", help="Path to Wiktionary XML dump")
    parser.add_argument("csv_out", help="Output CSV (tab separated)")
    parser.add_argument("--add-etymology", action="store_true")
    args = parser.parse_args()
    read_csv(args.csv_in, args.csv_out, add_etymology=args.add_etymology)
    # print(origin_markings)


if __name__ == "__main__":
    main()
