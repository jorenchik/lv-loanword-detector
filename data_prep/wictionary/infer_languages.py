
import csv
import argparse
import re
import json
from random import randrange

def classify_origin(templates):

    lang_map = {
        # Baltic
        'lv': 'baltic', 'lt': 'baltic', 'prg': 'baltic', 'xsv': 'baltic',
        'bat-pro': 'baltic',
        
        # Indo-European (including Balto-Slavic)
        'ine-pro': 'indoeuropean', 'ine-bsl-pro': 'indoeuropean',
        
        # Greek
        'grc': 'greek', 'el': 'greek',
        
        # Romance
        'la': 'romance', 'VL.': 'romance', 'ML.': 'romance', 'LL.': 'romance',
        'NL.': 'romance', 'it': 'romance', 'fr': 'romance', 'es': 'romance',
        'pt': 'romance', 'ro': 'romance', 'ca': 'romance',
        
        # Slavic
        'ru': 'slavic', 'pl': 'slavic', 'cs': 'slavic', 'uk': 'slavic',
        'be': 'slavic', 'bg': 'slavic', 'sh': 'slavic', 'cu': 'slavic',
        'orv': 'slavic', 'sla-pro': 'slavic',
        
        # North Germanic
        'non': 'north germ.', 'is': 'north germ.', 'no': 'north germ.',
        'nb': 'north germ.', 'sv': 'north germ.', 'da': 'north germ.',
        
        # West Germanic
        'de': 'west germ.', 'en': 'west germ.', 'nl': 'west germ.',
        'dum': 'west germ.', 'gml': 'west germ.', 'gmh': 'west germ.',
        'goh': 'west germ.', 'ang': 'west germ.', 'ofs': 'west germ.',
        'osx': 'west germ.', 'odt': 'west germ.', 'gmw-pro': 'west germ.',
        'gem-pro': 'west germ.', 'gem': 'west germ.', 'got': 'west germ.',
        
        # Uralic
        'fi': 'uralic', 'et': 'uralic', 'liv': 'uralic', 
        'urj-fin-pro': 'uralic', 'urj-fin': 'uralic', 
        'smi': 'uralic', 'se': 'uralic', 'krl': 'uralic',
    }
    
    # Templates indicating direct etymological relationship (not just cognates)
    origin_templates = {'inh', 'der', 'bor', 'uder', 'dercat', 'borrowed', 
                       'calque', 'cal', 'clq', 'bor+', 'der+'}
    
    origins = set()
    
    for template in templates:
        if not template or len(template) < 2:
            continue
        
        template_name = template[0]
        
        # Direct etymological relationship: check source language
        if template_name in origin_templates and len(template) >= 3:
            source_lang = template[2]  # template[1] is usually 'lv'
            if source_lang in lang_map:
                origins.add(lang_map[source_lang])
    
    return origins if origins else set()

def extract_templates(text):
    pattern = r'\{\{([^}]+)\}\}'
    matches = re.findall(pattern, text)
    return [[elem.strip() for elem in match.split('|')] for match in matches]

# Example usage
# text = "{{bor|aaa|bbb}} {{suf | bbb | aaa }}"
# result = extract_templates(text)
# print(result)  # [['bor', 'aaa', 'bbb'], ['suf', 'bbb', 'aaa']]


def read_csv(csv_in, csv_out):

    cols = None
    new_rows = []
    with (
        open(csv_in, "r", encoding="utf-8", errors="ignore") as in_
    ):

        reader = csv.DictReader(in_)
        cols = reader.__next__()

        # Headers.
        # line,language,word,etymology_text

        for row in reader:
            
            add = True 

            if row["language"].lower().strip() != "latvian":
                add = False
                continue

            templates = extract_templates(row['etymology_text'])

            if len(templates) == 1 and templates[0][0] == "rfe":
                add = False

            if len(templates) == 1 and templates[0][0] == "suffix":
                add = False

            if len(templates) == 1 and templates[0][0] == "af":
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

                # print(templates)
                # new_row = [word, row["etymology"], label, prob]

                new_rows.append(
                    [
                        row["line"], 
                        row["language"], 
                        row["word"], 
                        origin,
                        templates
                    ]
                )

    with (
        open(csv_out, "w", encoding="utf-8") as out
    ):

        writer = csv.writer(out, delimiter=",", quotechar='"')
        # writer.writerows([cols])
        writer.writerows(new_rows)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_in", help="Path to Wiktionary XML dump")
    parser.add_argument("csv_out", help="Output CSV (tab separated)")
    args = parser.parse_args()
    read_csv(
        args.csv_in,
        args.csv_out
    )


if __name__ == "__main__":
    main()
