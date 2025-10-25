
import csv
import argparse
from collections import Counter
import re

origin_counter = Counter()
loanword_counter = Counter()

ABBT_EXP_RE = re.compile(r"[^a-zāčēģīķļņšūž.]", re.IGNORECASE)

# CUE index.
LEV_CUES = {

    # Borrowings.
    "scandinavian": ["d.", "norv.", "zv.", "island.", "sisl.", "ssk.", "szv.", "sv."], # maybe. ssak., sensak?
    "germanic":     ["v.", "germ.", "ang.", "vv.", "vlv.", "bv.", "lv.", "sav.", "vav."
                    "sfrī.", "h.", "hol."],
    "romance":    	["lat.", "jlat.", "it.", "fr.", "rum.", "sfr.", "f.", ".vlat"],
    "greek":	    ["gr." ],
    "uralic":	    ["somu", "s‑u.", "ung.", "ig.", "līb." ],
    "slavic":	    ["kr.", "k.", "skr.", "sl.", "ukr.", "p.", "bulg.", "č", "ssl"],

    # Native classes.
    # "latvian":       [],
    "baltic":	    ["apv.", "la.", "b.", "ab.", "lš.", "pr.",
                     "narev.", "kurs.", "kursen."], # atv. (add retrieval for atv.?)
    "indoeuropean": ["ide.", "pirmside.", "indoeiropiešu", "lde."],
}

SUB_CUES = ["jaunvārds"]

# Inverse index.
inv_lev_cues = {}
for group, cues in LEV_CUES.items():
    for cue in cues:
        if cue in inv_lev_cues:
            raise Exception(
                "Duplicate cue:", cue, 
                " found in ", inv_lev_cues[cue],
                " want to add to ", group
            )
        else:
            inv_lev_cues[cue] = group

# CLI args.
parser = argparse.ArgumentParser(description="Label origins.")
parser.add_argument('input', help='Path(s) to input CSV file(s).')
args = parser.parse_args()

print(args.input)

def label(cues, inv_lev_cues):

    groups = []
    for cue in cues:
        g = inv_lev_cues.get(cue)
        if g:
            groups.append(g)
    if not groups:
        return {"unknown"}, None 

    group_cnt = Counter(groups)
    group_set = set(groups)

    is_loanword = not ({"baltic", "indoeuropean"} <= group_set)

    return group_set, is_loanword

    # if "indoeuropean" in group_set:
    #     return {"indoeuropean"}
    #
    # if group_set <= {"baltic"}:
    #     return {"baltic"}
    #
    # for x in ["romance", "scandinavian", "greek", "germanic", "slavic"]:
    #     if group_set <= {"baltic", x}:
    #         return {x}
    #
    # if group_set <= {"baltic", "romance", "scandinavian", "greek", "germanic", "slavic"}:
    #     return (group_set - {"baltic"})

    # return {"unknown"}


with open(args.input, newline='') as csvfile:

    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    cols = spamreader.__next__()

    # RANGE = [400,500]
    RANGE = None

    for i, row in enumerate(spamreader):

        if RANGE:
            if i < RANGE[0]:
                continue
            elif i >= RANGE[1]:
                break

        headword = row[cols.index("headword")]
        text = row[cols.index("text")]


        cues = []
        for w in text.split():
            candidate = w.lower()
            candidate = ABBT_EXP_RE.sub("", candidate)
            if candidate in inv_lev_cues or candidate in SUB_CUES:
                cues.append(candidate)

        origin, is_loanword = label(cues, inv_lev_cues)
        origin_counter.update(origin)
        loanword_counter.update({is_loanword})

        show = origin == {"unknown"}

        if show:
            print(
                headword,
                "cues: " + ', '.join(cues),
                "label: " + str(origin),
                "is_loanword: " + str(is_loanword),
                text,
                "__________________",
                sep="\n"
            )

print(origin_counter)
print(loanword_counter)
