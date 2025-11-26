
import csv
import argparse
from collections import Counter
import re

origin_counter = Counter()
# loanword_counter = Counter()

ABBT_EXP_RE = re.compile(r"[^a-zāčēģīķļņšūž.]", re.IGNORECASE)

# CUE index.
LEV_CUES = {
    # Borrowings.
    "north-germanic": ["d.", "norv.", "zv.", "island.", "sisl.", "ssk.", "szv.", "sv."], # AKA. Scandinavian; maybe also ssak., sensak?
    "west-germanic":  ["v.", "germ.", "ang.", "vv.", "vlv.", "bv.", "lv.", "sav.", "vav.", "sfrī.", "h.", "hol."],
    "romance":    	  ["lat.", "jlat.", "it.", "fr.", "rum.", "sfr.", "f.",  "vlat."],
    "greek":	      ["gr." ],

    "slavic":	      ["kr.", "k.", "skr.", "sl.", "ukr.", "p.", "bulg.", "č", "ssl"],

    # Native classes.
    "baltic":	      ["apv.", "la.", "b.", "ab.", "lš.", "pr.", "narev.", "kurs.", "kursen.", "rb."], # atv. (add retrieval for atv.?)
    "ide":            ["ide.", "pirmside.", "indoeiropiešu", "lde."],

    # Non‑IE
    "uralic": ["somu", "s-u.", "s.", "ung.", "ig.", "līb.", "ural."],
    # "uralic": ["somu", "s-u.", "s.", "ung.", "līb."],
    "etruscan": ["etr."],
    "semitic": ["he.", "sebr."],

    # Indo‑European subfamilies actually present
    "indo-iranian": ["ir.", "jr.", "spers.", "oset.", "tadž.", "rer.", "rir.", "si.", "afg.", "pers."],
    "armenian": ["arm."],
    "albanian": ["alb."],
    "illyrian": ["illīr."],
    "thracian": ["trāķ."],
    "tocharian": ["toh."],
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
parser.add_argument('input', help='Path to input CSV file.')
parser.add_argument('output', help='Path to output CSV file.')
parser.add_argument('--debug', action="store_true")
args = parser.parse_args()

def label(cues, inv_lev_cues):

    groups = []
    for cue in cues:
        g = inv_lev_cues.get(cue)
        if g:
            groups.append(g)
    if not groups:
        return {"unknown"}

    group_cnt = Counter(groups)
    group_set = set(groups)

    # is_loanword = not ({"baltic", "indoeuropean"} <= group_set)

    return group_set

    # TODO: remove.
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


with (
    open(args.input, newline='') as in_,
    open(args.output, "w", newline='') as out_,
):

    # Setup reader.
    spamreader = csv.reader(in_, delimiter=',', quotechar='"')
    cols = spamreader.__next__()

    # Setup writer.
    writer = csv.writer(out_)
    writer.writerow(["word", "origin"])

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

        origin = label(cues, inv_lev_cues)
        origin_counter.update(origin)
        # loanword_counter.update({is_loanword})

        # convert origin set to a stable string representation
        origin_str = "|".join(sorted(origin))
        writer.writerow([headword, origin_str]) 

        # DEBUG.
        show = origin == {"unknown"}
        if args.debug and show:
            print(
                headword,
                "cues: " + ', '.join(cues),
                "label: " + str(origin),
                text,
                "__________________",
                sep="\n"
            )

print("[I] Summary...")
print(origin_counter)
# print(loanword_counter)
