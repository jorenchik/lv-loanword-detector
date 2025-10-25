
import csv
import re
import argparse

# CLI args.
parser = argparse.ArgumentParser()
parser.add_argument('input', help='Path(s) to input CSV file.')
args = parser.parse_args()

ALLOWED_CHARS = r"[^a-zāčēģīķļņšūž\-–\[\]]"
ALLOWED_CHARS_RE = re.compile(ALLOWED_CHARS, re.IGNORECASE)
# ALLOWED_CHARS = r"[^a-zāčēģīķļņšūž,\.\!\?\;\-–\[\]]"

LATVIAN_TO_ASCII = str.maketrans({
    "ā": "a", "Ā": "A",
    "č": "c", "Č": "C",
    "ē": "e", "Ē": "E",
    "ģ": "g", "Ģ": "G",
    "ī": "i", "Ī": "I",
    "ķ": "k", "Ķ": "K",
    "ļ": "l", "Ļ": "L",
    "ņ": "n", "Ņ": "N",
    "š": "s", "Š": "S",
    "ū": "u", "Ū": "U",
    "ž": "z", "Ž": "Z"
})

def latvian_to_ascii(text):
    """Replace Latvian diacritic characters with ASCII equivalents."""
    return text.translate(LATVIAN_TO_ASCII)


def get_reference(words):

    ref = None

    for i, w in enumerate(words):
        if (
            (w == "sk." and len(words) > i + 1 and i > 0 and (words[i-1][-1] == "—")) 
            # (w == "(sk." and len(words) > i + 1 and words[i+1] != ")")

        ):
            # print(words[i], words[i+1])
            return ALLOWED_CHARS_RE.sub("", words[i+1])

        elif (
            ((w == "(sk.)." or w == "(sk.)" or w == "(sk.),") and i > 1)
        ):
            return ALLOWED_CHARS_RE.sub("", words[i-1])

        elif (
            ("—sk." in w and len(words)> i + 1)
        ):
            return ALLOWED_CHARS_RE.sub("", words[i+1])

        # elif (
        #     ("arī:" == w or "ari:" == w and len(words)> i + 1)
        # ):
        #     return ALLOWED_CHARS_RE.sub("", words[i+1])

    return ref


# def retrieve_reference(row, row_index, rows, cols):
#
#     text_col = cols.index("text")
#
#     words    = row[text_col].split()
#     headword = row[cols.index("headword")]
#     ref = get_reference(words)
#
#     i = None
#     if ref and ref in row_index:
#         i = row_index[ref]
#
#     if i and ref != headword:
#         retrieved = retrieve_reference(rows[i], row_index, rows, cols)
#         return retrieved + "<REF> " + row[text_col]
#     else:
#         return row[text_col]


def retrieve_reference(row, row_index, row_index_ascii, rows, cols,
                       visited=None, depth=0, max_depth=10):
    """
    Recursively retrieves referenced text, halts fully on cycle.
    """


    if visited is None:
        visited = set()

    head_col, text_col = cols.index("headword"), cols.index("text")
    headword = row[head_col]
    text = row[text_col]

    # --- guard 1: circular reference -----------------
    if headword in visited:
        # stop recursion immediately
        return f"<CYCLE:{headword}>"

    visited.add(headword)

    # --- guard 2: depth cap --------------------------
    if depth >= max_depth:
        return f"<DEPTH_LIMIT:{headword}>"

    # --- find inline reference -----------------------
    words = text.split()
    ref = get_reference(words)

    # locate referenced row (ascii tolerant)
    i = None
    if ref:
        i = row_index.get(ref)
        if i is None:
            ref_ascii = ref.translate(LATVIAN_TO_ASCII)
            i = row_index_ascii.get(ref_ascii)

    # --- recurse once if valid ------------------------
    if ref and i is not None and ref != headword and ref not in visited:
        retrieved = retrieve_reference(
            rows[i], row_index, row_index_ascii, rows, cols,
            visited=visited,
            depth=depth + 1,
            max_depth=max_depth
        )
        # if recursion terminated in a cycle marker
        # don't concatenate current text again
        if retrieved.startswith("<CYCLE"):
            return retrieved
        return f"{text} <REF> {retrieved}"

    return text


new_rows = []
with (
    open(args.input, newline='') as fin 
):

    reader = csv.reader(fin, delimiter=',', quotechar='"')
    cols = reader.__next__()

    #RANGE = [100, 200]
    RANGE = None

    rows = []
    for i, row in enumerate(reader):
        rows.append(row)

    row_index = {}
    row_index_ascii = {}
    for i, row in enumerate(rows):
        headword = row[cols.index("headword")]
        row_index[headword] = i
        row_index_ascii[headword.translate(LATVIAN_TO_ASCII)] = i

    for i, row in enumerate(rows):

        if RANGE:
            if i < RANGE[0]:
                continue
            elif i >= RANGE[1]:
                break

        headword = row[cols.index("headword")]
        text = row[cols.index("text")]

        # reference = get_reference(text.split())
        # if reference:

        new_row = row.copy()
        new_row[cols.index("text")] = retrieve_reference(
            row,
            row_index,
            row_index_ascii,
            rows, 
            cols,
        )
        new_rows.append(new_row)

with open('retrieved_entries.csv', "w") as fout:
    writer = csv.writer(fout)
    writer.writerows([cols])
    writer.writerows(new_rows)

