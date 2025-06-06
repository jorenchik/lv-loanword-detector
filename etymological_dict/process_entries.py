import csv

# Latvian alphabetical base order (accents treated as base)
base_order = list("abcdefghijklmnopqrstuvwxyz")
char_rank = {char: i for i, char in enumerate(base_order)}
normalize_map = str.maketrans({
    "ā": "a", "č": "c", "ē": "e", "ģ": "g", "ī": "i", "ķ": "k",
    "ļ": "l", "ņ": "n", "š": "s", "ū": "u", "ž": "z"
})

def normalize_latvian(s: str) -> str:
    return s.lower().translate(normalize_map)

def lv_compare(a: str, b: str) -> int:
    a = normalize_latvian(a)
    b = normalize_latvian(b)
    for ca, cb in zip(a, b):
        ra = char_rank.get(ca, -1)
        rb = char_rank.get(cb, -1)
        if ra != rb:
            return ra - rb
    return len(a) - len(b)

# Read CSV
with open("v1_entry_raw_data.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    data = list(reader)

header = data[0]
rows = data[1:]

# Group by page
from collections import defaultdict

page_to_words = defaultdict(list)
for row in rows:
    row_number, headword, page, text = row
    page_to_words[page].append(headword.strip())

# Check for disorder
disordered_pages = []

for page, words in page_to_words.items():
    last = None
    breaks = 0
    for word in words:
        if last is not None and lv_compare(word, last) < 0:
            breaks += 1
        last = word
    if breaks >= 2:
        disordered_pages.append((page, breaks, len(words)))

# Print results
print("Pages with broken alphabetical order:")
for page, breaks, total in disordered_pages:
    print(f"Page {page}: {breaks} breaks in {total} entries")

# Generate Python array of page numbers
disordered_page_numbers = [int(page) + 4 for page, _, _ in disordered_pages]
print("\nPython array of disordered pages (indixes):")
print(disordered_page_numbers)
