import fitz # PyMuPDF
import re

import csv
import io

# Load the PDF
# pdf_path = "data/LV_etimologijas_vardn.pdf"
pdf_path = "data/Latviesu etimologijas vardnica (2001).pdf"

doc = fitz.open(pdf_path)

# Extract text from the first few pages for demonstration
extracted_entries = []
allowed_chars_re = re.compile(r"[^a-zāčēģīķļņšūž\-–\[\]]", re.IGNORECASE)
is_first_odd = True

PAGE_OFFSET = 1 - 2

PAGE_AMOUNT = 4
# PAGE_AMOUNT = len(doc)

RANGE_START = 57
RANGE_END = 1219

entries = []
words = set()
current_entry = {"headword": "", "text": "", "page": 1 + PAGE_OFFSET}

page_range = range(RANGE_START - 1, RANGE_END)

page_minimums = []
for page_num in page_range:

    print("Page: " + str(page_num + 1))
    minimum = 10000
    minimal_text = None

    positions = []

    page = doc.load_page(page_num)
    blocks = page.get_text("dict")["blocks"]

    for block in blocks:

        for line in block.get("lines", []):
            for span in line.get("spans", []):

                text = span["text"].strip()
                x0 = span["bbox"][0]

                positions.append(x0)

                if x0 <= minimum:
                    minimum = x0
                    minimal_text = text

    positions = sorted(positions)

    jumps = []
    for i in range(1, len(positions)):
        if (positions[i] - positions[i - 1]) >= 5.5:
            jumps.append(positions[i])

    if len(jumps) > 1:
        first_jump_x0 = jumps[0]
    else:
        first_jump_x0 = -1

    print("Page minimum (" + str(page_num + 1) + ") " + str(first_jump_x0))
    page_minimums.append(first_jump_x0 - 2)

with open("temp_results.txt", "w") as f:

    for page_num in page_range:

        print("Processing page: " + str(page_num + 1) + "/" + str(PAGE_AMOUNT))

        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:

            for line in block.get("lines", []):

                for span in line.get("spans", []):
                    text = span["text"].strip()
                    x0 = span["bbox"][0]

                    # print(text, x0)

                    # Detect a new entry by left-aligned text
                    entry_threshold = page_minimums[page_num] + 1.5

                    if x0 < entry_threshold and len(text) > 1:
                        cleaned_text = allowed_chars_re.sub("", text)

                        if cleaned_text in words:
                            continue

                        if current_entry["headword"]:
                            entries.append(current_entry)

                        words.add(cleaned_text)

                        f.write(current_entry["headword"] + "(" + str(current_entry["page"]) + ")" + ": " + current_entry["text"] + "\n")
                        current_entry = {"headword": cleaned_text, "text": "", "page": page_num + 1 + PAGE_OFFSET}
                    else:
                        current_entry["text"] += text + " "

# Add final entry if exists
if current_entry["headword"]:
    entries.append(current_entry)

output = io.StringIO()
writer = csv.writer(output)

for entry in entries:
    row = [entry["headword"], entry["page"], entry["text"]]
    writer.writerow(row)

with open("entry_raw_data.csv", "w") as f:
    f.write(output.getvalue())
output.close()
