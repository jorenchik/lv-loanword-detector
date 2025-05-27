import fitz # PyMuPDF
import re

# Load the PDF
pdf_path = "data/LV_etimologijas_vardn.pdf"
doc = fitz.open(pdf_path)

# Extract text from the first few pages for demonstration
extracted_entries = []
allowed_chars_re = re.compile(r"[^a-zāčēģīķļņšūž\-–\[\]]", re.IGNORECASE)
is_first_odd = True

entries = []
words = set()
current_entry = {"headword": "", "text": ""}

first_threshold  = 64 + 3
second_threshold = 30 + 1

if not is_first_odd:
    t = first_threshold 
    first_threshold = second_threshold 
    second_threshold = t

# page_amount = len(doc)
page_amount = 18
for page_num in range(page_amount):
    print("Processing page: " + str(page_num + 1) + "/" + str(page_amount))
    page = doc.load_page(page_num)
    blocks = page.get_text("dict")["blocks"]
    
    is_odd = (page_num + 1) % 2 != 0
    if is_odd: 
        entry_threshold = first_threshold
    else:
        entry_threshold = second_threshold 

    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                x0 = span["bbox"][0]

                # Detect a new entry by left-aligned text
                if x0 < entry_threshold and len(text) > 1:
                    cleaned_text = allowed_chars_re.sub("", text)

                    if cleaned_text in words:
                        continue

                    if current_entry["headword"]:
                        entries.append(current_entry)

                    words.add(cleaned_text)

                    current_entry = {"headword": cleaned_text, "text": ""}
                else:
                    current_entry["text"] += text + " "

print(words)

# Add final entry if exists
if current_entry["headword"]:
    entries.append(current_entry)

with open("temp_results.txt", "w") as f:
    for en in entries:
        f.write(en["headword"] + ": " + en["text"] + "\n")

print(extracted_entries)
