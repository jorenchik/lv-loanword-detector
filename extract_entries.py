import fitz  # PyMuPDF
import re
import csv

# Load the PDF
pdf_path = "data/Latviesu etimologijas vardnica (2001).pdf"
doc = fitz.open(pdf_path)

# Configuration
allowed_chars_re = re.compile(r"[^a-zāčēģīķļņšūž\-–\[\]]", re.IGNORECASE)

PAGE_OFFSET = -3
RANGE_START = 57
RANGE_END = 1219

RANGE_START = 172
RANGE_END = RANGE_START 


page_range = range(RANGE_START - 1, RANGE_END)




# Track processed headwords
words = set()
current_entry = {"headword": "", "text": "", "page": 1 + PAGE_OFFSET}

# Open CSV file and temp output
row_number = 1
csv_file = open("entry_raw_data.csv", "w", encoding="utf-8", newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["row_number", "headword", "page", "text"])  # Header

with open("temp_results.txt", "w", encoding="utf-8") as f:

    for page_num in page_range:

        print("\nPage: " + str(page_num + PAGE_OFFSET))
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        # Positions of starting points in a line.
        # It is determined by decrease or same value as previous.
        # 10 20 30 15 20 15 -> 10 15 15.
        # Collect all x0 starting positions
        positions = []
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    x0 = span["bbox"][0]
                    # print(span["text"])
                    positions.append(x0)

        starting_positions = []
        for i in range(len(positions)):
            if i == 0 or positions[i] <= positions[i - 1]:
                starting_positions.append(positions[i])

        # Determine if there are at least 3 levels.
        max_indent = 1
        current_indent = 1
        for i in range(1, len(starting_positions)):
            if starting_positions[i] <= starting_positions[i - 1]:
                current_indent = 1
                continue
            if (starting_positions[i] - starting_positions[i - 1]) > 4:
                current_indent += 1
                if current_indent > max_indent:
                    max_indent = current_indent
        print(starting_positions)

        print(max_indent)

        # Cluster indent positions using threshold (used for headword detection)
        threshold = 4
        positions_sorted = sorted(set(positions))
        indent_levels = [positions_sorted[0]] if positions_sorted else []

        for pos in positions_sorted[1:]:
            if all(abs(pos - level) >= threshold for level in indent_levels):
                indent_levels.append(pos)
            if len(indent_levels) >= 3:
                break

        has_three_levels = len(indent_levels) >= 3

        # Main extraction logic
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    x0 = span["bbox"][0]

                    if has_three_levels:
                        # Assign indent level based on closest indent cluster
                        indent_level = None
                        for i, lvl in enumerate(indent_levels):
                            if abs(x0 - lvl) < threshold:
                                indent_level = i
                                break

                        if indent_level == 0 and len(text) > 1:
                            cleaned_text = allowed_chars_re.sub("", text)

                            if cleaned_text in words:
                                continue

                            if current_entry["headword"]:
                                # Write previous entry to CSV and temp file
                                csv_writer.writerow([
                                    row_number,
                                    current_entry["headword"],
                                    current_entry["page"] - 1,
                                    current_entry["text"]
                                ])
                                f.write(
                                    current_entry["headword"] + "(" +
                                    str(current_entry["page"]) + "): " +
                                    current_entry["text"] + "\n"
                                )
                                row_number += 1

                            words.add(cleaned_text)
                            current_entry = {
                                "headword": cleaned_text,
                                "text": "",
                                "page": page_num + 1 + PAGE_OFFSET
                            }
                            continue

                    # If not a headword, append text
                    current_entry["text"] += text + " "

# Handle final entry
if current_entry["headword"]:
    csv_writer.writerow([
        row_number,
        current_entry["headword"],
        current_entry["page"] - 1,
        current_entry["text"]
    ])

# Close the CSV file
csv_file.close()
