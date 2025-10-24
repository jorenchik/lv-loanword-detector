import fitz  # PyMuPDF
import re
import csv
import argparse
from tqdm import tqdm

def main():

    # CLI args.
    parser = argparse.ArgumentParser(description="Load and process a PDF file.")
    parser.add_argument(
        "pdf_path", 
        type=str, 
        help="Path to the input PDF file."
    )
    args = parser.parse_args()

    # Load the PDF
    pdf_path = args.pdf_path
    doc = fitz.open(pdf_path)

    # Constants. 
    ALLOWED_CHARS = r"[^a-zāčēģīķļņšūž\-–\[\]]"
    ALLOWED_CHARS_RE = re.compile(ALLOWED_CHARS, re.IGNORECASE)
    PAGE_OFFSET = -3
    RANGE_START = 57
    RANGE_END = 1219
    INDENT_THRESHOLD = 4
    page_range = range(RANGE_START - 1, RANGE_END)

    # Manual marking of pages where there is no headwords (entries).
    PAGE_WITH_NO_HEADWORDS = [
      117, 172, 271, 297, 334, 397, 472, 492, 493, 499, 510, 535, 582, 655, 
      670, 689, 743, 808, 975, 1009, 1025, 1043, 1049, 1092, 1157, 1214
    ]

    for i in range(len(PAGE_WITH_NO_HEADWORDS)): 
        PAGE_WITH_NO_HEADWORDS[i] -= 1

    # Track processed headwords
    words = set()
    current_entry = {"headword": "", "text_parts": [], "page": 1 + PAGE_OFFSET}

    # Open CSV file and temp output
    row_number = 1
    csv_file = open("entry_raw_data.csv", "w", encoding="utf-8", newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["row_number", "headword", "page", "text"])  # Header

    with open("temp_results.txt", "w", encoding="utf-8") as f:

        for page_num in tqdm(page_range, desc="Processing pages"):

            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]

            # Collect all positions in a page.
            positions = []
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        x0 = span["bbox"][0]
                        positions.append(x0)

            # Positions of starting points in a line.
            # It is determined by decrease or same value as previous.
            # 10 20 30 15 20 15 -> 10 15 15.
            starting_positions = []
            for i in range(len(positions)):
                if i == 0 or positions[i] <= positions[i - 1]:
                    starting_positions.append(positions[i])

            # Determine indentation using a predetermined threshold.
            # There are 3 relevan indentations.
            positions_sorted = sorted(set(positions))
            indent_levels = [positions_sorted[0]] if positions_sorted else []
            for pos in positions_sorted[1:]:
                if all(abs(pos - level) >= INDENT_THRESHOLD for level in indent_levels):
                    indent_levels.append(pos)
                if len(indent_levels) >= 3:
                    break

            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):

                        text = span["text"].strip()
                        x0 = span["bbox"][0]

                        if page_num in PAGE_WITH_NO_HEADWORDS:
                            current_entry["text_parts"].append(text)
                            continue

                        # Determine the indent level.
                        indent_level = None
                        for i, lvl in enumerate(indent_levels):
                            if abs(x0 - lvl) < INDENT_THRESHOLD:
                                indent_level = i
                                break

                        is_headword = indent_level == 0 and len(text) > 1
                        if is_headword:

                            # cleaned_text = ALLOWED_CHARS_RE.sub("", text)
                            if re.search(ALLOWED_CHARS, text, re.IGNORECASE):
                                cleaned_text = ALLOWED_CHARS_RE.sub("", text)
                            else:
                                cleaned_text = text

                            if cleaned_text in words:
                                continue

                            if current_entry["headword"]:

                                # Write to result.
                                csv_writer.writerow([
                                    row_number,
                                    current_entry["headword"],
                                    current_entry["page"] - 1,
                                    " ".join(current_entry["text_parts"])
                                ])

                                # Write to temp (progress) file.
                                f.write(
                                    current_entry["headword"] + "(" +
                                    str(current_entry["page"]) + "): " +
                                    " ".join(current_entry["text_parts"]) + "\n"
                                )

                                row_number += 1

                            words.add(cleaned_text)

                            # Start new entry.
                            current_entry = {
                                "headword": cleaned_text,
                                "text_parts": [],
                                "page": page_num + 1 + PAGE_OFFSET
                            }

                        else:
                            current_entry["text_parts"].append(text)


    # Handle final entry.
    if current_entry["headword"]:
        csv_writer.writerow([
            row_number,
            current_entry["headword"],
            current_entry["page"] - 1,
            current_entry["text"]
        ])

    # Close the CSV file
    csv_file.close()

if __name__ == "__main__":
    main()
