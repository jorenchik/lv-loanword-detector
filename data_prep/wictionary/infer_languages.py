

import csv
import argparse
import fasttext

model = fasttext.load_model("lid.176.ftz")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_in", help="Path to Wiktionary XML dump")
    parser.add_argument(
        "csv_out",
        help="Output CSV (tab separated)",
    )
    args = parser.parse_args()

    cols = None
    new_rows = []
    with (
        open(args.csv_in, "r", encoding="utf-8", errors="ignore") as f
    ):

        reader = csv.DictReader(f)
        cols = reader.__next__()

        for row in reader:
            word = row["word"]
            label, prob = model.predict([word], k=1)
            new_row = [word, row["etymology"], label, prob]
            new_rows.append(new_row)

    with open(args.csv_out, "w", encoding="utf-8") as out:

        writer = csv.writer(out, delimiter=",", quotechar='"')
        writer.writerows([cols])
        writer.writerows(new_rows)



if __name__ == "__main__":
    main()
