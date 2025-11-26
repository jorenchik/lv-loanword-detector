
import argparse
import re

from tqdm import tqdm

def remove_things(input_file, output_file):

    with (
        open(input_file, "r",  encoding='utf-8') as in_f,
        open(output_file, "w", encoding="utf-8") as out_f
    ) :

        raw_tokens = [line.strip('<>\n ') for line in in_f if line.strip()]
        for t in tqdm(raw_tokens, desc="Filtering/writing tokens"):
            out_f.write(f"{t}\n")

    print(f"Cleaned tokens saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="""
            Normalize and filter tokens from <token> format file.
        """
    )
    parser.add_argument(
        "input_file",
        help="Path to raw token file"
    )
    parser.add_argument(
        "output_file",
        default="cleaned_tokens.txt",
        help="Output path for cleaned token file"
    )
    args = parser.parse_args()
    remove_things(
        args.input_file,
        args.output_file,
    )

if __name__ == "__main__":
    main()
