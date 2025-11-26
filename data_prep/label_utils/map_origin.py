import argparse
import pandas as pd
import json
from collections import Counter

# level 0 -> level 1.
mapping = {
    "north-germanic": "germanic",
    "west-germanic":  "germanic",
    "romance":        "romance",
    "greek":          "greek",
    "slavic":         "slavic",
    "baltic":         "baltic",
    "ide":            "ide",
    # "indo-iranian"?

    # Non‑Indo-European
    "uralic":         "non-ide",
    "semitic":        "non-ide",
    "etru":           "non-ide",
    "afroasiatic":    "non-ide",
}

furthest_precedence = [
    "non-ide",
    "greek",
    "romance",
    "germanic",
    "slavic",
    "baltic",
    "ide"
]

direct_precedence = [
    x for x in reversed(furthest_precedence) if x not in {"baltic", "ide"}
]

# print(direct_precedence); exit(0)

def remap_origin_value(value, mapping):
    parts = str(value).split("|")
    out = []
    for p in parts:
        p = p.strip()
        if p in mapping:
            out.append(mapping.get(p, p))  # keep original if not mapped
    return "|".join(set(out))

def is_loanword(value):

    origins = set(str(value).split("|"))

    if "unknown" in origins:
        return None

    if "ide" in origins:
        return False 
    
    if origins == {"baltic"}:
        return False 

    return True 

def is_strictly_baltic(value):

    origins = set(str(value).split("|"))

    if origins == {"baltic"}:
        return True 

    return False 

def pick_origin_by_precedence(value, precedence):
    """Select the single origin based on defined precedence order."""
    origins = [p.strip() for p in str(value).split("|") if p.strip()]
    for p in precedence:
        if p in origins:
            return p
    return None  # if none matched (shouldn't normally happen after filtering)

def adjust_non_ide(origins):

    non_ide = "non-ide"
    s = set(origins.split("|"))

    if non_ide not in s:
        return origins

    if s <= {non_ide, "baltic"}:
        return non_ide

    s.discard(non_ide)
    return "|".join(sorted(s))

def count_origin_distribution(df):

    dict_ = {}
    
    for row in df:
        origins = str(row).split("|")
        for origin in origins:
            if origin in dict_:
                dict_[origin] += 1
            else:
                dict_[origin] = 1

    return dict_

def main():

    parser = argparse.ArgumentParser("Nomappo izcelsmes marķējumus uz noteikto līmeni")
    parser.add_argument("input",  help="Input CSV file")
    parser.add_argument("output", help="Output CSV file")
    parser.add_argument("--only-loanwords", action="store_true")
    # parser.add_argument("--no-loanword", help="Infer and add the is_loanword label", action="store_true", default=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    stats = Counter()
    stats["rows_total"] = len(df)

    if "origin" not in df.columns:
        raise ValueError("CSV must contain an 'origin' column")

    df["origin_l1"] = df["origin"].apply(lambda x: remap_origin_value(x, mapping))
    # df = df[df["origin"].str.strip() != ""]
    df["origin_l1"] = df["origin_l1"].apply(adjust_non_ide)

    df["is_loanword"] = df["origin_l1"].apply(lambda x: is_loanword(x))
    df["is_strictly_baltic"] = df["origin_l1"].apply(lambda x: is_strictly_baltic(x))

    df["furthest_origin"] = df["origin_l1"].apply(
        lambda x: pick_origin_by_precedence(x, furthest_precedence)
    )

    df["direct_origin"] = df["origin_l1"].apply(
        lambda x: pick_origin_by_precedence(x, direct_precedence)
    )

    if args.only_loanwords:
        df = df[df["is_loanword"] == True]

    df.to_csv(args.output, index=False)

    # Aggregate stats after processing
    # Row-level statistics
    stats["rows_final"] = len(df)
    
    # Loanword presence
    stats["has_loanword_label"] = int(df["is_loanword"].notna().sum())
    
    # Furthest origin presence
    stats["has_furthest_label"] = int(df["furthest_origin"].notna().sum())

    stats["has_baltic_label"] = int(df["is_strictly_baltic"].notna().sum())
    
    # Loanword distribution
    loanword_count = int(df["is_loanword"].sum()) 
    stats["loanwords"] = { "true": loanword_count, "false": stats["has_loanword_label"] - loanword_count}

    baltic_count = int(df["is_strictly_baltic"].sum()) 
    stats["strictly_baltic_count"] = { "true": baltic_count, "false": stats["has_baltic_label"] - baltic_count}

    # Origin label distribution
    stats["origins_distribution"] = count_origin_distribution(df["origin_l1"])
    stats["furthest_origins_distribution"] = dict(Counter(df["furthest_origin"].dropna()))
    stats["direct_origins_distribution"] = dict(Counter(df["direct_origin"].dropna()))

    # Print summary
    print("=== Processing summary ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
        # if k == "origins_distribution":
        #     print(f"{k}:")
        #     for wk, wv in v.items():
        #         print(f"  {wk}: {wv}")
        # else:
        #     print(f"{k}: {v}")
    print(f"\nSaved processed CSV to {args.output}")

if __name__ == "__main__":
    main()
