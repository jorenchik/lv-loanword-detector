import argparse
import pandas as pd
import json
from collections import Counter

# level 0 -> level 1.
# Keys of LEV_CUES and family_to_codes are the L0 origins found in the input 'origin' column.
# Values are the consolidated L1 categories.

mapping = {
    # --- GERMANIC ---
    "north-germanic": "germanic",
    "west-germanic":  "germanic",
    
    # --- ROMANCE ---
    "romance":        "romance",
    
    # --- GREEK ---
    "greek":          "greek",
    
    # --- SLAVIC ---
    "slavic":         "slavic",

    # --- BALTIC ---
    "baltic":         "baltic",
    
    # --- INDO-IRANIAN ---
    "indo-iranian":   "indo-iranian",
    
    # --- IDE-OTHER (for general IE and less common IE branches/proto-forms) ---
    "ide":            "ide",
    "armenian":       "ide-other",        # From LEV_CUES / family_to_codes
    "albanian":       "ide-other",        # From LEV_CUES / family_to_codes
    "illyrian":       "ide-other",        # From LEV_CUES
    "thracian":       "ide-other",        # From LEV_CUES
    "tocharian":      "ide-other",        # From LEV_CUES
    "celtic":         "ide-other",        # From family_to_codes

    # --- NON-IDE (for all non-Indo-European families) ---
    "uralic":         "non-ide",          # From LEV_CUES / family_to_codes
    "etruscan":       "non-ide",          # From LEV_CUES
    "semitic":        "non-ide",          # From LEV_CUES / family_to_codes
    "turkic":         "non-ide",          # From family_to_codes
    "caucasian":      "non-ide",          # From family_to_codes
    "altaic":         "non-ide",          # From family_to_codes
    "austronesian":   "non-ide",          # From family_to_codes
    "sino-tibetan":   "non-ide",          # From family_to_codes
    "aztec":          "non-ide",          # From family_to_codes
    "bantu":          "non-ide",          # From family_to_codes
    "austroasiatic":  "non-ide",          # From family_to_codes
    "austric":        "non-ide",          # From family_to_codes

    # --- SPECIAL / UNKNOWN ---
    "constructed":    "unknown",          # From family_to_codes
    "creole":         "unknown",          # From family_to_codes
    "unknown":        "unknown",          # Explicit 'unknown' tag (already in your mapping)
}

# level 1 -> level 2 (merge greek + romance)
mapping_l2 = {
    "greek":      "greco-romance",
    "romance":    "greco-romance",
    "germanic":   "germanic",
    "slavic":     "slavic",
    "non-ide":    "non-ide",
    "baltic":     "baltic",
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
    x for x in reversed(furthest_precedence) if x not in {"baltic", "ide", "non-ide"} 
]

# direct_precedence = [
#     "non-ide",
#     "greco-romance",
#     "germanic",
#     "slavic"
# ]

def remap_origin_value(value, mapping):
    parts = str(value).split("|")
    out = []
    for p in parts:
        p = p.strip()
        if p in mapping:
            out.append(mapping.get(p, p))
    return "|".join(set(out))

def is_loanword(value):
    origins = set(str(value).split("|"))

    if "unknown" in origins:
        return None

    if "ide" in origins: # If 'ide' is explicitly mentioned, it's considered native/inherited, not a loan.
        return False
    
    if origins == {"baltic"}: # Strictly Baltic means native, not a loan.
        return False
    
    # Filter out 'baltic' and 'ide' to count non-Baltic/non-IDE sources
    non_baltic_non_ide_sources = {o for o in origins if o not in {"baltic", "ide", "unknown"}}
    
    if not non_baltic_non_ide_sources:
        # If no non-Baltic/non-IDE sources are left, it's not a loan (e.g., just {'baltic'} or {'ide'})
        return False
    
    # Apply your specific logic for 'baltic' presence
    if "baltic" in origins:
        # Case: 'baltic' + one non-Baltic/non-IDE group
        if len(non_baltic_non_ide_sources) == 1:
            return True # This is a loan based on your rule
        # Case: 'baltic' + multiple non-Baltic/non-IDE groups
        elif len(non_baltic_non_ide_sources) > 1:
            return False # This is considered native/IDE based on your rule
    
    # If 'baltic' is NOT in origins, and there are non-Baltic sources, it's a clear loan.
    # Example: origins = {'slavic'} or {'slavic', 'germanic'} (without 'baltic')
    return True

def is_strictly_baltic(value):
    origins = set(str(value).split("|"))

    if origins == {"baltic"}:
        return True 

    return False

def categorize_as_native_or_loan(is_loanword_val, origin_val):
    """Return 'native' if not a loanword, otherwise return the origin."""
    if is_loanword_val is False:
        return "native"
    elif is_loanword_val is None:
        return "unknown"
    else:
        return origin_val

def pick_origin_by_precedence(value, precedence):
    """Select the single origin based on defined precedence order."""
    origins = [p.strip() for p in str(value).split("|") if p.strip()]
    for p in precedence:
        if p in origins:
            return p
    return None

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
    parser = argparse.ArgumentParser(
        "Nomappo izcelsmes marķējumus uz noteikto līmeni"
    )
    parser.add_argument("input",  help="Input CSV file")
    parser.add_argument("output", help="Output CSV file")
    parser.add_argument("--only-loanwords", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    stats = Counter()
    stats["rows_total"] = len(df)

    if "origin" not in df.columns:
        raise ValueError("CSV must contain an 'origin' column")

    df["origin_l1"] = df["origin"].apply(
        lambda x: remap_origin_value(x, mapping)
    )
    df["origin_l1"] = df["origin_l1"].apply(adjust_non_ide)

    # Apply l2 mapping for direct_origin
    df["origin_l2"] = df["origin_l1"].apply(
        lambda x: remap_origin_value(x, mapping_l2)
    )

    df["is_loanword"] = df["origin_l1"].apply(lambda x: is_loanword(x))
    df["is_strictly_baltic"] = df["origin_l1"].apply(
        lambda x: is_strictly_baltic(x)
    )

    df["furthest_origin"] = df["origin_l1"].apply(
        lambda x: pick_origin_by_precedence(x, furthest_precedence)
    )

    df["direct_origin"] = df["origin_l2"].apply(
        lambda x: pick_origin_by_precedence(x, direct_precedence)
    )

    # Add native/loan categorization
    df["furthest_category"] = df.apply(
        lambda row: categorize_as_native_or_loan(
            row["is_loanword"], row["furthest_origin"]
        ),
        axis=1
    )
    
    df["direct_category"] = df.apply(
        lambda row: categorize_as_native_or_loan(
            row["is_loanword"], row["direct_origin"]
        ),
        axis=1
    )

    if args.only_loanwords:
        df = df[df["is_loanword"] == True]

    df.to_csv(args.output, index=False)

    # Aggregate stats after processing
    stats["rows_final"] = len(df)
    
    stats["has_loanword_label"] = int(df["is_loanword"].notna().sum())
    stats["has_furthest_label"] = int(df["furthest_origin"].notna().sum())
    stats["has_baltic_label"] = int(df["is_strictly_baltic"].notna().sum())
    
    # Fixed loanword distribution
    loanword_true = int((df["is_loanword"] == True).sum())
    loanword_false = int((df["is_loanword"] == False).sum())
    loanword_none = int(df["is_loanword"].isna().sum())
    stats["loanwords"] = {
        "true": loanword_true,
        "false": loanword_false,
        "unknown": loanword_none
    }

    baltic_count = int(df["is_strictly_baltic"].sum()) 
    stats["strictly_baltic_count"] = {
        "true": baltic_count,
        "false": stats["has_baltic_label"] - baltic_count
    }

    # Origin label distribution
    stats["origins_distribution"] = count_origin_distribution(
        df["origin_l1"]
    )
    stats["furthest_origins_distribution"] = dict(
        Counter(df["furthest_origin"].dropna())
    )
    stats["direct_origins_distribution"] = dict(
        Counter(df["direct_origin"].dropna())
    )
    stats["furthest_category_distribution"] = dict(
        Counter(df["furthest_category"].dropna())
    )
    stats["direct_category_distribution"] = dict(
        Counter(df["direct_category"].dropna())
    )

    # Print summary
    print("=== Processing summary ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"\nSaved processed CSV to {args.output}")

if __name__ == "__main__":
    main()
