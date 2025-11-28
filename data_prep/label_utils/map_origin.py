import argparse
import pandas as pd
import json
import subprocess
from collections import Counter
from pathlib import Path
import sys
from datetime import datetime

# level 0 -> level 1.
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
    "indo-iranian":   "ide",
    
    # --- IDE-OTHER ---
    "ide":            "ide",
    "armenian":       "ide-other",
    "albanian":       "ide-other",
    "illyrian":       "ide-other",
    "thracian":       "ide-other",
    "tocharian":      "ide-other",
    "celtic":         "ide-other",

    # --- NON-IDE ---
    "uralic":         "non-ide",
    "etruscan":       "non-ide",
    "semitic":        "non-ide",
    "turkic":         "non-ide",
    "caucasian":      "non-ide",
    "altaic":         "non-ide",
    "austronesian":   "non-ide",
    "sino-tibetan":   "non-ide",
    "aztec":          "non-ide",
    "bantu":          "non-ide",
    "austroasiatic":  "non-ide",
    "austric":        "non-ide",

    # --- SPECIAL / UNKNOWN ---
    "constructed":    "unknown",
    "creole":         "unknown",
    "unknown":        "unknown",
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

def log(msg, level="INFO"):
    """Simple logger with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", flush=True)

def progress_bar(current, total, prefix="", length=50):
    """Display a progress bar."""
    percent = 100 * (current / float(total))
    filled = int(length * current // total)
    bar = "█" * filled + "-" * (length - filled)
    print(f"\r{prefix} |{bar}| {current}/{total} ({percent:.1f}%)", end="", flush=True)
    if current == total:
        print()

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

    if "ide" in origins:
        return False
    
    if origins == {"baltic"}:
        return False
    
    non_baltic_non_ide_sources = {o for o in origins if o not in {"baltic", "ide", "unknown"}}
    
    if not non_baltic_non_ide_sources:
        return False
    
    if "baltic" in origins:
        if len(non_baltic_non_ide_sources) == 1:
            return True
        elif len(non_baltic_non_ide_sources) > 1:
            return False
    
    return True

def is_strictly_baltic(value):
    origins = set(str(value).split("|"))
    return origins == {"baltic"}

def pick_origin_by_precedence(value, precedence):
    origins = [p.strip() for p in str(value).split("|") if p.strip()]

    if not is_loanword(value):
        return "native"

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

def get_inflections_bulk(words, jar_path, batch_size=100):
    """
    Get inflections for multiple words in batches using stdin.
    
    Args:
        words: List of words to inflect
        jar_path: Path to morphology JAR
        batch_size: Number of words per batch
        
    Returns:
        Dictionary mapping word -> list of inflections
    """
    result_map = {}
    total_batches = (len(words) + batch_size - 1) // batch_size
    
    log(f"Processing {len(words)} words in {total_batches} batches of {batch_size}")
    
    for batch_idx in range(0, len(words), batch_size):
        batch = words[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        
        progress_bar(batch_num, total_batches, prefix=f"Inflecting batch {batch_num}/{total_batches}")
        
        try:
            # Prepare stdin input
            stdin_input = "\n".join(batch)
            
            # Run JAR with stdin
            result = subprocess.run(
                ["java", "-jar", jar_path, "--stdin", "-f", "json"],
                input=stdin_input,
                capture_output=True,
                text=True,
                timeout=60  # Longer timeout for batches
            )
            
            if result.returncode != 0:
                log(f"Batch {batch_num} failed: {result.stderr}", "WARNING")
                for word in batch:
                    result_map[word] = []
                continue
            
            # Parse JSON output
            try:
                data = json.loads(result.stdout)
                
                # Map results back to words
                for idx, word in enumerate(batch):
                    if idx < len(data):
                        lemma_data = data[idx]
                        inflections = []
                        for inflection in lemma_data.get("inflections", []):
                            form = inflection.get("form", "")
                            if form and form != word:
                                inflections.append(form)
                        result_map[word] = list(set(inflections))
                    else:
                        result_map[word] = []
                        
            except json.JSONDecodeError:
                log(f"Invalid JSON in batch {batch_num}", "WARNING")
                for word in batch:
                    result_map[word] = []
        
        except subprocess.TimeoutExpired:
            log(f"Timeout on batch {batch_num}", "WARNING")
            for word in batch:
                result_map[word] = []
        except Exception as e:
            log(f"Error in batch {batch_num}: {e}", "ERROR")
            for word in batch:
                result_map[word] = []
    
    successful = sum(1 for v in result_map.values() if v)
    log(f"Successfully inflected {successful}/{len(words)} words")
    
    return result_map

def augment_dataset(df, jar_path, strategy="upsample", target_count=None, 
                   max_inflections_per_word=5, batch_size=100):
    """
    Augment dataset by generating inflections for underrepresented origins.
    """
    if "direct_origin" not in df.columns:
        raise ValueError("DataFrame must have 'direct_origin' column")
    
    log("Starting dataset augmentation")
    
    # Get distribution
    origin_counts = Counter(df[df["direct_origin"].notna()]["direct_origin"])
    
    if not origin_counts:
        log("No origins to augment", "WARNING")
        return df
    
    # Determine target
    if target_count is None:
        if strategy == "upsample":
            target_count = max(origin_counts.values())
        else:
            target_count = sum(origin_counts.values()) // len(origin_counts)
    
    log(f"Strategy: {strategy}, Target: {target_count} samples per origin")
    log(f"Current distribution: {dict(origin_counts)}")
    
    # Find word column
    word_col = None
    for col in ["word", "lemma", "token", "form"]:
        if col in df.columns:
            word_col = col
            break
    
    if word_col is None:
        log("No word column found", "ERROR")
        return df
    
    log(f"Using column '{word_col}' for inflection")
    
    # Collect all words to inflect per origin
    words_to_inflect = {}
    augmentation_plan = {}
    
    for origin, current_count in origin_counts.items():
        needed = target_count - current_count
        
        if needed <= 0:
            log(f"{origin}: {current_count} samples (no augmentation needed)")
            continue
        
        log(f"{origin}: {current_count} samples, need {needed} more")
        
        origin_df = df[df["direct_origin"] == origin].copy()
        
        # Collect unique words for this origin
        words = origin_df[word_col].dropna().unique().tolist()
        words = [w.strip() for w in words if w and w.strip()]
        
        if not words:
            log(f"{origin}: No valid words found", "WARNING")
            continue
        
        words_to_inflect[origin] = words
        augmentation_plan[origin] = {
            "needed": needed,
            "source_df": origin_df,
            "words": words
        }
        
        log(f"{origin}: Collected {len(words)} unique words for inflection")
    
    # Bulk inflect all words
    all_words = []
    word_to_origins = {}
    
    for origin, words in words_to_inflect.items():
        for word in words:
            if word not in word_to_origins:
                word_to_origins[word] = []
                all_words.append(word)
            word_to_origins[word].append(origin)
    
    log(f"Total unique words to inflect: {len(all_words)}")
    
    inflection_map = get_inflections_bulk(all_words, jar_path, batch_size)
    
    # Generate augmented samples
    augmented_rows = []
    
    for origin, plan in augmentation_plan.items():
        needed = plan["needed"]
        source_df = plan["source_df"]
        words = plan["words"]
        
        log(f"Generating augmented samples for {origin}")
        
        generated = 0
        word_idx = 0
        max_rounds = 10  # Prevent infinite loops
        
        for round_num in range(max_rounds):
            if generated >= needed:
                break
            
            for _ in range(len(words)):
                if generated >= needed:
                    break
                
                word = words[word_idx % len(words)]
                word_idx += 1
                
                # Get inflections for this word
                inflections = inflection_map.get(word, [])
                
                if not inflections:
                    continue
                
                # Get source row
                source_rows = source_df[source_df[word_col] == word]
                if source_rows.empty:
                    continue
                
                sample_row = source_rows.sample(1).iloc[0]
                
                # Add inflections
                for inflection in inflections[:max_inflections_per_word]:
                    if generated >= needed:
                        break
                    
                    new_row = sample_row.copy()
                    new_row[word_col] = inflection
                    new_row["augmented"] = True
                    new_row["source_word"] = word
                    
                    augmented_rows.append(new_row)
                    generated += 1
        
        log(f"{origin}: Generated {generated}/{needed} augmented samples")
    
    # Combine original and augmented data
    if augmented_rows:
        augmented_df = pd.DataFrame(augmented_rows)
        df = pd.concat([df, augmented_df], ignore_index=True)
        df["augmented"] = df["augmented"].fillna(False)
        log(f"Total augmented samples added: {len(augmented_rows)}")
    else:
        log("No augmented samples generated", "WARNING")
        df["augmented"] = False
    
    # Final distribution
    final_counts = Counter(df[df["direct_origin"].notna()]["direct_origin"])
    log(f"Final distribution: {dict(final_counts)}")
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Map etymology labels and optionally augment dataset"
    )
    parser.add_argument("input",  help="Input CSV file")
    parser.add_argument("output", help="Output CSV file")
    parser.add_argument("--only-loanwords", action="store_true")
    
    # Augmentation
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--jar-path", default="morphology.jar")
    parser.add_argument("--augment-strategy", choices=["upsample", "balance"],
                       default="upsample")
    parser.add_argument("--target-count", type=int)
    parser.add_argument("--max-inflections", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Number of words per batch for inflection")
    
    args = parser.parse_args()

    log(f"Processing input file: {args.input}")
    df = pd.read_csv(args.input)
    stats = Counter()
    stats["rows_total"] = len(df)
    log(f"Loaded {len(df)} rows")

    if "origin" not in df.columns:
        log("CSV must contain 'origin' column", "ERROR")
        return

    # Process origins
    log("Processing origin labels")
    df["origin_l1"] = df["origin"].apply(
        lambda x: remap_origin_value(x, mapping)
    )
    df["origin_l1"] = df["origin_l1"].apply(adjust_non_ide)

    log("Computing loanword labels")
    df["is_loanword"] = df["origin_l1"].apply(lambda x: is_loanword(x))
    df["is_strictly_baltic"] = df["origin_l1"].apply(
        lambda x: is_strictly_baltic(x)
    )

    log("Computing origin precedence")
    df["furthest_origin"] = df["origin_l1"].apply(
        lambda x: pick_origin_by_precedence(x, furthest_precedence)
    )
    df["direct_origin"] = df["origin_l1"].apply(
        lambda x: pick_origin_by_precedence(x, direct_precedence)
    )

    if args.only_loanwords:
        log("Filtering to loanwords only")
        df = df[df["is_loanword"] == True]
        stats["filtered_to_loanwords"] = True
        log(f"After filtering: {len(df)} rows")

    # Augmentation
    if args.augment:
        if not Path(args.jar_path).exists():
            log(f"JAR not found: {args.jar_path}", "ERROR")
            return
        
        log("=" * 60)
        log("STARTING DATA AUGMENTATION")
        log("=" * 60)
        
        df = augment_dataset(
            df,
            args.jar_path,
            strategy=args.augment_strategy,
            target_count=args.target_count,
            max_inflections_per_word=args.max_inflections,
            batch_size=args.batch_size
        )

    log(f"Saving to {args.output}")
    df.to_csv(args.output, index=False)

    # Statistics
    stats["rows_final"] = len(df)
    
    if args.augment:
        stats["rows_augmented"] = int((df.get("augmented", False) == True).sum())
    
    stats["has_loanword_label"] = int(df["is_loanword"].notna().sum())
    stats["has_furthest_label"] = int(df["furthest_origin"].notna().sum())
    stats["has_baltic_label"] = int(df["is_strictly_baltic"].notna().sum())
    
    loanword_true = int((df["is_loanword"] == True).sum())
    loanword_false = int((df["is_loanword"] == False).sum())
    loanword_none = int(df["is_loanword"].isna().sum())
    stats["loanwords"] = {
        "true": loanword_true,
        "false": loanword_false,
        "unknown": loanword_none
    }

    stats["origins_distribution"] = count_origin_distribution(df["origin_l1"])
    stats["furthest_origins_distribution"] = dict(
        Counter(df["furthest_origin"].dropna())
    )
    stats["direct_origins_distribution"] = dict(
        Counter(df["direct_origin"].dropna())
    )

    # Print summary
    log("=" * 60)
    log("PROCESSING SUMMARY")
    log("=" * 60)
    for k, v in stats.items():
        log(f"{k}: {v}")
    log(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()
