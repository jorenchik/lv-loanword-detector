from datasets import load_dataset

# Choose a language, e.g. French ("fr")
lang = "fr"
dataset = load_dataset("oscar-corpus/OSCAR-2301", lang, split="train")

# Save to plain text
with open(f"oscar_{lang}.txt", "w", encoding="utf-8") as f:
    for sample in dataset:
        text = sample["text"].replace("\n", " ").strip()
        f.write(text + "\n")
