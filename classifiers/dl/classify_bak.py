"""
Classifier script for trained origin models (updated for new train_).
"""

import structlog
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from classifiers.dl.train_ import (
    GRUClassifier,
    CNNClassifier,
    TaskConfig,
    TrainConfig,
    ModelConfig,
    char2idx,
    unk,
    device,
    load_model,
    compute_model_hash,
    inspect_loader,
    save_evaluation_results,
    evaluate_binary,
    evaluate_multilabel,
    evaluate_multiclass,
    collect_logits_targets,
    find_best_thresholds,
    ByT5Tokenizer,
)

log = structlog.get_logger()


# === Tokenization Utilities ==================================================


def tokenize(word: str) -> torch.Tensor:
    """Convert word to tensor of character indices."""
    indices = [char2idx.get(ch, unk) for ch in word]
    return torch.tensor(indices, dtype=torch.long)


def pad_sequence(seq: torch.Tensor, max_len: int) -> torch.Tensor:
    """Pad sequence to max_len."""
    padded = torch.zeros(max_len, dtype=torch.long)
    L = min(len(seq), max_len)
    padded[:L] = seq[:L]
    return padded


# === Prediction ==============================================================


def predict(
    model: nn.Module,
    word: str,
    task_config: TaskConfig,
    max_len: int,
    byt5_tokenizer: ByT5Tokenizer = None,
    thresholds=None,
) -> dict:
    """
    Predict origin for a single word.
    Returns dict with prediction details based on task type.
    """
    word = word.lower()

    # Encode input based on model type
    if byt5_tokenizer is None:
        seq = tokenize(word)
        seq = pad_sequence(seq, max_len).unsqueeze(0).to(device)
        attention_mask = None
    else:
        embeddings, attention_mask = byt5_tokenizer.encode_batch([word], max_length=max_len)
        seq = embeddings
        attention_mask = attention_mask

    # Predict
    with torch.no_grad():
        logits = model(seq, attention_mask).squeeze(0)

    result = {"word": word}


    if task_config.task_type == "multilabel":
        probs = torch.sigmoid(logits)

        # if thresholds is not None and len(thresholds) == len(probs):
        #     ref = torch.ones_like(probs) * 0.5
        #     probs = (probs - thresholds) / (0.5 - thresholds + 1e-8) * 0.5 + 0.5
        #     probs = probs.clamp(0, 1)
        if thresholds is not None and len(thresholds) == len(probs):
            # smooth transition around threshold instead of hard cut
            alpha = 0.5  # smaller = smoother ease
            probs = torch.sigmoid((logits - torch.logit(thresholds)) / alpha) 

        idx_to_label = {v: k for k, v in task_config.label_to_idx.items()}
        predictions = [{"label": idx_to_label[i], "prob": p.item()} for i, p in enumerate(probs)]
        predictions.sort(key=lambda x: x["prob"], reverse=True)
        result["predictions"] = predictions
        result["predicted_labels"] = [p["label"] for i, p in enumerate(predictions) if p["prob"] > thresholds[i]]

    elif task_config.task_type == "binary":
        prob = torch.sigmoid(logits).item()
        result["probability"] = prob
        result["prediction"] = prob > thresholds[0]

    elif task_config.task_type == "multiclass":
        probs = torch.softmax(logits, dim=0)
        pred_idx = logits.argmax().item()
        idx_to_label = {v: k for k, v in task_config.label_to_idx.items()}
        result["prediction"] = idx_to_label[pred_idx]
        result["confidence"] = probs[pred_idx].item()
        result["all_probs"] = {idx_to_label[i]: p.item() for i, p in enumerate(probs)}

    return result


# === Output Formatting =======================================================


def format_prediction(result: dict, task_config: TaskConfig) -> str:
    """Format prediction result for display."""
    lines = [f"Word: {result['word']}"]

    if task_config.task_type == "multilabel":
        lines.append(f"Predicted: {', '.join(result['predicted_labels'])}")
        lines.append("\nAll probabilities:")
        for pred in result["predictions"]:
            bar = "█" * int(pred["prob"] * 20)
            lines.append(f"  {pred['label']:12s} {pred['prob']:.3f} {bar}")

    elif task_config.task_type == "binary":
        pred = "True" if result["prediction"] else "False"
        lines.append(f"Prediction: {pred}")
        lines.append(f"Probability: {result['probability']:.3f}")

    elif task_config.task_type == "multiclass":
        lines.append(f"Prediction: {result['prediction']}")
        lines.append(f"Confidence: {result['confidence']:.3f}")
        lines.append("\nAll probabilities:")
        sorted_probs = sorted(result["all_probs"].items(), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_probs:
            bar = "█" * int(prob * 20)
            lines.append(f"  {label:12s} {prob:.3f} {bar}")

    return "\n".join(lines)


# === Interactive Mode ========================================================


def interactive_mode(
    model: nn.Module,
    task_config: TaskConfig,
    max_len: int,
    byt5_tokenizer: ByT5Tokenizer = None,
    thresholds: torch.Tensor | None = None,
):
    """Run interactive command-line prediction (task- and threshold-aware)."""
    print(f"\n{'='*60}")
    print(f"Interactive Mode - Task: {task_config.name}")
    print(f"Type 'quit' or 'exit' to stop")
    print(f"{'='*60}\n")

    # Display info about thresholds, if available
    if thresholds is not None:
        if task_config.task_type == "binary":
            print(f"[Using binary threshold: {thresholds.item():.3f}]")
        elif task_config.task_type == "multilabel":
            thr_values = ", ".join(f"{t:.2f}" for t in thresholds.tolist())
            print(f"[Using multilabel thresholds: {thr_values}]")

    while True:
        try:
            word = input("Enter word: ").strip()
            if word.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break
            if not word:
                continue

            # Normalize and truncate input
            word = word.lower()
            if len(word) > max_len:
                print(f"(truncated to {max_len} characters)")
                word = word[:max_len]

            # Task-aware feedback
            if task_config.task_type == "multiclass":
                print("\n[Task type: Multiclass classification]")
            elif task_config.task_type == "binary":
                print("\n[Task type: Binary classification]")
            elif task_config.task_type == "multilabel":
                print("\n[Task type: Multilabel classification]")

            # Perform prediction
            result = predict(
                model=model,
                word=word,
                task_config=task_config,
                max_len=max_len,
                byt5_tokenizer=byt5_tokenizer,
                thresholds=thresholds,
            )

            # Display formatted prediction
            print(f"\n{format_prediction(result, task_config)}\n")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

# === File Mode ===============================================================


def file_mode(model, task_config, max_len, input_path, output_path, byt5_tokenizer=None):
    """Process a text file with one word per line."""
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    words = [line.strip() for line in input_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Processing {len(words)} words...")

    results = []
    for w in words:
        try:
            results.append(predict(model, w, task_config, max_len, byt5_tokenizer))
        except Exception as e:
            results.append({"word": w, "error": str(e)})

    if output_path:
        write_results(results, task_config, output_path)
        print(f"Results written to {output_path}")
    else:
        for result in results:
            if "error" in result:
                print(f"{result['word']}: ERROR - {result['error']}")
            else:
                print(format_prediction(result, task_config))
                print("-" * 60)


def write_results(results: list[dict], task_config: TaskConfig, output_path: str):
    """Write results to CSV."""
    import csv

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if task_config.task_type == "multilabel":
            writer.writerow(["word", "predicted_labels", "top_label", "top_prob"])
            for r in results:
                if "error" in r:
                    writer.writerow([r["word"], "ERROR", "", ""])
                else:
                    labels = "|".join(r["predicted_labels"])
                    top = r["predictions"][0]
                    writer.writerow([r["word"], labels, top["label"], f"{top['prob']:.3f}"])

        elif task_config.task_type == "binary":
            writer.writerow(["word", "prediction", "probability"])
            for r in results:
                if "error" in r:
                    writer.writerow([r["word"], "ERROR", ""])
                else:
                    writer.writerow([r["word"], r["prediction"], f"{r['probability']:.3f}"])

        elif task_config.task_type == "multiclass":
            writer.writerow(["word", "prediction", "confidence"])
            for r in results:
                if "error" in r:
                    writer.writerow([r["word"], "ERROR", ""])
                else:
                    writer.writerow([r["word"], r["prediction"], f"{r['confidence']:.3f}"])


# === Evaluation Mode =========================================================


def prepare_eval_loader(df, task_config, max_len, batch_size=64):
    """Prepare DataLoader like during training."""
    xs, ys, attn_masks = [], [], []

    use_byt5 = "byt5_tokenizer" in globals() and globals()["byt5_tokenizer"] is not None
    tokenizer = globals().get("byt5_tokenizer", None)

    for word, target in zip(df["word"].astype(str), df[task_config.target_column].astype(str)):

        if not use_byt5:
            seq = torch.tensor([char2idx.get(ch, unk) for ch in word.lower()], dtype=torch.long)
            seq = pad_sequence(seq, max_len)
            xs.append(seq)
            attn_masks.append(None)
        else:
            emb, mask = tokenizer.encode_batch([word.lower()], max_length=max_len)
            xs.append(emb.squeeze(0).cpu())
            attn_masks.append(mask.squeeze(0).cpu())

        if task_config.task_type == "multilabel":
            vec = torch.zeros(len(task_config.label_to_idx), dtype=torch.float32)
            for label in target.split("|"):
                if label.strip() in task_config.label_to_idx:
                    vec[task_config.label_to_idx[label.strip()]] = 1.0
            ys.append(vec)
        elif task_config.task_type == "binary":
            label = 1.0 if target == "True" else 0.0
            ys.append(torch.tensor([label], dtype=torch.float32))
        elif task_config.task_type == "multiclass":
            ys.append(torch.tensor(task_config.label_to_idx.get(target, -1), dtype=torch.long))

    dataset = TensorDataset(torch.stack(xs), torch.stack(ys))
    # align output with train_.py -> include attention_mask (or None)
    if attn_masks[0] is None:
        def collate_fn_no_mask(batch):
            x, y = zip(*batch)
            return torch.stack(x), torch.stack(y), None
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn_no_mask)
    else:
        attn_stack = torch.stack(attn_masks)
        dataset = TensorDataset(torch.stack(xs), torch.stack(ys), attn_stack)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader


def evaluation_mode(
    model,
    task_config,
    max_len,
    thresholds,
    input_path,
    label_column=None,
    output_path=None,
):
    df = pd.read_csv(input_path)
    label_column = label_column or task_config.target_column
    if "word" not in df.columns or label_column not in df.columns:
        log.error("CSV must contain columns: 'word' and label")
        sys.exit(1)

    eval_loader = prepare_eval_loader(df, task_config, max_len)
    # inspect_loader(eval_loader, task_config, num_samples=10)

    log.info(f"Evaluating {len(df)} words from {input_path}")

    if task_config.task_type == "binary":
        acc, prec, rec, f1 = evaluate_binary(model, eval_loader, device, threshold=thresholds[0])
        log.info(f"Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

    elif task_config.task_type == "multilabel":
        logits, targets = collect_logits_targets(model, eval_loader, device)
        best_t, _ = find_best_thresholds(logits, targets, task_config.task_type, beta=0.7)
        prec, rec, fb = evaluate_multilabel(model, eval_loader, device, best_t, beta=0.7)
        log.info(f"Precision: {prec}")
        log.info(f"Recall: {rec}")
        log.info(f"Fβ=0.7: {fb}")

    elif task_config.task_type == "multiclass":
        acc, prec, rec, f1_scores = evaluate_multiclass(model, eval_loader, device, thresholds=thresholds)
        log.info(f"Accuracy: {acc:.4f}")
        log.info(f"Precision: {prec}")
        log.info(f"Recall: {rec}")
        log.info(f"F1 per class: {f1_scores}")

    dataset_name = Path(input_path).stem.split("_")[-1]
    output_path = Path(output_path or f"results_{dataset_name}.csv")
    save_evaluation_results(model, eval_loader, task_config, output_path, device)


# === Main ====================================================================


def main():
    parser = argparse.ArgumentParser(description="Classify word origins using trained model")
    parser.add_argument("model", help="Path to model checkpoint")

    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    modes.add_argument("-f", "--file", help="Input file with one word per line")
    modes.add_argument("-e", "--evaluate", help="CSV with words and true labels")

    parser.add_argument("--label-column", help="Label column for evaluation mode")
    parser.add_argument("-o", "--output", help="Output CSV path (for file/eval mode)")

    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    model, checkpoint, byt5_tokenizer = load_model(args.model)
    task_config = checkpoint["task_config"]
    train_config = checkpoint["train_config"]
    model_config = checkpoint["model_config"]
    thresholds = checkpoint["thresholds"]
    max_len = train_config.max_seq_len

    current_hash = compute_model_hash(model)
    original_hash = checkpoint["model_hash"]
    log.info(f"Model hash (current):  {current_hash}")
    log.info(f"Model hash (checkpoint): {original_hash}")
    log.info(f"Match: {current_hash == original_hash}")
    log.info(f"Thresholds: {thresholds}")

    print(f"Task: {task_config.name} ({task_config.task_type})")
    print(f"Device: {device}")

    if args.interactive:
        interactive_mode(model, task_config, max_len, byt5_tokenizer, thresholds=thresholds)
    elif args.file:
        file_mode(model, task_config, max_len, args.file, args.output, byt5_tokenizer)
    elif args.evaluate:
        evaluation_mode(model, task_config, max_len, thresholds, args.evaluate, args.label_column, args.output)
        log.info(f"Model config: {model_config}")
        log.info(f"Task config: {task_config}")
        log.info(f"Train config: {train_config}")
        log.info(f"Labels: {task_config.label_to_idx}")


if __name__ == "__main__":
    main()
