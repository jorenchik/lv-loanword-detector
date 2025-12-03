import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from classifiers.dl.models_ import (
    GRUClassifier,
    CNNClassifier,
    ByT5Tokenizer,
    load_model,
    load_charlm_encoder,
    compute_model_hash,
    evaluate_binary,
    evaluate_multilabel,
    evaluate_multiclass,
    collect_logits_targets,
    find_best_thresholds,
)
from classifiers.dl.task_config import TaskConfig
from classifiers.dl.torch_config import device
from classifiers.dl.logging_ import log
from classifiers.dl.data_ import (
    OriginDataset,
    collate
)

def tokenize(word: str, char2idx: dict[str, int], unk: int) -> torch.Tensor:
    indices = [char2idx.get(ch, unk) for ch in word]
    return torch.tensor(indices, dtype=torch.long)

def pad_sequence(seq: torch.Tensor, max_len: int) -> torch.Tensor:
    padded = torch.zeros(max_len, dtype=torch.long)
    L = min(len(seq), max_len)
    padded[:L] = seq[:L]
    return padded

def predict(
    model: nn.Module,
    word: str,
    task_config: TaskConfig,
    max_len: int,
    char2idx: dict[str, int],
    unk: int,
    byt5_tokenizer: ByT5Tokenizer | None = None,
    thresholds: torch.Tensor | None = None,
) -> dict:
    model.eval()
    word = word.lower()

    if byt5_tokenizer is not None:
        seq = byt5_tokenizer.encode_batch([word], max_length=max_len)
    else:
        seq = pad_sequence(tokenize(word, char2idx, unk), max_len).unsqueeze(0)

    seq = seq.to(device)
    with torch.no_grad():
        logits = model(seq)
        if logits.ndim == 2 and logits.shape[0] == 1:
            logits = logits.squeeze(0)

    result = {"word": word}

    if task_config.task_type == "multilabel":
        probs = torch.sigmoid(logits)
        idx_to_label = {v: k for k, v in task_config.label_to_idx.items()}
        predictions = [{"label": idx_to_label[i], "prob": p.item()} for i, p in enumerate(probs)]
        predictions.sort(key=lambda x: x["prob"], reverse=True)
        best_labels = []
        if thresholds is not None and len(thresholds) == len(probs):
            for i, p in enumerate(probs):
                if p.item() >= thresholds[i].item():
                    best_labels.append(idx_to_label[i])
        result["predictions"] = predictions
        result["predicted_labels"] = best_labels

    elif task_config.task_type == "binary":
        prob = torch.sigmoid(logits).item()
        threshold = 0.5 if thresholds is None else thresholds[0].item()
        result["probability"] = prob
        result["prediction"] = prob >= threshold

    elif task_config.task_type == "multiclass":
        probs = torch.softmax(logits, dim=0)
        pred_idx = probs.argmax().item()
        idx_to_label = {v: k for k, v in task_config.label_to_idx.items()}
        result["prediction"] = idx_to_label[pred_idx]
        result["confidence"] = probs[pred_idx].item()
        result["all_probs"] = {idx_to_label[i]: p.item() for i, p in enumerate(probs)}

    return result

def format_prediction(result: dict, task_config: TaskConfig) -> str:
    lines = [f"Word: {result['word']}"]
    if task_config.task_type == "multilabel":
        lines.append(f"Predicted: {', '.join(result['predicted_labels'])}")
        lines.append("All probabilities:")
        for p in result["predictions"]:
            bar = "█" * int(p["prob"] * 20)
            lines.append(f"  {p['label']:12s} {p['prob']:.3f} {bar}")
    elif task_config.task_type == "binary":
        pred = "True" if result["prediction"] else "False"
        lines.append(f"Prediction: {pred}")
        lines.append(f"Probability: {result['probability']:.3f}")
    elif task_config.task_type == "multiclass":
        lines.append(f"Prediction: {result['prediction']}")
        lines.append(f"Confidence: {result['confidence']:.3f}")
        lines.append("All probabilities:")
        for label, prob in sorted(result["all_probs"].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 20)
            lines.append(f"  {label:12s} {prob:.3f} {bar}")
    return "\n".join(lines)

def interactive_mode(model, task_config, char2idx, unk, max_len, byt5_tokenizer, thresholds):
    print("=" * 60)
    print(f"Interactive Mode — Task: {task_config.name}")
    print("Type 'quit' or 'exit' to terminate.")
    print("=" * 60)
    while True:
        try:
            word = input("\nEnter word: ").strip()
            if not word:
                continue
            if word.lower() in {"quit", "exit"}:
                print("Goodbye.")
                break
            result = predict(
                model,
                word,
                task_config,
                max_len,
                char2idx,
                unk,
                byt5_tokenizer,
                thresholds,
            )
            print(format_prediction(result, task_config))
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as exc:
            print(f"Error: {exc}")

# Add this function after format_prediction()
def charlm_interactive_mode(model, max_len, char2idx, unk, temperature=1.0):
    print("=" * 60)
    print("CharLM Generation Mode")
    print("Type 'quit' or 'exit' to terminate.")
    print("=" * 60)
    while True:
        try:
            prompt = input("\nEnter prompt: ").strip()
            if not prompt:
                continue
            if prompt.lower() in {"quit", "exit"}:
                print("Goodbye.")
                break
            generated = model.generate(prompt, char2idx, unk, max_len=max_len, temperature=temperature)
            print(f"Generated: {generated}")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as exc:
            print(f"Error: {exc}")

def charlm_file_mode(model, input_path, output_path, max_len, temperature=1.0):
    prompts = [
        line.strip()
        for line in Path(input_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = []
    for prompt in prompts:
        try:
            generated = model.generate(prompt, max_len=max_len, temperature=temperature)
            results.append({"prompt": prompt, "generated": generated})
        except Exception as e:
            results.append({"prompt": prompt, "generated": f"ERROR: {e}"})

    if output_path:
        import csv
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["prompt", "generated"])
            writer.writeheader()
            writer.writerows(results)
        print(f"Results written to {output_path}")
    else:
        for r in results:
            print(f"{r['prompt']} → {r['generated']}")

def charlm_evaluation_mode(model, eval_loader, task_config, device):
    avg_loss, perplexity, bpc = evaluate_charlm(model, eval_loader, task_config, device)
    log.info(f"Loss: {avg_loss:.4f}")
    log.info(f"Perplexity: {perplexity:.2f}")
    log.info(f"BPC: {bpc:.4f}")

def file_mode(
    model,
    task_config,
    char2idx,
    unk,
    max_len,
    input_path,
    output_path,
    byt5_tokenizer,
    thresholds,
):
    words = [
        line.strip()
        for line in Path(input_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = []
    for w in words:
        try:
            res = predict(model, w, task_config, max_len, char2idx, unk, byt5_tokenizer, thresholds)
            results.append(res)
        except Exception as e:
            results.append({"word": w, "error": str(e)})

    import csv

    if output_path:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if task_config.task_type == "binary":
                writer.writerow(["word", "prediction", "prob"])
                for r in results:
                    writer.writerow(
                        [r["word"], r.get("prediction", "ERROR"), f"{r.get('probability', 0):.3f}"]
                    )
            elif task_config.task_type == "multilabel":
                writer.writerow(["word", "predicted_labels"])
                for r in results:
                    writer.writerow([r["word"], "|".join(r.get("predicted_labels", []))])
            else:
                writer.writerow(["word", "prediction", "confidence"])
                for r in results:
                    writer.writerow([r["word"], r.get("prediction", "ERROR"), f"{r.get('confidence', 0):.3f}"])
        print(f"Results written to {output_path}")
    else:
        for r in results:
            if "error" in r:
                print(f"{r['word']}: ERROR — {r['error']}")
            else:
                print(format_prediction(r, task_config))
                print("-" * 60)

def evaluation_mode(model, task_config, eval_loader, thresholds):

    if task_config.task_type == "binary":

        acc, prec, rec, f1 = evaluate_binary(model, eval_loader, device, threshold=thresholds[0])
        log.info(f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    elif task_config.task_type == "multilabel":

        logits, targets = collect_logits_targets(model, eval_loader, device)
        best_t, _ = find_best_thresholds(logits, targets, "multilabel")
        prec, rec, f1s = evaluate_multilabel(model, eval_loader, device, best_t)
        log.info(f"Precision: {prec}")
        log.info(f"Recall: {rec}")
        log.info(f"F1: {f1s}")

    elif task_config.task_type == "multiclass":

        acc, prec, rec, f1s = evaluate_multiclass(model, eval_loader, device)
        log.info(f"Accuracy={acc:.4f}")
        log.info(f"Precision={prec}")
        log.info(f"Recall={rec}")
        log.info(f"F1 per class={f1s}")

def main():
    parser = argparse.ArgumentParser(description="Predict/generate using trained models.")
    parser.add_argument("model", help="Path to model .pt file")
    
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    mode.add_argument("-f", "--file", help="Input file (words or prompts)")
    mode.add_argument("-e", "--eval", help="Evaluate on dataset (path to txt/csv)")
    
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    parser.add_argument("--max-gen-len", type=int, default=35, help="Max generation length")
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.model}")
    raw_ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model_config = raw_ckpt["model_config"]
    task_config = raw_ckpt["task_config"]
    train_config = raw_ckpt["train_config"]
    char2idx = raw_ckpt["char2idx"]
    thresholds = raw_ckpt.get("thresholds", None)
    unk = char2idx.get("<UNK>", max(char2idx.values()))

    # Load CharLM encoder if needed
    charlm_encoder = None
    if model_config.use_charlm:
        if "charlm_state" not in raw_ckpt or "charlm_config" not in raw_ckpt:
            raise ValueError("Model uses CharLM but checkpoint doesn't contain CharLM state")
        log.info("Loading CharLM encoder from checkpoint")
        charlm_config = raw_ckpt["charlm_config"]
        charlm_encoder = CharLanguageModel(charlm_config).to(device)
        charlm_encoder.load_state_dict(raw_ckpt["charlm_state"])
        for param in charlm_encoder.parameters():
            param.requires_grad = False

    model, checkpoint, byt5_tokenizer = load_model(args.model, charlm_encoder)
    
    log.info(f"Model hash: {compute_model_hash(model)}")
    print(f"Task: {task_config.name} ({task_config.task_type})")
    print(f"Device: {device}")

    # CharLM-specific modes
    if task_config.task_type == "generative":
        if args.eval:
            # Evaluation mode
            dataset = WordDataset(args.eval, task_config)
            eval_loader = DataLoader(
                dataset,
                batch_size=train_config.batch_size,
                shuffle=False,
                collate_fn=lambda b: collate_lm(b, train_config.max_seq_len),
            )
            charlm_evaluation_mode(model, eval_loader, task_config, device)
        elif args.interactive:
            charlm_interactive_mode(model, args.max_gen_len, char2idx, args.temperature)
        elif args.file:
            output_path = Path(args.output) if args.output else None
            charlm_file_mode(model, args.file, output_path, args.max_gen_len, args.temperature)
        return

    # Classification modes (existing logic)
    if args.eval:
        dataset = OriginDataset(args.eval, task_config, use_byt5=model_config.use_byt5)
        eval_loader = DataLoader(
            dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate(b, train_config.max_seq_len, byt5_tokenizer),
        )
        evaluation_mode(model, task_config, eval_loader, thresholds)
    elif args.interactive:
        interactive_mode(model, task_config, char2idx, unk, train_config.max_seq_len, byt5_tokenizer, thresholds)
    elif args.file:
        output_path = Path(args.output) if args.output else None
        file_mode(model, task_config, char2idx, unk, train_config.max_seq_len, args.file, output_path, byt5_tokenizer, thresholds)


if __name__ == "__main__":
    main()
