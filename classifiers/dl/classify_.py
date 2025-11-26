import argparse
from pathlib import Path
import torch
import torch.nn as nn
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

    parser = argparse.ArgumentParser(description="Predict word origins using trained models.")
    parser.add_argument("model", help="Path to classifier model .pt file")
    parser.add_argument("--charlm", help="Path to pretrained CharLM encoder if required", type=str)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    mode.add_argument("-f", "--file", help="File containing one word per line")
    parser.add_argument("-o", "--output", help="Output CSV path for file mode")
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.model}")
    raw_ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model_config = raw_ckpt["model_config"]
    task_config = raw_ckpt["task_config"]
    train_config = raw_ckpt["train_config"]
    char2idx = raw_ckpt["char2idx"]
    thresholds = raw_ckpt.get("thresholds", None)
    unk = char2idx.get("<UNK>", max(char2idx.values()))

    # Load CharLM encoder if the model requires it.
    charlm_encoder = None
    if model_config.use_charlm:
        charlm_path = args.charlm
        if not charlm_path:
            raise ValueError(
                "This model requires a CharLM encoder. Provide it via --charlm path/to/charlm.pt"
            )
        log.info(f"Loading CharLM encoder from {charlm_path}")
        charlm_encoder, _ = load_charlm_encoder(charlm_path, device, freeze=True)

    # Load model properly.
    model, checkpoint, byt5_tokenizer = load_model(args.model, charlm_encoder)

    log.info(f"Model hash: {compute_model_hash(model)}")
    print(f"Task: {task_config.name} ({task_config.task_type})")
    print(f"Device: {device}")

    # Interactive / File.
    if args.interactive:
        interactive_mode(
            model,
            task_config,
            char2idx,
            unk,
            train_config.max_seq_len,
            byt5_tokenizer,
            thresholds,
        )
    elif args.file:
        output_path = Path(args.output) if args.output else None
        file_mode(
            model,
            task_config,
            char2idx,
            unk,
            train_config.max_seq_len,
            args.file,
            output_path,
            byt5_tokenizer,
            thresholds,
        )

if __name__ == "__main__":
    main()
