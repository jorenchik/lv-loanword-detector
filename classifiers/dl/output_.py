
from pathlib import Path
from datetime import datetime
import pandas as pd
import torch 

from classifiers.dl.logging_ import (
    log,
)
from classifiers.dl.models_ import (
    evaluate_multiclass,
    evaluate_binary,
    evaluate_multilabel,
    evaluate_charlm,
)
from classifiers.dl.task_config import (
    TaskConfig,
)

def setup_output_dir(base_name: str) -> Path:

    """Create timestamped output directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{base_name}_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "datasets").mkdir(parents=True, exist_ok=True)
    
    return output_dir

def save_evaluation_results(
    model, loader, task_config, output_path, device
):
    """Save detailed evaluation results to CSV."""
    model.eval()
    
    # Reverse char mapping for decoding
    idx2char = {i: c for c, i in char2idx.items()}
    idx2char[unk] = "<UNK>"
    
    results = []
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # if attention_mask is not None:
            #     attention_mask = attention_mask.to(device)

            logits = model(x_batch)
            
            for x, y, logit in zip(x_batch, y_batch, logits):

                # Decode word
                if x.dim() == 1:
                    # Traditional: x is token indices
                    chars = [idx2char[idx.item()] for idx in x if idx.item() != 0]
                    word = "".join(chars)
                else:
                    # ByT5: x is embeddings, can't decode back to text
                    word = "N/A"
                
                row = {"word": word}
                
                if task_config.task_type == "binary":
                    prob = torch.sigmoid(logit).item()
                    pred = prob >= 0.5
                    true_label = "True" if y.item() >= 0.5 else "False"
                    
                    row["true_label"] = true_label
                    row["predicted_label"] = "True" if pred else "False"
                    row["probability"] = f"{prob:.4f}"
                    row["correct"] = (pred == (y.item() >= 0.5))
                
                elif task_config.task_type == "multiclass":
                    idx2label = {v: k for k, v in task_config.label_to_idx.items()}
                    probs = torch.softmax(logit, dim=0)
                    pred_idx = logit.argmax().item()
                    
                    row["true_label"] = idx2label.get(y.item(), "UNKNOWN")
                    row["predicted_label"] = idx2label[pred_idx]
                    row["confidence"] = f"{probs[pred_idx].item():.4f}"
                    row["correct"] = (pred_idx == y.item())
                
                elif task_config.task_type == "multilabel":
                    idx2label = {v: k for k, v in task_config.label_to_idx.items()}
                    probs = torch.sigmoid(logit)
                    preds = (probs >= 0.5).int()
                    
                    true_labels = [idx2label[i] for i, val in enumerate(y) if val > 0.5]
                    pred_labels = [idx2label[i] for i, val in enumerate(preds) if val > 0.5]
                    
                    row["true_labels"] = "|".join(true_labels) if true_labels else "NONE"
                    row["predicted_labels"] = "|".join(pred_labels) if pred_labels else "NONE"
                    row["correct"] = (preds == y.int()).all().item()
                
                results.append(row)
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    log.info(f"Saved {len(results)} predictions to {output_path}")
    
    # Print summary
    if "correct" in df.columns:
        accuracy = df["correct"].mean()
        log.info(f"Overall accuracy: {accuracy:.4f}")

def save_test_metrics(
    task_config: TaskConfig,
    metrics: dict,
    output_path: Path
):
    """Save test metrics to CSV."""
    
    if task_config.task_type == "binary":
        # Single row for binary classification
        df = pd.DataFrame([{
            'metric': 'overall',
            'accuracy': f"{metrics['accuracy']:.4f}",
            'precision': f"{metrics['precision']:.4f}",
            'recall': f"{metrics['recall']:.4f}",
            'f1': f"{metrics['f1']:.4f}",
        }])
    
    elif task_config.task_type == "multiclass":
        # One row per class + overall accuracy
        idx2label = {v: k for k, v in task_config.label_to_idx.items()}
        rows = []
        
        # Overall accuracy row
        rows.append({
            'class': 'overall',
            'accuracy': f"{metrics['accuracy']:.4f}",
            'precision': '-',
            'recall': '-',
            'f1': '-',
        })
        
        # Per-class metrics
        for i, label in sorted(idx2label.items()):
            rows.append({
                'class': label,
                'accuracy': '-',
                # 'precision': f"{metrics['precision'][i]:.4f}",
                # 'recall': f"{metrics['recall'][i]:.4f}",
                # 'f1': f"{metrics['f1'][i]:.4f}",
            })
        
        df = pd.DataFrame(rows)
    
    elif task_config.task_type == "multilabel":
        # One row per label
        idx2label = {v: k for k, v in task_config.label_to_idx.items()}
        rows = []
        
        for i, label in sorted(idx2label.items()):
            rows.append({
                'label': label,
                'precision': f"{metrics['precision'][i]:.4f}",
                'recall': f"{metrics['recall'][i]:.4f}",
                'f_beta': f"{metrics['f_beta'][i]:.4f}",
            })
        
        df = pd.DataFrame(rows)
    
    df.to_csv(output_path, index=False)
    log.info(f"Saved test metrics to {output_path}")


def output_evaluation(model, train_config, task_config, loader, device, thresholds, output_dir):

    metrics_file = output_dir / "test_metrics.csv"

    if task_config.task_type == "multilabel":

        prec, rec, fb = evaluate_multilabel(
            model, loader, device, thresholds, train_config.beta
        )
        log.info(f"Test - Labels: {task_config.label_to_idx}")
        log.info(f"Test - Precision: {prec}")
        log.info(f"Test - Recall: {rec}")
        log.info(f"Test - F{train_config.beta}: {fb}")
        save_test_metrics(
            task_config,
            {
                'precision': prec.cpu().numpy(),
                'recall': rec.cpu().numpy(),
                'f_beta': fb.cpu().numpy(),
            },
            metrics_file
        )

    elif task_config.task_type == "charlm":

        log.info(f"Best epoch: {best_epoch}, dev_loss: {best_dev_loss:.4f}, dev_perplexity: {best_dev_perplexity:.2f}")
        test_loss, test_perplexity, test_bpc = evaluate_charlm(
            model, test_loader, criterion, device
        )
        log.info(
            f"Test set evaluation for CharLM: "
            f"Loss={test_loss:.4f}, Perplexity={test_perplexity:.2f}, BPC={test_bpc:.2f}"
        )

    elif task_config.task_type == "binary":

        acc, prec, rec, f1 = evaluate_binary(model, loader, device, thresholds[0])
        log.info(
            f"TEST - Acc: {acc:.4f}, Prec: {prec:.4f}, "
            f"Rec: {rec:.4f}, F1: {f1:.4f}"
        )
        save_test_metrics(
            task_config,
            {
                'accuracy': acc.item() if torch.is_tensor(acc) else acc,
                'precision': prec.item() if torch.is_tensor(prec) else prec,
                'recall': rec.item() if torch.is_tensor(rec) else rec,
                'f1': f1.item() if torch.is_tensor(f1) else f1,
            },
            metrics_file
        )

        log.info(f"Task config: {task_config}")
        log.info(f"Train config: {train_config}")
        log.info(f"Label mapping: {task_config.label_to_idx}")

    elif task_config.task_type == "multiclass":

        acc, prec, rec, f1_scores = evaluate_multiclass(model, loader, device)
        save_test_metrics(
            task_config,
            {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1_scores,
            },
            metrics_file
        )
        log.info(f"Test - Accuracy: {acc:.4f}")
        log.info(f"Test - Labels: {task_config.label_to_idx}")
        log.info(f"Test - Precision per class: {prec}")
        log.info(f"Test - Recall per class: {rec}")
        log.info(f"Test - F1 per class: {f1_scores}")
