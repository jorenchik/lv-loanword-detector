
from torch.utils.data import DataLoader, TensorDataset, Dataset
from classifiers.dl.task_config import (
    TaskConfig
)
import pandas as pd
import torch

# == Alphabet

LV_ALPHABET      = "aābcčdeēfgģhiījkķlļnņoprsštuūvzž"
FOREIGN_LETTERS  = "qwxyòóôöüäéèêçñ"
ADD_SYMBOLS      = "-'"
ALPHABET         = sorted(LV_ALPHABET + FOREIGN_LETTERS + ADD_SYMBOLS)
# char → index mapping
# reserve last index for <UNK>
char2idx = {c: i for i, c in enumerate(ALPHABET)}
unk = len(ALPHABET)


class WordDataset(Dataset):

    def __init__(self, datapath: str, task_config: TaskConfig, use_byt5=False):

        super().__init__()

        words = []

        with open(datapath, encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip() if len(line) >= 2]

        self.x = []
        for word in words:
            word = word.lower()
            encoded = torch.tensor(
                [char2idx.get(ch, unk) for ch in word], dtype=torch.long
            )
            self.x.append(encoded)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]

class OriginDataset(Dataset):

    def __init__(self, datapath: str, task_config: TaskConfig, use_byt5=False):

        super().__init__()

        df = pd.read_csv(datapath)
        words = df["word"].astype(str).tolist()

        self.use_byt5 = use_byt5
        
        if use_byt5:
            self.x = words  # Keep as strings
        else:
            self.x = []
            for word in words:
                word = word.lower()
                enc = torch.tensor(
                    [char2idx.get(ch, unk) for ch in word], dtype=torch.long
                )
                self.x.append(enc)

        # Encode labels based on task type
        self.y = []
        targets = df[task_config.target_column].astype(str).tolist()

        if task_config.task_type == "multilabel":
            num_classes = len(task_config.label_to_idx)
            for target in targets:
                vec = torch.zeros(num_classes, dtype=torch.float32)
                for label in target.split("|"):
                    label = label.strip()
                    if label in task_config.label_to_idx:
                        vec[task_config.label_to_idx[label]] = 1.0
                self.y.append(vec)

        elif task_config.task_type == "binary":
            for target in targets:
                label = 1.0 if target == "True" else 0.0
                self.y.append(torch.tensor([label], dtype=torch.float32))

        elif task_config.task_type == "multiclass":
            for target in targets:
                if target in task_config.label_to_idx:
                    idx = task_config.label_to_idx[target]
                else:
                    idx = -1  # ignore in loss
                self.y.append(torch.tensor(idx, dtype=torch.long))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

def collate_lm(batch, max_len: int, use_byt5 = False):

    B = len(batch)
    padded = torch.zeros(B, max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        L = min(len(seq), max_len)
        padded[i, :L] = seq[:L]

    x_out = padded

    return x_out

def collate(batch, max_len: int, byt5_tokenizer=None):
    xs, ys = zip(*batch)

    if byt5_tokenizer is None:
        B = len(xs)
        padded = torch.zeros(B, max_len, dtype=torch.long)
        for i, seq in enumerate(xs):
            L = min(len(seq), max_len)
            padded[i, :L] = seq[:L]
        x_out = padded
        attention_mask = None
    else:
        x_out = byt5_tokenizer.encode_batch(list(xs), max_length=max_len)

    # Stack depends on task type
    if ys[0].dim() == 0:  # scalar (multiclass)
        ys = torch.stack(ys)
    else:  # vector (multilabel or binary)
        ys = torch.stack(ys)

    # return x_out, ys, attention_mask
    return x_out, ys

def inspect_loader(loader, task_config, num_samples=10):
    """Extract and display first N samples from loader."""

    total = 0
    total_true = 0
    total_false = 0
    for x_batch, y_batch, _ in loader:
        total += x_batch.size(0)
        total_true += (y_batch >= 0.5).sum().item()
        total_false += (y_batch < 0.5).sum().item()
    print(f"Total length: {total}")
    print(f"True: {total_true}, False: {total_false}")
    print(f"Ratio True: {total_true/total:.3f}")

    # Reverse char mapping
    idx2char = {i: c for c, i in char2idx.items()}
    idx2char[unk] = "<UNK>"
    
    samples = []
    for x_batch, y_batch, _ in loader:
        for x, y in zip(x_batch, y_batch):
            if len(samples) >= num_samples:
                break
            
            # Decode word (only for non-ByT5 models)
            if x.dim() == 1:
                # Traditional: x is token indices
                chars = [idx2char[idx.item()] for idx in x if idx.item() != 0]
                word = "".join(chars)
            else:
                # ByT5: x is embeddings, can't decode back to text
                word = "N/A"

            # Decode label based on task type
            if task_config.task_type == "binary":
                label = "True" if y.item() >= 0.5 else "False"
            elif task_config.task_type == "multiclass":
                idx2label = {v: k for k, v in task_config.label_to_idx.items()}
                label = idx2label.get(y.item(), "UNKNOWN")
            elif task_config.task_type == "multilabel":
                idx2label = {v: k for k, v in task_config.label_to_idx.items()}
                active = [idx2label[i] for i, val in enumerate(y) if val > 0.5]
                label = "|".join(active) if active else "NONE"
            
            samples.append((word, label))
        
        if len(samples) >= num_samples:
            break
    
    # Display
    print(f"\n{'='*60}")
    print(f"First {len(samples)} samples from loader:")
    print(f"{'='*60}")
    for i, (word, label) in enumerate(samples, 1):
        print(f"{i:2d}. {word:20s} → {label}")
    print(f"{'='*60}\n")
    
    return samples

def compute_pos_weights(loader, device, task_type: str):

    """Only for multilabel/binary tasks."""
    if task_type not in ["multilabel", "binary"]:
        return None

    total_pos = None
    total_count = 0

    for _, y in loader:
        y = y.to(device)
        if total_pos is None:
            total_pos = y.sum(dim=0)
        else:
            total_pos += y.sum(dim=0)
        total_count += y.size(0)

    pos = total_pos
    neg = total_count - pos
    return neg / (pos + 1e-8)

def compute_class_weights(task_config, loader, device):
    """Compute per-class weights aligned with task_config.label_to_idx."""
    num_classes = len(task_config.label_to_idx)
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, y in loader:
        mask = y >= 0
        valid_y = y[mask]
        for idx in valid_y:
            counts[idx.item()] += 1
    counts = torch.where(counts == 0, torch.ones_like(counts), counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return weights.to(device)
