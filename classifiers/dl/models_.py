
import heapq
from pathlib import Path
import torch
import torch.nn as nn
from classifiers.dl.model_config import (
    ModelConfig
)
from classifiers.dl.task_config import (
    TaskConfig 
)
from classifiers.dl.torch_config import (
    device
)
from transformers import (
    T5EncoderModel,
    AutoTokenizer,
)

class ByT5Tokenizer:
    """ByT5 tokenizer and encoder for character-level embeddings."""
    
    def __init__(self, model_name="google/byt5-small", device="cuda", freeze=True):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name).to(device)
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.embedding_dim = self.encoder.config.d_model
    
    def encode_batch(self, words, max_length=None):
        """
        Encode batch of words to embeddings.
        Returns: (batch_size, seq_len, embedding_dim)
        """
        inputs = self.tokenizer(
            words,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)
        
        with torch.set_grad_enabled(not self._is_frozen()):
            outputs = self.encoder(**inputs)
            embeddings = outputs.last_hidden_state
        
        # return embeddings, inputs.attention_mask
        return embeddings
    
    def _is_frozen(self):
        return not next(self.encoder.parameters()).requires_grad


class CharLanguageModel(nn.Module):
    """
    Simple generative char-level language model.
    Can be used as an alternative model option (--model charlm).
    Trained to predict next character in sequence.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.rnn = nn.GRU(
            config.embed_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.rnn_dropout if config.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc_out = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.rnn(emb, hidden)
        out = self.dropout(out)
        logits = self.fc_out(out)
        return logits, hidden

    def encode(self, x):
        """Extract RNN hidden states for use as encoder."""
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        return out  # (batch, seq_len, hidden_dim)

    def generate(self, start_seq, max_len=100, temperature=1.0):
        """
        Generate text starting from a given character sequence.
        """
        self.eval()
        idx2char = {i: c for c, i in char2idx.items()}
        idx2char[unk] = "<UNK>"

        input_seq = torch.tensor(
            [char2idx.get(ch, unk) for ch in start_seq], dtype=torch.long
        ).unsqueeze(0).to(next(self.parameters()).device)

        hidden = None
        generated = list(start_seq)

        for _ in range(max_len):
            logits, hidden = self.forward(input_seq, hidden)
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = idx2char.get(next_idx, "<UNK>")
            if next_char == "<UNK>":
                break
            generated.append(next_char)
            input_seq = torch.tensor([[next_idx]], device=input_seq.device)

        return "".join(generated)

class GRUClassifier(nn.Module):

    def __init__(self, config: ModelConfig, output_dim: int, charlm_encoder=None):

        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.charlm_encoder = charlm_encoder
        
        if config.use_charlm:
            self.embedding = None
            self.byt5_projection = None
            input_dim = config.charlm_hidden_dim
        elif config.use_byt5:
            self.embedding = None
            self.byt5_projection = nn.Linear(config.byt5_dim, config.embed_dim)
            input_dim = config.embed_dim
        else:
            input_dim = config.embed_dim
            self.byt5_projection = None
            input_dim = config.embed_dim

        self.rnn = nn.GRU(
            input_dim,
            config.hidden_dim,
            batch_first=True,
            num_layers=config.num_layers,
            dropout=config.rnn_dropout if config.num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_dim * 2, output_dim)

    def forward(self, x):
        if self.charlm_encoder is not None:
            with torch.set_grad_enabled(self.charlm_encoder.training):
                emb = self.charlm_encoder.encode(x)
        elif self.embedding is not None:
            emb = self.embedding(x)
        else:
            emb = self.byt5_projection(x)
        _, h_last = self.rnn(emb)
        # Concatenate forward and backward final states
        h_fwd = h_last[-2]
        h_bwd = h_last[-1]
        h = torch.cat([h_fwd, h_bwd], dim=1)
        h = self.dropout(h)
        return self.fc(h)

class CNNClassifier(nn.Module):
    def __init__(self, config: ModelConfig, output_dim: int, charlm_encoder=None):
        super().__init__()

        self.charlm_encoder = charlm_encoder
        
        if config.use_charlm:
            self.embedding = None
            self.byt5_projection = None
            conv_input_dim = config.charlm_hidden_dim
        elif config.use_byt5:
            self.embedding = None
            self.byt5_projection = nn.Linear(config.byt5_dim, config.embed_dim)
            conv_input_dim = config.embed_dim
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
            self.byt5_projection = None
            conv_input_dim = config.embed_dim

        kernel_sizes = config.kernel_sizes or [3, 4, 5]
        if config.num_filters:
            if isinstance(config.num_filters, int):
                num_filters = [config.num_filters] * len(kernel_sizes)
            else:
                num_filters = config.num_filters
        else:
            num_filters = [config.hidden_dim] * len(kernel_sizes)

        self.convs = nn.ModuleList([
            nn.Conv1d(conv_input_dim, nf, kernel_size=k)
            for k, nf in zip(kernel_sizes, num_filters)
        ])

        if config.use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(nf) for nf in num_filters
            ])
        else:
            self.batch_norms = None

        self.activation = getattr(torch.nn.functional, config.activation)

        fc_input_dim = sum(num_filters)
        fc_dims = config.fc_layers or []
        fc_layers = []
        prev_dim = fc_input_dim
        for hidden in fc_dims:
            fc_layers.append(nn.Linear(prev_dim, hidden))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden
        fc_layers.append(nn.Linear(prev_dim, output_dim))
        self.fc = nn.Sequential(*fc_layers)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):

        if self.charlm_encoder is not None:
            with torch.set_grad_enabled(self.charlm_encoder.training):
                emb = self.charlm_encoder.encode(x).transpose(1, 2)
        elif self.embedding is not None:
            emb = self.embedding(x).transpose(1, 2)
        else:
            emb = self.byt5_projection(x).transpose(1, 2)

        conv_outs = []
        for i, conv in enumerate(self.convs):
            out = conv(emb)
            if self.batch_norms:
                out = self.batch_norms[i](out)
            out = self.activation(out)
            out = out.max(dim=2)[0]  # Global max pooling
            conv_outs.append(out)
        
        h = torch.cat(conv_outs, dim=1)
        h = self.dropout(h)
        return self.fc(h)

def create_model(
    model_config: ModelConfig, task_config: TaskConfig, charlm_encoder=None
) -> nn.Module:
    output_dim = len(task_config.label_to_idx)
    if task_config.task_type == "binary":
        output_dim = 1
    if model_config.model_type == "gru":
        return GRUClassifier(model_config, output_dim, charlm_encoder)
    elif model_config.model_type == "cnn":
        return CNNClassifier(model_config, output_dim, charlm_encoder)
    elif model_config.model_type == "charlm":
        return CharLanguageModel(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_config.model_type}")

def load_charlm_encoder(model_file, device, freeze=True):

    """Load pretrained CharLM to use as encoder."""
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    
    if checkpoint["task_config"].name != "charlm":
        raise ValueError("Provided model is not a CharLM")
    
    encoder = CharLanguageModel(checkpoint["model_config"])
    encoder.load_state_dict(checkpoint["model_state"])
    encoder.to(device)
    
    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False
    
    return encoder, checkpoint["model_config"].hidden_dim

def get_loss_fn(task_config: TaskConfig, pos_weight=None):

    if task_config.task_type in ["multilabel", "binary"]:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif task_config.task_type == "multiclass":
        return nn.CrossEntropyLoss(ignore_index=-1)
    elif task_config.task_type == "generative":
        return nn.CrossEntropyLoss()


def load_model(model_file, charlm_encoder):
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)

    byt5_tokenizer = None
    if checkpoint["model_config"].use_byt5:
        byt5_model_name = checkpoint.get("byt5_model", "google/byt5-small")
        log.info(f"Loading ByT5 for inference: {byt5_model_name}")
        byt5_tokenizer = ByT5Tokenizer(byt5_model_name, device=device, freeze=True)

    model = create_model(
        checkpoint["model_config"], checkpoint["task_config"], charlm_encoder
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, checkpoint, byt5_tokenizer

def compute_model_hash(model):
    """Compute SHA256 hash of model weights."""
    import hashlib
    hasher = hashlib.sha256()
    for param in model.parameters():
        hasher.update(param.data.cpu().numpy().tobytes())
    return hasher.hexdigest()

def evaluate_multilabel(model, loader, device, thresholds, beta=1.0):
    """Returns precision, recall, F-beta per class."""

    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            all_logits.append(model(x))
            all_targets.append(y)

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0).int()

    preds = (logits.sigmoid() >= thresholds).int()

    tp = (preds & targets).sum(dim=0).float()
    fp = (preds & (1 - targets)).sum(dim=0).float()
    fn = ((1 - preds) & targets).sum(dim=0).float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f_beta = (
        (1 + beta**2)
        * precision
        * recall
        / ((beta**2 * precision) + recall + 1e-8)
    )

    return precision, recall, f_beta

def evaluate_binary(model, loader, device, threshold=0.5):
    
    """Returns accuracy, precision, recall, F1."""
    model.eval()
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            # if attention_mask is not None:
            #     attention_mask = attention_mask.to(device)
            all_logits.append(model(x))
            all_targets.append(y)

    logits = torch.cat(all_logits, dim=0).squeeze()
    targets = torch.cat(all_targets, dim=0).squeeze().int()

    preds = (logits.sigmoid() >= threshold).int()

    tp = (preds & targets).sum().float()
    fp = (preds & (1 - targets)).sum().float()
    fn = ((1 - preds) & targets).sum().float()
    tn = ((1 - preds) & (1 - targets)).sum().float()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return accuracy, precision, recall, f1

def evaluate_multiclass(model, loader, device):
    """Returns accuracy and per-class (precision, recall, F1)."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            # if attention_mask is not None:
            #     attention_mask = attention_mask.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds)
            all_targets.append(y)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    # Filter out ignored indices
    mask = targets != -1
    preds = preds[mask]
    targets = targets[mask]

    accuracy = (preds == targets).float().mean()

    num_classes = int(targets.max()) + 1
    precisions, recalls, f1_scores = [], [], []

    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().float()
        fp = ((preds == c) & (targets != c)).sum().float()
        fn = ((preds != c) & (targets == c)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        precisions.append(precision.item())
        recalls.append(recall.item())
        f1_scores.append(f1.item())

    return accuracy.item(), precisions, recalls, f1_scores

def find_best_thresholds(logits, targets, task_type, steps=50, beta=1.0):

    """Find optimal thresholds for binary or multilabel tasks."""
    device = logits.device
    targets = targets.int()
    
    thresholds = torch.linspace(0.0, 1.0, steps, device=device)
    
    if task_type == "binary":
        # Binary: find single optimal threshold
        best_t = torch.tensor(0.5, device=device)
        best_f = torch.tensor(0.0, device=device)
        
        for t in thresholds:
            preds = (logits.sigmoid() >= t).int()
            tp = (preds & targets).sum().float()
            fp = (preds & (1 - targets)).sum().float()
            fn = ((1 - preds) & targets).sum().float()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f_beta = (
                (1 + beta**2) * precision * recall 
                / ((beta**2 * precision) + recall + 1e-8)
            )
            
            if f_beta > best_f:
                best_f = f_beta
                best_t = t
        
        return best_t.unsqueeze(0), best_f.unsqueeze(0)
    
    elif task_type == "multilabel":
        # Multilabel: find per-class thresholds
        best_t = torch.zeros(logits.size(1), device=device)
        best_f = torch.zeros(logits.size(1), device=device)

        for c in range(logits.size(1)):
            for t in thresholds:
                preds = (logits[:, c].sigmoid() >= t).int()
                tp = (preds & targets[:, c]).sum().float()
                fp = (preds & (1 - targets[:, c])).sum().float()
                fn = ((1 - preds) & targets[:, c]).sum().float()

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f_beta = (
                    (1 + beta**2) * precision * recall 
                    / ((beta**2 * precision) + recall + 1e-8)
                )

                if f_beta > best_f[c]:
                    best_f[c] = f_beta
                    best_t[c] = t

        return best_t, best_f
    
    else:
        return None, None

def collect_logits_targets(model, loader, device):
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            # if attention_mask is not None:
            #     attention_mask = attention_mask.to(device)
            all_logits.append(model(x))
            all_targets.append(y)

    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)

class TopNModels:
    """Track and save top-N best models."""
    
    def __init__(self, n: int, models_dir: Path):
        self.n = n
        self.models_dir = models_dir
        self.models = []  # heap of (dev_loss, epoch, path)
    
    def add(self, dev_loss: float, epoch: int, checkpoint: dict):
        """Add model if it's in top-N."""
        model_path = self.models_dir / f"model_epoch{epoch}_loss{dev_loss:.4f}.pt"
        
        # If we have < N models, just add
        if len(self.models) < self.n:
            torch.save(checkpoint, model_path)
            heapq.heappush(self.models, (-dev_loss, epoch, model_path))
        # If this model is better than worst in top-N
        elif dev_loss < -self.models[0][0]:
            # Remove worst model
            _, _, old_path = heapq.heappop(self.models)
            if old_path.exists():
                old_path.unlink()
            
            # Add new model
            torch.save(checkpoint, model_path)
            heapq.heappush(self.models, (-dev_loss, epoch, model_path))
    
    def get_best_path(self) -> Path:
        """Return path to best model."""
        return min(self.models, key=lambda x: x[0])[2]
