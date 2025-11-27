
import argparse
from pathlib import Path

import torch
import pandas as pd

from torch.utils.data import (
    DataLoader,
    random_split,
    Subset,
)
from classifiers.dl.logging_ import (
    EpochLogger,
    log,
    setup_output_file_log
) 
from classifiers.dl.models_ import (
    create_model,
    get_loss_fn,
    collect_logits_targets,
    find_best_thresholds,
    compute_model_hash,
    TopNModels,
    load_model,
    evaluate_multilabel,
    evaluate_binary,
    evaluate_multiclass,
    load_charlm_encoder,
    ByT5Tokenizer,
)
from classifiers.dl.model_config import (
    ModelConfig,
    get_model_config,
)
from classifiers.dl.task_config import (
    TASKS,
    TaskConfig,
    TrainConfig
)
from classifiers.dl.data_ import (
    WordDataset,
    OriginDataset,
    collate,
    collate_lm,
    compute_pos_weights,
    char2idx
)
from classifiers.dl.output_ import (
    setup_output_dir,
    save_test_metrics,
    output_evaluation,
)
from classifiers.dl.torch_config import (
    device
)

def run_epoch(model, loader, criterion, optimizer, device, train=False):

    model.train() if train else model.eval()
    torch.set_grad_enabled(train)
    total_loss = 0.0
    steps = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / steps

def run_epoch_lm(model, loader, criterion, optimizer, device, train=False):

    model.train() if train else model.eval()
    torch.set_grad_enabled(train)
    total_loss = 0.0
    steps = 0

    for x in loader:

        x = x.to(device)
        logits, _ = model(x[:, :-1])
        targets = x[:, 1:]
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )
            optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / steps

def main():

    # CLIargs.
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to CSV dataset")
    parser.add_argument(
        "--name",
        default="model",
        help="Base name for output directory"
    )
    parser.add_argument(
        "--task",
        choices=list(TASKS.keys()),
        required=True,
        help="Classification task",
    )
    parser.add_argument(
        "--model",
        choices=["gru", "cnn", "charlm"],
        default="gru",
        help="Model architecture (gru, cnn, charlm)",
    )
    parser.add_argument(
        "--use-byt5", action="store_true", help="Use ByT5 embeddings"
    )
    parser.add_argument(
        "--byt5-model", default="google/byt5-base", help="ByT5 model name"
    )
    parser.add_argument(
        "--byt5-finetune", action="store_true", help="Fine-tune ByT5"
    )
    parser.add_argument(
        "--charlm-encoder",
        type=str,
        help="Path to pretrained CharLM .pt file to use as encoder"
    )
    parser.add_argument(
        "--charlm-finetune", action="store_true", help="Fine-tune CharLM encoder"
    )
    parser.add_argument(
        "--downsample", type=int, default=None, help="Limit dataset size"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pretrained_models",
        help="Base directory to store experiment outputs (default: ./results)"
    )
    args = parser.parse_args()

    # Setup output directory inside specified base output path.
    model_variant = f"{args.task}_{args.model}"
    if args.use_byt5:
        model_variant += "_byt5"

    # Setup output dir.
    base_output = Path(args.output)
    base_output.mkdir(parents=True, exist_ok=True)
    output_dir = base_output / model_variant
    output_dir = setup_output_dir(output_dir)
    setup_output_file_log(output_dir)
    
    log.info(f"Output directory: {output_dir}")

    # Initialize ByT5 if requested
    byt5_tokenizer = None
    if args.use_byt5:
        log.info(f"Loading ByT5: {args.byt5_model}")
        byt5_tokenizer = ByT5Tokenizer(
            args.byt5_model,
            device=device,
            freeze=not args.byt5_finetune
        )
        log.info(f"ByT5 embedding dim: {byt5_tokenizer.embedding_dim}")

    # Initialize CharLM encoder if requested.
    charlm_encoder = None
    charlm_hidden_dim = None
    if args.charlm_encoder:
        log.info(f"Loading CharLM encoder: {args.charlm_encoder}")
        charlm_encoder, charlm_hidden_dim = load_charlm_encoder(
            args.charlm_encoder, device, freeze=not args.charlm_finetune
        )
        log.info(f"CharLM hidden dim: {charlm_hidden_dim}")

    # Configs
    task_config = TASKS[args.task]
    encoder_type = "charlm" if args.charlm_encoder else (
        "byt5" if args.use_byt5 else "raw"
    )
    model_config = get_model_config(
        task=args.task,
        model_type=args.model,
        encoder_type=encoder_type,
        byt5_dim=byt5_tokenizer.embedding_dim if byt5_tokenizer else None,
        charlm_hidden_dim=charlm_hidden_dim,
    )
    train_config = TrainConfig(
        batch_size=64,
        learning_rate=3e-4,
        weight_decay=1e-3,
        num_epochs=1000,
        patience=1, # 5
        max_seq_len=25,
        beta=1, # 0.7
    )
    log.info(f"Device: {device}")
    log.info(f"Task: {task_config.name} ({task_config.task_type})")
    log.info(f"Model config: {model_config.model_type}")

    # Load the data..
    full_df = pd.read_csv(args.input)
    if args.task == "charlm":
        dataset = WordDataset(args.input, task_config)
    else:
        dataset = OriginDataset(args.input, task_config, use_byt5=args.use_byt5)
    if args.downsample:
        dataset = Subset(dataset, range(min(args.downsample, len(dataset))))
        full_df = full_df.iloc[:args.downsample]

    # Split the dataset
    n = len(dataset)
    g = torch.Generator().manual_seed(42)
    n_train = int(n * 0.8)
    n_dev = int(n * 0.1)
    n_test = n - n_train - n_dev
    train_ds, dev_ds, test_ds = random_split(
        dataset,
        [n_train, n_dev, n_test],
        generator=g
    )
    train_indices = train_ds.indices
    dev_indices = dev_ds.indices
    test_indices = test_ds.indices

    # Save the datasets for reference.
    dataset_dir = output_dir / "datasets"
    if args.task != "charlm":
        full_df.iloc[train_indices].to_csv(
            dataset_dir / "train.csv", index=False
        )
        full_df.iloc[dev_indices].to_csv(
            dataset_dir / "dev.csv", index=False
        )
        full_df.iloc[test_indices].to_csv(
            dataset_dir / "test.csv", index=False
        )
    log.info(f"Saved data splits to {dataset_dir}")

    collate_fn = collate_lm if args.task == "charlm" else collate
    train_loader = DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, train_config.max_seq_len, byt5_tokenizer),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, train_config.max_seq_len, byt5_tokenizer),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, train_config.max_seq_len, byt5_tokenizer),
    )

    # Model.
    model = create_model(model_config, task_config, charlm_encoder).to(device)
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss.
    criterion = get_loss_fn(
        task_config, 
        train_loader,
        device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # Initialize trackers
    top_models = TopNModels(n=5, models_dir=output_dir / "models")
    epoch_logger = EpochLogger(output_dir / "metrics.csv")

    # Training loop
    best_dev_loss = float("inf")
    best_epoch = -1
    patience_left = train_config.patience

    # Training cycle.
    current_lr = optimizer.param_groups[0]['lr']
    for epoch in range(train_config.num_epochs):

        run_fn = run_epoch_lm if args.task == "charlm" else run_epoch
        train_loss = run_fn(
            model, train_loader, criterion, optimizer, device, train=True
        )
        dev_loss = run_fn(
            model, dev_loader, criterion, optimizer, device, train=False
        )
        log.info(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}"
            f", dev_loss={dev_loss:.4f}"
        )

        epoch_logger.log_epoch(epoch, train_loss, dev_loss, current_lr)
        scheduler.step(dev_loss)

        if dev_loss < best_dev_loss:

            best_dev_loss = dev_loss
            patience_left = train_config.patience
            best_epoch = epoch

            if task_config.task_type != "generative":
                logits, targets = collect_logits_targets(model, dev_loader, device)
                best_t, _ = find_best_thresholds(
                    logits, targets, task_config.task_type, beta=train_config.beta
                )
            else:
                best_t = 0

            checkpoint = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "dev_loss": dev_loss,
                "model_config": model_config,
                "task_config": task_config,
                "train_config": train_config,
                "model_hash": compute_model_hash(model),
                "char2idx": char2idx,
                "byt5_model": args.byt5_model if args.use_byt5 else None,
                "thresholds": best_t,
            }

            top_models.add(dev_loss, epoch, checkpoint)
        else:
            patience_left -= 1
            if patience_left == 0:
                log.info("Early stopping")
                break

    # Reload the model.
    best_model_path = top_models.get_best_path()
    log.info(f"Loading best model from: {best_model_path}")
    model, checkpoint, _ = load_model(best_model_path, charlm_encoder)
    task_config = checkpoint["task_config"]
    thresholds = checkpoint["thresholds"]

    # Evaluate on test.
    log.info(f"Best epoch: {best_epoch}, dev_loss: {best_dev_loss:.4f}")
    output_evaluation(
        model,
        train_config,
        task_config,
        test_loader,
        device,
        thresholds,
        output_dir,
    )

if __name__ == "__main__":
    main()
