"""Main training script for toxic comment classification."""

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ToxicCommentTransformer
from src.data import load_civil_comments_data, create_dataloaders
from src.trainer import (
    Trainer, 
    compute_class_weights,
    compute_metrics, 
    compute_pos_weight,
    find_best_threshold, 
    get_warmup_cosine_schedule,
)
from src.config import (
    AGGRESSIVE_TRAINING,
    Config,
    MEDIUM_MODEL,
    SMALL_MODEL,
)

logger = logging.getLogger(__name__)


CONFIG_PRESETS = {
    "default": Config(),
    "small": SMALL_MODEL,
    "medium": MEDIUM_MODEL,
    "aggressive": AGGRESSIVE_TRAINING,
}


def create_run_directory(base_directory: str) -> Path:
    """Create a unique timestamped directory for one training run."""
    base_path = Path(base_directory)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_path = base_path / timestamp
    suffix = 1

    while run_path.exists():
        run_path = base_path / f"{timestamp}_{suffix}"
        suffix += 1

    run_path.mkdir(parents=True)
    return run_path


def apply_config(args):
    """Apply a config preset, keeping explicitly provided CLI values."""
    config = CONFIG_PRESETS[args.config]
    data_config = config.data
    model_config = config.model
    training_config = config.training

    data_config = replace(
        data_config,
        data_path=args.data_path if args.data_path is not None else data_config.data_path,
        text_column=args.text_column if args.text_column is not None else data_config.text_column,
        target_column=args.target_column if args.target_column is not None else data_config.target_column,
        test_size=args.test_size if args.test_size is not None else data_config.test_size,
        val_size=args.val_size if args.val_size is not None else data_config.val_size,
        max_seq_len=args.max_seq_len if args.max_seq_len is not None else data_config.max_seq_len,
        vocab_size=args.vocab_size if args.vocab_size is not None else data_config.vocab_size,
        tokenizer=args.tokenizer if args.tokenizer is not None else data_config.tokenizer,
        soft_labels=args.soft_labels if args.soft_labels is not None else data_config.soft_labels,
        dedup=args.dedup if args.dedup is not None else data_config.dedup,
        official_test=args.official_test if args.official_test is not None else data_config.official_test,
        seed=args.seed if args.seed is not None else data_config.seed,
    )
    model_config = replace(
        model_config,
        vocab_size=data_config.vocab_size,
        max_seq_len=data_config.max_seq_len,
        embedding_dim=args.embedding_dim if args.embedding_dim is not None else model_config.embedding_dim,
        num_layers=args.num_layers if args.num_layers is not None else model_config.num_layers,
        nhead=args.nhead if args.nhead is not None else model_config.nhead,
        dim_feedforward=(
            args.dim_feedforward
            if args.dim_feedforward is not None
            else model_config.dim_feedforward
        ),
        dropout=args.dropout if args.dropout is not None else model_config.dropout,
    )
    training_config = replace(
        training_config,
        batch_size=args.batch_size if args.batch_size is not None else training_config.batch_size,
        learning_rate=(
            args.learning_rate
            if args.learning_rate is not None
            else training_config.learning_rate
        ),
        weight_decay=args.weight_decay if args.weight_decay is not None else training_config.weight_decay,
        num_epochs=args.num_epochs if args.num_epochs is not None else training_config.num_epochs,
        early_stopping_patience=(
            args.early_stopping_patience
            if args.early_stopping_patience is not None
            else training_config.early_stopping_patience
        ),
        checkpoint_dir=(
            args.checkpoint_dir
            if args.checkpoint_dir is not None
            else training_config.checkpoint_dir
        ),
        use_amp=args.use_amp if args.use_amp is not None else training_config.use_amp,
        use_pos_weight=(
            not args.no_pos_weight
            if args.no_pos_weight is not None
            else training_config.use_pos_weight
        ),
    )

    args.data_path = data_config.data_path
    args.text_column = data_config.text_column
    args.target_column = data_config.target_column
    args.test_size = data_config.test_size
    args.val_size = data_config.val_size
    args.max_seq_len = data_config.max_seq_len
    args.vocab_size = data_config.vocab_size
    args.tokenizer = data_config.tokenizer
    args.soft_labels = data_config.soft_labels
    args.dedup = data_config.dedup
    args.official_test = data_config.official_test
    args.seed = data_config.seed
    args.embedding_dim = model_config.embedding_dim
    args.num_layers = model_config.num_layers
    args.nhead = model_config.nhead
    args.dim_feedforward = model_config.dim_feedforward
    args.dropout = model_config.dropout
    args.batch_size = training_config.batch_size
    args.learning_rate = training_config.learning_rate
    args.weight_decay = training_config.weight_decay
    args.num_epochs = training_config.num_epochs
    args.early_stopping_patience = training_config.early_stopping_patience
    args.checkpoint_dir = training_config.checkpoint_dir
    args.use_amp = training_config.use_amp
    args.use_pos_weight = training_config.use_pos_weight
    return args


def main(args):
    """Main training function."""

    checkpoint_dir = create_run_directory(args.checkpoint_dir)
    args.checkpoint_dir = str(checkpoint_dir)
    log_file = checkpoint_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger.info(f"Logging training output to {log_file}")
    logger.info(
        "Training configuration: batch_size=%d, max_train_batches=%s, "
        "max_val_batches=%s, max_test_batches=%s",
        args.batch_size,
        args.max_train_batches or "all",
        args.max_val_batches or "all",
        args.max_test_batches or "all",
    )
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Device setup: CUDA > Apple Silicon (MPS) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    
    # Use DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    train_df, val_df, test_df = load_civil_comments_data(
        args.data_path,
        text_column=args.text_column,
        target_column=args.target_column,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        dedup=args.dedup,
        official_test=args.official_test,
    )
    
    logger.info(f"Train set size: {len(train_df)}")
    logger.info(f"Val set size: {len(val_df)}")
    logger.info(f"Test set size: {len(test_df)}")
    
    # Class distribution
    logger.info(f"Train label distribution:\n{train_df['label'].value_counts()}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
        train_df,
        val_df,
        test_df,
        text_column=args.text_column,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        vocab_size=args.vocab_size,
        num_workers=args.num_workers,
        tokenizer_type=args.tokenizer,
        seed=args.seed,
        soft_target_column=args.target_column if args.soft_labels else None,
    )

    logger.info(
        "Available batches: train=%d, validation=%d, test=%d",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )
    
    logger.info(f"Tokenizer: {args.tokenizer}, vocabulary size: {len(tokenizer.token2id)}")
    logger.info(
        "Loss targets: %s",
        f"soft scores from '{args.target_column}'" if args.soft_labels else "0/1 labels",
    )

    # Save the BPE tokenizer with the run, so evaluate.py uses exactly the same one
    if args.tokenizer == "bpe":
        tokenizer.save(checkpoint_dir / "tokenizer.json")
    
    # Initialize model
    logger.info("Initializing model...")
    model = ToxicCommentTransformer(
        vocab_size=len(tokenizer.token2id),
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Compute class weights for handling imbalance
    class_weights = compute_class_weights(train_df["label"].values)
    class_weights = class_weights.to(device)
    logger.info(f"Class weights: {class_weights}")

    # BCEWithLogitsLoss pos_weight should match the true imbalance ratio. Without
    # it, predicted probabilities stay calibrated; threshold tuning on validation
    # then handles the imbalance when making decisions.
    if args.use_pos_weight:
        pos_weight = compute_pos_weight(train_df["label"].values).to(device)
        logger.info(f"Positive class pos_weight: {pos_weight.item():.4f}")
    else:
        pos_weight = None
        logger.info("Positive class pos_weight: none (unweighted loss)")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler: linear warmup, then cosine decay over the whole run
    steps_per_epoch = len(train_loader)
    if args.max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, args.max_train_batches)
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_warmup_cosine_schedule(optimizer, warmup_steps, total_steps)
    logger.info(f"LR schedule: {warmup_steps} warmup steps, {total_steps} total steps")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=args.use_amp,
        log_every=args.log_every,
        summary_writer=SummaryWriter(log_dir=checkpoint_dir / "tensorboard"),
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        loss_fn=loss_fn,
        patience=args.early_stopping_patience,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
    
    logger.info("Training completed!")

    # Tune the decision threshold on validation (best checkpoint is already restored)
    logger.info("Tuning decision threshold on validation set...")
    _, val_probs, val_labels = trainer.predict(
        val_loader,
        loss_fn,
        max_batches=args.max_val_batches,
    )
    threshold = find_best_threshold(val_labels, val_probs)
    logger.info(f"Best validation threshold: {threshold:.4f}")
    logger.info(f"Val metrics at tuned threshold: {compute_metrics(val_labels, val_probs, threshold)}")

    # Save the threshold with the best checkpoint so evaluate.py can reuse it
    if "best_epoch" in history:
        best_checkpoint_path = Path(args.checkpoint_dir) / f"model_epoch_{history['best_epoch']}.pt"
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        checkpoint["threshold"] = threshold
        torch.save(checkpoint, best_checkpoint_path)

    # Final evaluation on test set, at the default and the tuned threshold
    logger.info("Evaluating on test set...")
    test_loss, test_probs, test_labels = trainer.predict(
        test_loader,
        loss_fn,
        max_batches=args.max_test_batches,
    )
    logger.info(f"Test loss: {test_loss:.4f}")
    logger.info(f"Test metrics at 0.5: {compute_metrics(test_labels, test_probs, 0.5)}")
    logger.info(f"Test metrics at tuned threshold {threshold:.4f}: {compute_metrics(test_labels, test_probs, threshold)}")
    if trainer.summary_writer is not None:
        trainer.summary_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train toxic comment classifier"
    )

    parser.add_argument(
        "--config",
        choices=sorted(CONFIG_PRESETS),
        default="default",
        help="Configuration preset from src/config.py",
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="Name of text column in CSV",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default=None,
        help="Name of target column in CSV",
    )
    
    # Data split arguments
    parser.add_argument(
        "--test_size",
        type=float,
        default=None,
        help="Test set proportion",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=None,
        help="Validation set proportion",
    )
    
    # Model arguments
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--tokenizer",
        choices=["word", "bpe"],
        default=None,
        help="Tokenizer: word-level (word) or byte-level BPE subwords (bpe)",
    )
    parser.add_argument(
        "--soft_labels",
        action="store_true",
        default=None,
        help="Train on the soft toxicity score (target column) instead of the 0/1 label",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        default=None,
        help="Collapse duplicate texts (case/whitespace-insensitive) before splitting",
    )
    parser.add_argument(
        "--official_test",
        action="store_true",
        default=None,
        help="Test on the official hold-out set (test_public/private_expanded.csv); the CSV is split into train/val only",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=None,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="Number of Transformer layers",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=None,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=None,
        help="Feedforward dimension",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout rate",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Weight decay",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of epochs",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Fraction of total training steps used for linear warmup",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=200,
        help="Log training loss to TensorBoard every this many steps",
    )
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
        help="Maximum training batches per epoch (useful for debugging)",
    )
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=None,
        help="Maximum validation batches per epoch (useful for debugging)",
    )
    parser.add_argument(
        "--max_test_batches",
        type=int,
        default=None,
        help="Maximum test batches for final evaluation (useful for debugging)",
    )
    
    # Other arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for DataLoader",
    )
    parser.add_argument(
        "--no_pos_weight",
        action="store_true",
        default=None,
        help="Train with an unweighted loss instead of pos_weight = negatives/positives",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=None,
        help="Use automatic mixed precision",
    )
    
    args = apply_config(parser.parse_args())
    main(args)
