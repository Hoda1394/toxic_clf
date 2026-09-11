"""Main training script for toxic comment classification."""

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataclasses import replace
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ToxicCommentTransformer
from src.data import load_civil_comments_data, create_dataloaders
from src.trainer import Trainer, compute_class_weights
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
    )

    args.data_path = data_config.data_path
    args.text_column = data_config.text_column
    args.target_column = data_config.target_column
    args.test_size = data_config.test_size
    args.val_size = data_config.val_size
    args.max_seq_len = data_config.max_seq_len
    args.vocab_size = data_config.vocab_size
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
    return args


def main(args):
    """Main training function."""

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
    )
    
    logger.info(f"Vocabulary size: {len(tokenizer.token2id)}")
    
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
    
    # Loss function with class weights
    # Clamp pos_weight to prevent numerical instability
    pos_weight_ratio = class_weights[1] / class_weights[0] if class_weights[0] > 0 else 1.0
    pos_weight = torch.clamp(pos_weight_ratio, min=1.0, max=10.0)  # Reasonable bounds
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=args.use_amp,
        warmup_steps=args.warmup_steps,
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
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(
        test_loader,
        loss_fn,
        max_batches=args.max_test_batches,
    )
    logger.info(f"Test loss: {test_loss:.4f}")
    logger.info(f"Test metrics: {test_metrics}")
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
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps for learning rate",
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
        "--use_amp",
        action="store_true",
        default=None,
        help="Use automatic mixed precision",
    )
    
    args = apply_config(parser.parse_args())
    main(args)
