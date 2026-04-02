"""Main training script for toxic comment classification."""

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ToxicCommentTransformer
from src.data import load_civil_comments_data, create_dataloaders
from src.trainer import Trainer, compute_class_weights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """Main training function."""
    
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
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        loss_fn=loss_fn,
        patience=args.early_stopping_patience,
    )
    
    logger.info("Training completed!")
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader, loss_fn)
    logger.info(f"Test loss: {test_loss:.4f}")
    logger.info(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train toxic comment classifier"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/train.csv",
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of text column in CSV",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="toxicity",
        help="Name of target column in CSV",
    )
    
    # Data split arguments
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set proportion",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Validation set proportion",
    )
    
    # Model arguments
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10000,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of Transformer layers",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=4,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=512,
        help="Feedforward dimension",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=256,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,  # Reduced from 1e-3 for stability
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Early stopping patience",
    )
    
    # Other arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
        help="Use automatic mixed precision",
    )
    
    args = parser.parse_args()
    main(args)
