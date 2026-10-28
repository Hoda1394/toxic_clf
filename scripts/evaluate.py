"""Evaluation script for trained models."""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ToxicCommentTransformer
from src.data import BPETokenizer, load_civil_comments_data, create_dataloaders
from src.trainer import Trainer, compute_pos_weight


def main(args):
    """Evaluate a trained model."""
    
    # Device setup: CUDA > Apple Silicon (MPS) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_path}")
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
    
    # A BPE run saves its tokenizer next to the checkpoints; reuse it. Otherwise
    # rebuild the word vocabulary from the training split, as train.py did.
    tokenizer = None
    tokenizer_path = Path(args.checkpoint_path).parent / "tokenizer.json"
    if tokenizer_path.exists():
        print(f"Loading BPE tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        print("No tokenizer.json next to the checkpoint; rebuilding the word vocabulary")

    # Create dataloaders
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
        train_df,
        val_df,
        test_df,
        text_column=args.text_column,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        vocab_size=args.vocab_size,
        tokenizer=tokenizer,
        seed=args.seed,
        soft_target_column=args.target_column if args.soft_labels else None,
    )
    
    # Load model
    print(f"Loading model from {args.checkpoint_path}")
    model = ToxicCommentTransformer(
        vocab_size=len(tokenizer.token2id),
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    # Loss function: same pos_weight as training, so the loss matches the training log
    pos_weight = None if args.no_pos_weight else compute_pos_weight(train_df["label"].values).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Create trainer for evaluation
    dummy_optimizer = torch.optim.Adam(model.parameters())
    trainer = Trainer(model, dummy_optimizer, device)
    
    # Evaluate
    print("Evaluating on test set...")
    # Use the threshold tuned during training, unless one is given explicitly
    threshold = args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5)
    print(f"Using threshold: {threshold:.4f}")
    test_loss, test_metrics = trainer.evaluate(
        test_loader,
        loss_fn,
        max_batches=args.max_test_batches,
        threshold=threshold,
    )
    
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test metrics: {test_metrics}")
    
    # Save results
    if args.output_path:
        results = {
            "loss": test_loss,
            "metrics": test_metrics,
            "threshold": threshold,
        }
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate toxicity model")
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/train.csv",
        help="Path to data CSV",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of text column",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="target",
        help="Name of target column",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold (default: the one saved in the checkpoint, else 0.5)",
    )
    parser.add_argument(
        "--soft_labels",
        action="store_true",
        help="Compute the loss on soft targets, as in a --soft_labels training run (metrics are unaffected)",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Collapse duplicate texts before splitting, as in a --dedup training run (changes the split)",
    )
    parser.add_argument(
        "--official_test",
        action="store_true",
        help="Test on the official hold-out set (test_public/private_expanded.csv); the CSV is split into train/val only",
    )
    parser.add_argument(
        "--no_pos_weight",
        action="store_true",
        help="Compute the loss without pos_weight, as in a --no_pos_weight training run (metrics are unaffected)",
    )
    parser.add_argument(
        "--max_test_batches",
        type=int,
        default=None,
        help="Maximum test batches to evaluate (useful for quick checks)",
    )
    
    args = parser.parse_args()
    main(args)
