"""Fine-tune a pretrained transformer (e.g. DistilBERT) as a reference model.

Uses the same train/val/test split, metrics, checkpoint selection, threshold
tuning and training loop (src/trainer.py) as scripts/train.py, so the results
go straight into the results table next to the from-scratch models. Only the
model and tokenizer differ: both come from the Hugging Face hub.

Pretrained models need far less data than training from scratch, and are much
slower per example, so by default only a random subset of the training split
is used.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train import create_run_directory
from src.data import LengthGroupedBatchSampler, load_civil_comments_data
from src.trainer import (
    Trainer,
    compute_metrics,
    find_best_threshold,
    get_warmup_cosine_schedule,
)

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Raw texts with their 0/1 labels (for metrics) and loss targets."""

    def __init__(self, texts: list, labels: list, targets: list):
        self.texts = texts
        self.labels = labels
        self.targets = targets

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        return (
            text if isinstance(text, str) else "",
            self.labels[idx],
            self.targets[idx],
        )


class PretrainedClassifier(nn.Module):
    """Wraps a Hugging Face model so it returns plain logits, like ToxicCommentTransformer.

    This lets the existing Trainer use it unchanged.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def make_loader(
    df: pd.DataFrame,
    tokenizer,
    args,
    reshuffle: bool,
) -> DataLoader:
    """DataLoader that tokenizes each batch with the pretrained model's tokenizer."""
    targets = df[args.target_column] if args.soft_labels else df["label"]
    dataset = TextDataset(
        df[args.text_column].tolist(),
        df["label"].tolist(),
        targets.tolist(),
    )

    def collate(batch: list) -> dict:
        texts, labels, batch_targets = zip(*batch)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
            pad_to_multiple_of=args.pad_to_multiple_of,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.float),
            "targets": torch.tensor(batch_targets, dtype=torch.float),
        }

    lengths = [len(text) if isinstance(text, str) else 0 for text in df[args.text_column]]
    return DataLoader(
        dataset,
        batch_sampler=LengthGroupedBatchSampler(
            lengths,
            args.batch_size,
            seed=args.seed,
            reshuffle=reshuffle,
        ),
        collate_fn=collate,
        num_workers=args.num_workers,
    )


def subsample(df: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    """Random subset of at most max_samples rows (all rows if max_samples is None)."""
    if max_samples is None or max_samples >= len(df):
        return df
    return df.sample(max_samples, random_state=seed)


def main(args):
    """Fine-tune and evaluate a pretrained model."""

    checkpoint_dir = create_run_directory(args.checkpoint_dir)
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
    logger.info(f"Arguments: {vars(args)}")

    torch.manual_seed(args.seed)

    # Device setup: CUDA > Apple Silicon (MPS) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Same split as scripts/train.py
    logger.info(f"Loading data from {args.data_path}")
    train_df, val_df, test_df = load_civil_comments_data(
        args.data_path,
        text_column=args.text_column,
        target_column=args.target_column,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        dedup=args.dedup,
    )
    train_df = subsample(train_df, args.max_train_samples, args.seed)
    val_df = subsample(val_df, args.max_val_samples, args.seed)
    test_df = subsample(test_df, args.max_test_samples, args.seed)
    logger.info(f"Train size: {len(train_df)}, val size: {len(val_df)}, test size: {len(test_df)}")
    logger.info(
        "Loss targets: %s",
        f"soft scores from '{args.target_column}'" if args.soft_labels else "0/1 labels",
    )

    logger.info(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = PretrainedClassifier(args.model_name).to(device)
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader = make_loader(train_df, tokenizer, args, reshuffle=True)
    val_loader = make_loader(val_df, tokenizer, args, reshuffle=False)
    test_loader = make_loader(test_df, tokenizer, args, reshuffle=False)

    # Unweighted loss: threshold tuning on validation handles the class imbalance
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_warmup_cosine_schedule(optimizer, warmup_steps, total_steps)
    logger.info(f"LR schedule: {warmup_steps} warmup steps, {total_steps} total steps")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=str(checkpoint_dir),
        log_every=args.log_every,
        summary_writer=SummaryWriter(log_dir=checkpoint_dir / "tensorboard"),
    )

    logger.info("Starting fine-tuning...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        loss_fn=loss_fn,
        patience=args.num_epochs,
    )
    logger.info("Fine-tuning completed!")

    # Tune the decision threshold on validation (best checkpoint is already restored)
    logger.info("Tuning decision threshold on validation set...")
    _, val_probs, val_labels = trainer.predict(val_loader, loss_fn)
    threshold = find_best_threshold(val_labels, val_probs)
    logger.info(f"Best validation threshold: {threshold:.4f}")
    logger.info(f"Val metrics at tuned threshold: {compute_metrics(val_labels, val_probs, threshold)}")

    # Final evaluation on test set, at the default and the tuned threshold
    logger.info("Evaluating on test set...")
    test_loss, test_probs, test_labels = trainer.predict(test_loader, loss_fn)
    logger.info(f"Test loss: {test_loss:.4f}")
    logger.info(f"Test metrics at 0.5: {compute_metrics(test_labels, test_probs, 0.5)}")
    logger.info(
        f"Test metrics at tuned threshold {threshold:.4f}: "
        f"{compute_metrics(test_labels, test_probs, threshold)}"
    )
    trainer.summary_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained transformer as a reference model"
    )

    # Data arguments (defaults match src/config.py so the split is identical)
    parser.add_argument("--data_path", type=str, default="data/train.csv")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--target_column", type=str, default="target")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=200_000,
        help="Random subset of the training split to fine-tune on",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="Random subset of the validation split (default: all)",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Random subset of the test split (default: all)",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Collapse duplicate texts before splitting",
    )
    parser.add_argument(
        "--soft_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train on the soft toxicity score (default) or, with --no-soft_labels, the 0/1 label",
    )

    # Model and training arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Hugging Face model name",
    )
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="16 fits a 24 GB Mac; 32 runs out of MPS memory on long batches",
    )
    parser.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=64,
        help="Round padded lengths up to a multiple of this, so MPS sees only a few batch shapes "
        "(every new shape makes it cache more buffers)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Peak learning rate (pretrained models need a much lower one than training from scratch)",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/pretrained",
        help="Each run gets its own timestamped folder in here",
    )

    args = parser.parse_args()
    main(args)
