"""Training loop and utilities."""
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,)
from tqdm import tqdm
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for model training and evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        scheduler=None,
        checkpoint_dir: str = "./checkpoints",
        use_amp: bool = False,
        log_every: int = 200,
        summary_writer=None,
    ):
        """Initialize Trainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            device: Device to train on
            scheduler: Learning rate scheduler (optional)
            checkpoint_dir: Directory to save checkpoints
            use_amp: Whether to use automatic mixed precision
            log_every: Log training loss to TensorBoard every this many steps
            summary_writer: Optional TensorBoard SummaryWriter
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.use_amp = use_amp
        self.log_every = log_every
        self.summary_writer = summary_writer
        self.current_step = 0
        
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        class_weights: torch.Tensor = None,
        max_batches: int = None,
    ) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training dataloader
            loss_fn: Loss function
            class_weights: Class weights for weighted loss
            max_batches: Optional maximum number of batches to process
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        running_loss = 0.0
        
        progress_bar = tqdm(
            train_loader,
            desc="Training",
            disable=False,
        )
        
        processed_batches = 0
        for batch_index, batch in enumerate(progress_bar):
            if max_batches is not None and batch_index >= max_batches:
                break

            processed_batches += 1
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["targets"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(input_ids, attention_mask)
                    loss = loss_fn(logits.squeeze(-1), targets)
            else:
                logits = self.model(input_ids, attention_mask)
                loss = loss_fn(logits.squeeze(-1), targets)

            # Fail loudly: a non-finite loss means a real bug or a too-high learning rate
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss ({loss.item()}) at step {self.current_step}. "
                    "Try a lower learning rate."
                )

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Per-step learning rate schedule (warmup + cosine decay)
            if self.scheduler is not None:
                self.scheduler.step()
            self.current_step += 1
            
            loss_value = loss.item()
            total_loss += loss_value
            running_loss += loss_value
            progress_bar.set_postfix({"loss": loss_value})

            # Average loss and current learning rate over the last log_every steps
            if self.summary_writer is not None and self.current_step % self.log_every == 0:
                self.summary_writer.add_scalar(
                    "Training/loss_per_step", running_loss / self.log_every, self.current_step
                )
                self.summary_writer.add_scalar(
                    "Training/lr_per_step", self.optimizer.param_groups[0]["lr"], self.current_step
                )
                running_loss = 0.0
        
        avg_loss = total_loss / processed_batches if processed_batches else 0.0
        return avg_loss
    
    @torch.no_grad()
    def predict(
        self,
        loader: DataLoader,
        loss_fn: nn.Module,
        max_batches: int = None,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Run the model over a dataloader.
        
        Args:
            loader: Dataloader
            loss_fn: Loss function
            max_batches: Optional maximum number of batches to process
        
        Returns:
            Tuple of (average loss, predicted probabilities, true labels)
        """
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_labels = []
        
        progress_bar = tqdm(
            loader,
            desc="Evaluating",
            disable=False,
        )
        
        processed_batches = 0
        for batch_index, batch in enumerate(progress_bar):
            if max_batches is not None and batch_index >= max_batches:
                break

            processed_batches += 1
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            targets = batch["targets"].to(self.device)
            
            # Loss on the training targets (comparable to the training loss);
            # metrics on the 0/1 labels
            logits = self.model(input_ids, attention_mask)
            loss = loss_fn(logits.squeeze(-1), targets)
            
            total_loss += loss.item()
            
            # Get predictions
            probs = torch.sigmoid(logits.squeeze(-1))
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().astype(int))
            
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / processed_batches if processed_batches else 0.0
        return avg_loss, np.array(all_probs), np.array(all_labels)

    def evaluate(
        self,
        loader: DataLoader,
        loss_fn: nn.Module,
        max_batches: int = None,
        threshold: float = 0.5,
    ) -> Tuple[float, Dict]:
        """Evaluate model.
        
        Args:
            loader: Dataloader to evaluate on
            loss_fn: Loss function
            max_batches: Optional maximum number of batches to process
            threshold: Probability threshold for the thresholded metrics (precision, recall, F1)

        Returns:
            Tuple of (average loss, metrics dict)
        """
        avg_loss, probs, labels = self.predict(loader, loss_fn, max_batches)
        return avg_loss, compute_metrics(labels, probs, threshold)
        
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        loss_fn: nn.Module,
        patience: int = 3,
        max_train_batches: int = None,
        max_val_batches: int = None,
        selection_metric: str = "pr_auc",
    ) -> Dict:
        """Full training loop.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of epochs
            loss_fn: Loss function
            patience: Early stopping patience
            max_train_batches: Optional maximum training batches per epoch
            max_val_batches: Optional maximum validation batches per epoch
            selection_metric: Metric to use for checkpointing and early stopping
        Returns:
            Training history dictionary
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": [],
        }
        
        best_val_score = float("-inf")
        best_epoch = None
        best_checkpoint_path = None
        patience_counter = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(
                train_loader,
                loss_fn,
                max_batches=max_train_batches,
            )
            history["train_loss"].append(train_loss)
            logger.info(f"Train loss: {train_loss:.4f}")
            
            # Evaluate
            val_loss, metrics = self.evaluate(
                val_loader,
                loss_fn,
                max_batches=max_val_batches,
            )
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(metrics)
            
            logger.info(f"Val loss: {val_loss:.4f}")
            logger.info(f"Metrics: {metrics}")

            if self.summary_writer is not None:
                self.summary_writer.add_scalars(
                    "Loss",
                    {"train": train_loss, "validation": val_loss},
                    epoch,
                )
                self.summary_writer.add_scalars(
                    "Metrics",
                    metrics,
                    epoch,
                )

                self.summary_writer.flush()
            
            # Early stopping on the selection metric (PR-AUC by default: threshold-free, so much less noisy between epochs than F1 at 0.5)
            score = metrics.get(selection_metric, float("-inf"))
            if score > best_val_score:
                best_val_score = score
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(epoch, metrics)
                best_checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch}.pt"
                logger.info(f"Best {selection_metric}: {best_val_score:.4f}, saving checkpoint")
            else:
                patience_counter += 1
                logger.info(f"No improvement in {selection_metric}. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logger.info(f"Early stopping after ({epoch + 1}) epochs with no improvement in {selection_metric}.")
                    break
        
        if best_checkpoint_path is not None:
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(
                "Restored best checkpoint from epoch %d (validation %s: %.4f)",
                best_epoch + 1,
                selection_metric,
                best_val_score,
            )
            history["best_epoch"] = best_epoch
            history["best_val_" + selection_metric] = best_val_score
        else:
            logger.warning("No checkpoint was saved; model remains at its final state.")

        return history
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Evaluation metrics
        """
        checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch}.pt"
        safe_metrics = {name: float(value) for name, value in metrics.items()}
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": safe_metrics,
            },
            checkpoint_path,
        )
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Checkpoint loaded from {checkpoint_path}")


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency class weights for binary labels.

    For binary labels with classes {0, 1}, the weight for each class is:
    total / (2 * count(class))

    This yields a positive weight larger than 1 when class 1 is underrepresented.
    """
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    total = len(labels)

    if total == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float)

    weights = total / (2 * counts)
    return torch.tensor(weights, dtype=torch.float)


def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    """Compute BCEWithLogitsLoss pos_weight for the positive class.

    For binary labels, this matches the expected imbalance ratio used by the
    README and the project documentation:
    pos_weight = count(negative) / count(positive)
    """
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2)

    if counts[1] == 0:
        return torch.tensor(1.0, dtype=torch.float)

    pos_weight = counts[0] / counts[1]
    return torch.tensor(pos_weight, dtype=torch.float)


def get_warmup_cosine_schedule(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Linear warmup from 0 to the base learning rate, then cosine decay to 0.

    Call scheduler.step() once per optimizer step (every batch), not per epoch.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, lr_lambda)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute threshold-free (AUC) and thresholded metrics from probabilities."""
    preds = (probs >= threshold).astype(int)

    # AUC metrics need both classes present (not guaranteed with tiny max_batches)
    has_both_classes = len(np.unique(labels)) == 2

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)) if has_both_classes else float("nan"),
        "pr_auc": float(average_precision_score(labels, probs)) if has_both_classes else float("nan"),
    }


def find_best_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Return the probability threshold that maximizes F1."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    # The last precision/recall pair has no matching threshold
    return float(thresholds[f1[:-1].argmax()])
