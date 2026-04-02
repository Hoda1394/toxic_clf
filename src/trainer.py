"""Training loop and utilities."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import numpy as np
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
    ):
        """Initialize Trainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            device: Device to train on
            scheduler: Learning rate scheduler (optional)
            checkpoint_dir: Directory to save checkpoints
            use_amp: Whether to use automatic mixed precision
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.use_amp = use_amp
        
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        class_weights: torch.Tensor = None,
    ) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training dataloader
            loss_fn: Loss function
            class_weights: Class weights for weighted loss
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(
            train_loader,
            desc="Training",
            disable=False,
        )
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Validate inputs
            if torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
                logger.warning("NaN or Inf values in input_ids. Skipping batch.")
                continue
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                logger.warning("NaN or Inf values in labels. Skipping batch.")
                continue
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(input_ids, attention_mask)
                    loss = loss_fn(logits.squeeze(), labels)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    logger.warning(f"NaN loss detected in batch. Logits range: [{logits.min():.3f}, {logits.max():.3f}], Labels range: [{labels.min():.3f}, {labels.max():.3f}]")
                    continue
                
                # Backward pass with scaler
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                # Check for NaN gradients
                has_nan_grad = any(torch.isnan(p.grad).any() for p in self.model.parameters() if p.grad is not None)
                if has_nan_grad:
                    logger.warning(f"NaN gradients detected. Skipping optimizer step.")
                    self.optimizer.zero_grad()
                    continue
                    
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(input_ids, attention_mask)
                loss = loss_fn(logits.squeeze(), labels)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    logger.warning(f"NaN loss detected in batch. Logits range: [{logits.min():.3f}, {logits.max():.3f}], Labels range: [{labels.min():.3f}, {labels.max():.3f}]")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Check for NaN gradients
                has_nan_grad = any(torch.isnan(p.grad).any() for p in self.model.parameters() if p.grad is not None)
                if has_nan_grad:
                    logger.warning(f"NaN gradients detected. Skipping optimizer step.")
                    self.optimizer.zero_grad()
                    continue
                    
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> Tuple[float, Dict]:
        """Evaluate model.
        
        Args:
            val_loader: Validation dataloader
            loss_fn: Loss function
        
        Returns:
            Tuple of (average loss, metrics dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(
            val_loader,
            desc="Evaluating",
            disable=False,
        )
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            logits = self.model(input_ids, attention_mask)
            loss = loss_fn(logits.squeeze(), labels)
            
            total_loss += loss.item()
            
            # Get predictions
            probs = torch.sigmoid(logits.squeeze())
            preds = (probs >= 0.5).long().cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().astype(int))
            
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        tp = np.sum((all_preds == 1) & (all_labels == 1))
        fp = np.sum((all_preds == 1) & (all_labels == 0))
        fn = np.sum((all_preds == 0) & (all_labels == 1))
        tn = np.sum((all_preds == 0) & (all_labels == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(all_labels)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        loss_fn: nn.Module,
        patience: int = 3,
    ) -> Dict:
        """Full training loop.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of epochs
            loss_fn: Loss function
            patience: Early stopping patience
        
        Returns:
            Training history dictionary
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": [],
        }
        
        best_val_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, loss_fn)
            history["train_loss"].append(train_loss)
            logger.info(f"Train loss: {train_loss:.4f}")
            
            # Evaluate
            val_loss, metrics = self.evaluate(val_loader, loss_fn)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(metrics)
            
            logger.info(f"Val loss: {val_loss:.4f}")
            logger.info(f"Metrics: {metrics}")
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Early stopping
            if metrics["f1"] > best_val_f1:
                best_val_f1 = metrics["f1"]
                patience_counter = 0
                self.save_checkpoint(epoch, metrics)
                logger.info(f"Best F1 score: {best_val_f1:.4f}, saving checkpoint")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    break
        
        return history
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Evaluation metrics
        """
        checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
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
    """Compute class weights to handle imbalance.
    
    Using inverse frequency weighting.
    
    Args:
        labels: Array of binary labels
    
    Returns:
        Tensor of class weights
    """
    counts = np.bincount(labels)
    total = len(labels)
    
    # Weight is inversely proportional to frequency
    weights = total / (2 * counts)
    
    return torch.tensor(weights, dtype=torch.float)
