"""Data loading and preprocessing pipeline."""

import torch
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import re


class SimpleTokenizer:
    """Simple word-level tokenizer without external dependencies.
    
    This tokenizer:
    - Converts text to lowercase
    - Splits by whitespace and punctuation
    - Maps tokens to integer IDs
    - Handles unknown tokens
    """
    
    def __init__(self, vocab_size: int = 10000):
        """Initialize tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
        """
        self.vocab_size = vocab_size
        self.token2id = {"<PAD>": 0, "<UNK>": 1}
        self.id2token = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_built = False
    
    def build_vocab(self, texts: list):
        """Build vocabulary from texts.
        
        Args:
            texts: List of text strings
        """
        token_counts = Counter()
        
        for text in texts:
            tokens = self._tokenize(text)
            token_counts.update(tokens)
        
        # Add most common tokens to vocab
        for token, _ in token_counts.most_common(self.vocab_size - 2):
            token_id = len(self.token2id)
            self.token2id[token] = token_id
            self.id2token[token_id] = token
        
        self.vocab_built = True
    
    def _tokenize(self, text: str) -> list:
        """Tokenize text.
        
        Args:
            text: Input text
        
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Replace URLs and mentions
        text = re.sub(r'http\S+|www\S+', '<URL>', text)
        text = re.sub(r'@\w+', '<USER>', text)
        
        # Split by whitespace and basic punctuation
        tokens = re.findall(r'\b\w+\b|<\w+>', text)
        
        return tokens
    
    def encode(self, text: str) -> list:
        """Encode text to token IDs.
        
        Args:
            text: Input text
        
        Returns:
            List of token IDs
        """
        tokens = self._tokenize(text)
        ids = []
        for token in tokens:
            token_id = self.token2id.get(token, self.token2id["<UNK>"])
            ids.append(token_id)
        return ids
    
    def decode(self, ids: list) -> str:
        """Decode token IDs to text.
        
        Args:
            ids: List of token IDs
        
        Returns:
            Decoded text
        """
        tokens = [self.id2token.get(i, "<UNK>") for i in ids]
        return " ".join(tokens)


class ToxicCommentDataset(Dataset):
    """Dataset for toxic comment classification."""
    
    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer: SimpleTokenizer,
        max_seq_len: int = 256,
    ):
        """Initialize dataset.
        
        Args:
            texts: List of text strings
            labels: List of binary labels (0 or 1)
            tokenizer: Tokenizer instance
            max_seq_len: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary with input_ids, attention_mask, and label
        """
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Validate input
        if not isinstance(text, str) or len(text.strip()) == 0:
            # Return a dummy sample for empty/invalid text
            input_ids = [0] * self.max_seq_len
            attention_mask = [0] * self.max_seq_len
            label = 0.0
        else:
            # Tokenize on the fly
            input_ids = self.tokenizer.encode(text)
            
            # Validate tokenization result
            if len(input_ids) == 0:
                # Empty after tokenization, use dummy
                input_ids = [0] * self.max_seq_len
                attention_mask = [0] * self.max_seq_len
                label = 0.0
            else:
                # Truncate to max_seq_len
                input_ids = input_ids[:self.max_seq_len]
                
                # Create attention mask (1 for valid tokens, 0 for padding)
                attention_mask = [1] * len(input_ids)
                
                # Pad to max_seq_len
                padding_len = self.max_seq_len - len(input_ids)
                input_ids = input_ids + [0] * padding_len
                attention_mask = attention_mask + [0] * padding_len
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
        }


def collate_fn(batch: list) -> Dict:
    """Collate function for DataLoader.
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Batched tensor dictionary
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def load_civil_comments_data(
    csv_path: str,
    text_column: str = "text",
    target_column: str = "toxicity",
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and split Civil Comments dataset.
    
    Args:
        csv_path: Path to CSV file
        text_column: Name of text column
        target_column: Name of target column
        test_size: Test set proportion
        val_size: Validation set proportion (of remaining data)
        seed: Random seed
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df = pd.read_csv(csv_path)
    
    # Create binary labels (>= 0.5 is toxic)
    df["label"] = (df[target_column] >= 0.5).astype(int)
    
    # Remove rows with missing text or labels
    df = df.dropna(subset=[text_column, "label"])
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    
    # Second split: train vs val
    val_split = val_size / (1 - test_size)  # Adjust for already removed test
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_split,
        random_state=seed,
        stratify=train_val_df["label"],
    )
    
    return train_df, val_df, test_df


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = "text",
    batch_size: int = 32,
    max_seq_len: int = 256,
    vocab_size: int = 10000,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, SimpleTokenizer]:
    """Create DataLoaders for train/val/test sets.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        text_column: Name of text column
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        vocab_size: Vocabulary size
        num_workers: Number of workers for DataLoader
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, tokenizer)
    """
    # Build tokenizer on training data
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab(train_df[text_column].tolist())
    
    # Create datasets
    train_dataset = ToxicCommentDataset(
        train_df[text_column].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        max_seq_len=max_seq_len,
    )
    
    val_dataset = ToxicCommentDataset(
        val_df[text_column].tolist(),
        val_df["label"].tolist(),
        tokenizer,
        max_seq_len=max_seq_len,
    )
    
    test_dataset = ToxicCommentDataset(
        test_df[text_column].tolist(),
        test_df["label"].tolist(),
        tokenizer,
        max_seq_len=max_seq_len,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader, test_loader, tokenizer
