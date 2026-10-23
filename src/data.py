"""Data loading and preprocessing pipeline."""

import math
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
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


class BPETokenizer:
    """Byte-level BPE subword tokenizer, built on HuggingFace `tokenizers`.

    Has the same interface as SimpleTokenizer (build_vocab, encode, decode,
    token2id), so the dataset and model don't need to know which one is used.

    Byte-level means any string can be encoded without <UNK>: unseen or
    obfuscated words such as "sh*tty" become several subword pieces instead
    of disappearing, and emoticons such as ":(" become real tokens.
    """

    def __init__(self, vocab_size: int = 10000):
        """Initialize tokenizer.

        Args:
            vocab_size: Maximum vocabulary size, including special tokens
        """
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(models.BPE())
        # Lowercase like SimpleTokenizer, so only the subword splitting differs
        self.tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.Lowercase(),
        ])
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.token2id = {}

    def build_vocab(self, texts: list):
        """Learn BPE merges from texts.

        Args:
            texts: List of text strings
        """
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            # Listed first so they get the same ids as in SimpleTokenizer
            special_tokens=["<PAD>", "<UNK>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        texts = [text for text in texts if isinstance(text, str)]
        self.tokenizer.train_from_iterator(texts, trainer=trainer)
        self.token2id = self.tokenizer.get_vocab()

    def encode(self, text: str) -> list:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(ids)

    def save(self, path: Union[str, Path]):
        """Save the trained tokenizer to a JSON file."""
        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BPETokenizer":
        """Load a tokenizer saved with save()."""
        bpe_tokenizer = cls()
        bpe_tokenizer.tokenizer = Tokenizer.from_file(str(path))
        bpe_tokenizer.token2id = bpe_tokenizer.tokenizer.get_vocab()
        bpe_tokenizer.vocab_size = len(bpe_tokenizer.token2id)
        return bpe_tokenizer


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
        
        if not isinstance(text, str):
            text = ""

        # Tokenize on the fly and truncate to max_seq_len
        input_ids = self.tokenizer.encode(text)[:self.max_seq_len]

        # Texts with no word tokens (e.g. ":(") become a single <UNK>, so
        # attention always has at least one position to attend to
        if len(input_ids) == 0:
            input_ids = [self.tokenizer.token2id["<UNK>"]]

        # No padding here: collate_fn pads each batch to its own longest sequence
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
        }


def collate_fn(batch: list) -> Dict:
    """Collate function for DataLoader.

    Pads every sequence to the longest one in this batch (not to max_seq_len),
    so short comments don't pay for padding they don't need.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched tensor dictionary
    """
    sequences = [item["input_ids"] for item in batch]
    lengths = torch.tensor([len(seq) for seq in sequences])

    input_ids = pad_sequence(sequences, batch_first=True, padding_value=0)

    # 1 for real tokens, 0 for padding
    attention_mask = (torch.arange(input_ids.size(1)) < lengths.unsqueeze(1)).long()

    labels = torch.stack([item["label"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class LengthGroupedBatchSampler(Sampler):
    """Yields batches of similar-length examples, in random order.

    Shuffles all indices, cuts them into chunks of `chunk_batches` batches,
    sorts each chunk by length and splits it into batches, then shuffles the
    order of the batches. Each batch needs very little padding, while the
    sequence of batches the model sees is still random.
    """

    def __init__(
        self,
        lengths: list,
        batch_size: int,
        seed: int = 42,
        reshuffle: bool = True,
        chunk_batches: int = 100,
    ):
        """Initialize sampler.

        Args:
            lengths: Approximate length of each example (only used for sorting)
            batch_size: Batch size
            seed: Random seed
            reshuffle: New order every epoch (training); if False, the same
                order every time (evaluation), so max_batches always picks the
                same random, unbiased subset
            chunk_batches: Number of batches per sorted chunk
        """
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.reshuffle = reshuffle
        self.chunk_size = batch_size * chunk_batches
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.reshuffle:
            self.epoch += 1

        indices = rng.permutation(len(self.lengths))
        batches = []
        for start in range(0, len(indices), self.chunk_size):
            chunk = indices[start:start + self.chunk_size]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            for batch_start in range(0, len(chunk), self.batch_size):
                batches.append(chunk[batch_start:batch_start + self.batch_size].tolist())

        rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)


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
    tokenizer_type: str = "word",
    tokenizer=None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Union[SimpleTokenizer, BPETokenizer]]:
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
        tokenizer_type: "word" (SimpleTokenizer) or "bpe" (BPETokenizer)
        tokenizer: Optional already-built tokenizer; if given, tokenizer_type
            and vocab_size are ignored and no vocabulary is built
        seed: Random seed for batch order

    Returns:
        Tuple of (train_loader, val_loader, test_loader, tokenizer)
    """
    # Build tokenizer on training data only, so val/test don't leak into the vocabulary
    if tokenizer is None:
        if tokenizer_type == "word":
            tokenizer = SimpleTokenizer(vocab_size=vocab_size)
        elif tokenizer_type == "bpe":
            tokenizer = BPETokenizer(vocab_size=vocab_size)
        else:
            raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")
        tokenizer.build_vocab(train_df[text_column].tolist())

    # Create dataloaders. Batches group comments of similar length so they need
    # little padding; character length is a good enough proxy for token length.
    def make_loader(df: pd.DataFrame, reshuffle: bool) -> DataLoader:
        dataset = ToxicCommentDataset(
            df[text_column].tolist(),
            df["label"].tolist(),
            tokenizer,
            max_seq_len=max_seq_len,
        )
        lengths = [len(text) if isinstance(text, str) else 0 for text in df[text_column]]
        return DataLoader(
            dataset,
            batch_sampler=LengthGroupedBatchSampler(
                lengths,
                batch_size,
                seed=seed,
                reshuffle=reshuffle,
            ),
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

    train_loader = make_loader(train_df, reshuffle=True)
    val_loader = make_loader(val_df, reshuffle=False)
    test_loader = make_loader(test_df, reshuffle=False)
    
    return train_loader, val_loader, test_loader, tokenizer
