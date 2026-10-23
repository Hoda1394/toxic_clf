"""
Configuration file for hyperparameters.
Can be extended to support loading from YAML/JSON files.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """Data configuration."""
    data_path: str = "data/train.csv"
    text_column: str = "text"
    target_column: str = "target"
    test_size: float = 0.2
    val_size: float = 0.1
    max_seq_len: int = 256
    vocab_size: int = 10000
    tokenizer: str = "word"
    seed: int = 42


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    vocab_size: int = 10000
    embedding_dim: int = 128
    num_layers: int = 3
    nhead: int = 4
    dim_feedforward: int = 512
    max_seq_len: int = 256
    dropout: float = 0.1
    num_classes: int = 1


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 10
    early_stopping_patience: int = 3
    checkpoint_dir: str = "checkpoints"
    use_amp: bool = False


@dataclass
class Config:
    """Complete configuration."""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()


# Default configurations
SMALL_MODEL = Config(
    model=ModelConfig(
        embedding_dim=64,
        num_layers=2,
        nhead=2,
        dim_feedforward=256,
    ),
)

MEDIUM_MODEL = Config(
    model=ModelConfig(
        embedding_dim=256,
        num_layers=4,
        nhead=8,
        dim_feedforward=1024,
    ),
)

AGGRESSIVE_TRAINING = Config(
    training=TrainingConfig(
        batch_size=64,
        learning_rate=2e-3,
        num_epochs=20,
    ),
)
