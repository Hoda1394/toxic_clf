# Toxic Comment Classifier

A custom lightweight Transformer-based classifier for detecting toxic comments in high-volume text data.

## Overview

This project implements a custom Transformer architecture from scratch using PyTorch (no pre-trained weights) for binary classification of toxic comments from the [Civil Comments dataset](https://www.kaggle.com/datasets/jigsaw-unintended-bias-in-toxicity-classification). The model uses pre-layer normalization for improved training stability.

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd toxic-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install as a package:
```bash
pip install -e .
```

### Training

Basic usage:
```bash
python scripts/train.py --data_path data/train.csv
```

With custom hyperparameters:
```bash
python scripts/train.py \
    --data_path data/train.csv \
    --batch_size 64 \
    --num_epochs 20 \
    --learning_rate 0.001 \
    --embedding_dim 256 \
    --num_layers 4
```

### Evaluation

Evaluate a trained model:
```bash
python scripts/evaluate.py \
    --checkpoint_path checkpoints/model_epoch_5.pt \
    --data_path data/test.csv
```

## Architecture Details

### Custom Transformer

The model implements a lightweight Transformer encoder with the following features:

- **Pre-Layer Normalization**: Applies layer norm before attention and feedforward, improving training stability
- **Multi-Head Attention**: Self-attention mechanism for capturing token relationships
- **Positional Encoding**: Learnable positional embeddings to encode sequence positions
- **Feedforward Network**: Two-layer feedforward with GELU activation

Key hyperparameters:
- `embedding_dim`: 128 (embedding dimension)
- `num_layers`: 3 (number of transformer blocks)
- `nhead`: 4 (number of attention heads)
- `dim_feedforward`: 512 (width of intermediate layer)
- `max_seq_len`: 256 (maximum sequence length)

### Data Pipeline

- **SimpleTokenizer**: Custom word-level tokenizer that:
  - Converts to lowercase
  - Handles URLs and mentions
  - Maps tokens to integer IDs
  - No external spaCy/NLTK dependencies

- **On-the-fly Tokenization**: Text is tokenized during batch loading, not pre-processed, allowing:
  - Reduced storage requirements
  - Flexibility for future tokenization changes
  - Efficient memory usage

- **Stratified Splits**: Data is split with stratification to maintain class balance:
  - 80% train, 10% val, 10% test

### Class Imbalance Handling

1. **Stratified Sampling**: Maintains label distribution across splits
2. **Weighted Loss**: BCEWithLogitsLoss with `pos_weight` to give higher weight to toxic examples:
   - `pos_weight = count(negative) / count(positive)`

### Loss Function

**Binary Cross Entropy with Logits (BCEWithLogitsLoss)**: 
- Numerically stable compared to sigmoid + BCE
- Supports pos_weight for handling class imbalance
- Output is binary classification: toxic (1) or non-toxic (0)

## Multi-GPU Training

The trainer automatically detects available GPUs:

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

This uses `nn.DataParallel` for simple multi-GPU distribution. For distributed training across multiple machines, use:

```python
torch.distributed.launch --nproc_per_node=4 scripts/train.py
```

And wrap the model with `DistributedDataParallel`.

## Data Format

CSV file should contain at minimum:
- `text` column: comment text
- `toxicity` column: toxicity score (0-1)

Labels are created as: `label = 1 if toxicity >= 0.5 else 0`

## Project Structure

```
toxic-classifier/
├── src/
│   ├── __init__.py
│   ├── model.py          # Transformer architecture
│   ├── data.py           # Data loading and preprocessing
│   └── trainer.py        # Training and evaluation loop
├── scripts/
│   ├── train.py          # Main training script
│   └── evaluate.py       # Evaluation script
├── requirements.txt      # Dependencies
├── setup.py              # Package setup
└── README.md             # This file
```

## Configuration

All hyperparameters can be configured via command-line arguments. See `scripts/train.py --help` for full list.

Key hyperparameters:

```python
--vocab_size 10000           # Vocabulary size
--embedding_dim 128          # Embedding dimension
--num_layers 3               # Number of transformer layers
--nhead 4                    # Number of attention heads
--dim_feedforward 512        # Feedforward dimension
--max_seq_len 256            # Maximum sequence length
--batch_size 32              # Batch size
--learning_rate 0.001        # Learning rate
--num_epochs 10              # Number of epochs
--early_stopping_patience 3  # Early stopping patience
```

## Model Checkpoints

Checkpoints are saved to `checkpoints/` directory after each improvement in validation F1 score. Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Epoch number
- Validation metrics

## Reporting Metrics

After training, the model reports:
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP) - How many predicted toxic are actually toxic
- **Recall**: TP / (TP + FN) - What fraction of actual toxic comments are caught
- **F1 Score**: Harmonic mean of precision and recall

## Dependencies

- `torch>=1.9.0`: Deep learning framework
- `pandas>=1.3.0`: Data manipulation
- `numpy>=1.21.0`: Numerical computing
- `scikit-learn>=0.24.0`: ML utilities
- `tqdm>=4.62.0`: Progress bars

## Optional: Docker

To run in Docker:

```bash
docker build -t toxic-classifier .
docker run --gpus all -v /path/to/data:/data toxic-classifier
```

## Performance Characteristics

- **Model Size**: ~4-5MB for the default configuration
- **Inference Speed**: ~1000+ predictions/second on a single GPU
- **Memory**: ~500MB during training with batch size 32

## Known Limitations

1. Custom tokenizer is basic; consider using subword tokenization for production
2. No special handling for out-of-vocabulary words beyond `<UNK>`
3. No automatic batch size optimization based on available memory
4. `nn.DataParallel` has synchronization overhead; `DistributedDataParallel` recommended for large clusters

## Future Improvements

- [ ] Add BERT tokenization for better subword handling
- [ ] Implement DistributedDataParallel for multi-machine training
- [ ] Add hyperparameter search (Ray Tune)
- [ ] Add model quantization for deployment
- [ ] Implement inference batch pipeline
- [ ] Add confidence calibration
- [ ] Support for model export (ONNX, TorchScript)

## References

- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Pre Layernorm Transformer: https://arxiv.org/abs/2002.04745
- Civil Comments Dataset: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification

## License

MIT

## Contact

For questions or issues, please open an issue on GitHub.
