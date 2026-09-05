# Architecture and Design Report

## Executive Summary

This report details the design decisions for a custom Transformer-based toxic comment classifier built without pre-trained weights. The model is designed to be lightweight, efficient, and fully controllable for future optimizations in a high-volume production environment.

---

## 1. Architecture Choices

### 1.1 Pre-Layer Normalization (Pre-LN)

**Decision**: Use Pre-Layer Normalization instead of Post-LN

**Rationale**:
- **Training Stability**: Pre-LN applies normalization before sub-layers, which stabilizes the training dynamics and prevents gradient explosion
- **Faster Convergence**: Models with Pre-LN converge faster and require less hyperparameter tuning
- **Better Scaling**: Pre-LN has better performance as model depth increases
- **Reference**: Empirically shown in "Pre Layernorm Transformer" (https://arxiv.org/abs/2002.04745)

**Implementation**:
```
Input → LayerNorm → MultiheadAttention → Residual → Next Block
       → LayerNorm → FeedForward → Residual → Next Block
```

### 1.2 Custom Transformer Architecture

**Choice of Components**:
- **nn.MultiheadAttention**: PyTorch's optimized implementation for self-attention
- **nn.Linear**: Simple dense layers for feedforward network
- **nn.LayerNorm**: Layer normalization for stability
- **Learnable Positional Embeddings**: Instead of sinusoidal, to give model flexibility
- **GELU Activation**: Smoother than ReLU, better for language tasks (or RELU)

### 1.3 Model Size

**Default Configuration**:
- Embedding dimension: 128
- Number of layers: 3
- Attention heads: 4
- Feedforward dimension: 512
- Vocabulary: 10,000 tokens
- Max sequence: 128 tokens

**Total Parameters**: ~4-5 million (~5MB)

**Rationale**:
- Lightweight enough for serverless/edge deployment
- Still large enough to capture semantic information
- Balances between model capacity and inference speed
- Can be scaled up (embedding_dim=256, num_layers=6) if needed

---

## 2. Loss Function Choice

### 2.1 Binary Cross Entropy with Logits

**Decision**: `BCEWithLogitsLoss` with class weighting

**Why BCEWithLogitsLoss?**
1. **Numerical Stability**: Combines sigmoid and BCE for numerical stability
2. **No Manual Sigmoid**: Model outputs logits directly
3. **Class Weighting Support**: Native `pos_weight` parameter for imbalance
4. **Standard for Binary Classification**: Industry standard choice

**Implementation**:
```python
loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weight)
```

Where:
$$pos\_weight = \frac{\text{count(negative)}}{\text{count(positive)}}$$

This makes positive examples contribute more to the loss, compensating for class imbalance.

### 2.2 Threshold and Output

- **Output**: Single neuron with sigmoid activation at inference time
- **Threshold**: 0.5 for binary classification
- **Formula**: prediction = (sigmoid(logit) >= 0.5) ? 1 : 0

---

## 3. Class Imbalance Handling

### 3.1 Multi-Level Approach

**Problem**: Toxic comments are typically a minority (~5-15% in civil comments dataset)

**Solution Stack**:

1. **Stratified Sampling** (Primary)
   - Maintains label distribution across train/val/test splits
   - Prevents train/test mismatch
   - Implemented: `train_test_split(..., stratify=labels)`

2. **Weighted Loss** (Secondary)
   - BCEWithLogitsLoss with pos_weight
   - Gives higher penalty for misclassifying positive (toxic) examples
   - Computes weights as: `weights = total_count / (2 * class_counts)`

3. **Evaluation Metrics** (Tertiary)
   - Don't rely solely on accuracy (misleading with imbalance)
   - Report: Precision, Recall, F1 Score
   - F1 balances both false positives and false negatives

### 3.2 Why Not Other Methods?

- **Oversampling**: Would increase training time, not used at scale
- **Undersampling**: Wastes data, not used for initial training
- **Hard Negative Mining**: More suited for ensemble methods
- **SMOTE**: Overkill for relatively small imbalance ratio

---

## 4. Hyperparameter Justification

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **vocab_size** | 10,000 | Covers ~95% of common English; good balance |
| **embedding_dim** | 128 | Power of 2; divisible by nhead; good capacity |
| **num_layers** | 3 | Lightweight; enough for toxic phrase patterns |
| **nhead** | 4 | embedding_dim (128) / nhead (4) = 32 (standard) |
| **dim_feedforward** | 512 | 4x embedding_dim (standard recommendation) |
| **max_seq_len** | 256 | Covers 95% of real-world comments; manageable |
| **dropout** | 0.1 | Moderate regularization; prevents overfitting |
| **batch_size** | 32 | Balanced between gradient stability and memory |
| **learning_rate** | 1e-7 | Conservative starting point for AdamW |
| **weight_decay** | 1e-5 | Light L2 regularization |

---

## 5. Data Pipeline Design

### 5.1 On-the-Fly vs Pre-Processed Tokenization

**Decision**: On-the-fly tokenization

**Advantages**:
- **Storage**: No need to store tokenized data (CSV vs tokenized tensors)
- **Flexibility**: Can change tokenizer without re-processing entire dataset
- **Memory**: Doesn't require loading full pre-tokenized dataset into memory
- **Processing**: Can apply different tokenization strategies in different epochs

**Trade-off**:
- Slight computational overhead during training (2-5%)
- Offset by reduced storage and preparation time

### 5.2 SimpleTokenizer Implementation

**Features**:
- Word-level tokenization (no subword)
- Handles URLs (`<URL>`) and mentions (`<USER>`)
- Unknown token normalization (`<UNK>`)
- No external dependencies (spaCy, NLTK)

**Limitations**:
- Not handling morphology (e.g., "running", "runs" are different tokens)
- Better tokenizer would use BPE (Byte Pair Encoding) for subword handling
- Future improvement: Integrate tokenizers library

### 5.3 Data Split Strategy

```
Raw Data (100%)
├── Train (70%)
│   ├── Stratified by label
│   └── Used for tokenizer vocab building
├── Validation (10%)
│   └── Stratified by label
│   └── Used for early stopping
└── Test (20%)
    └── Stratified by label
    └── Used for final evaluation
```

**Stratification Ensures**: Each split has similar toxic/non-toxic ratio as original

---

## 6. Training Strategy

### 6.1 Optimizer and Scheduling

- **Optimizer**: AdamW (Adam with weight decay)
  - Better generalization than Adam
  - Weight decay decoupled from gradient updates
  
- **Learning Rate Scheduler**: ReduceLROnPlateau
  - Reduces LR when validation loss plateaus
  - Adaptive to convergence speed
  - Factor: 0.5 (halve on plateau)

### 6.2 Early Stopping

- **Metric**: Validation F1 Score
- **Patience**: 3 epochs
- **Criterion**: Stop if no improvement for 3 consecutive epochs
- **Checkpoint**: Save model at each best F1

### 6.3 Learning Rate Warmup

**What is Warmup?**

Learning rate warmup is a strategy where the learning rate starts very small and gradually increases to the target learning rate over the first N steps/epochs. This is crucial for Transformer training stability.

**Implementation**:
```python
warmup_steps = 100
initial_lr = 1e-7
for step in range(num_steps):
    if step <= warmup_steps:
        current_lr = initial_lr * (step / warmup_steps)  # Linear warmup
    else:
        current_lr = initial_lr  # Normal learning rate
```

**Why Warmup is Critical for Transformers**:

1. **Prevents Gradient Explosion**: 
   - At initialization, gradients can be very large (>1e5)
   - Jumping directly to normal LR causes numerical instability
   - Warmup gradually "ramps up" to safe learning rate

2. **Better Convergence**:
   - Warmup helps model settle into good local minima
   - Without warmup: model may diverge or get stuck in poor local minima
   - With warmup: smoother, more stable training trajectory

3. **Handles Initialization Variance**:
   - Different random seeds cause different initialization magnitudes
   - Warmup makes training robust to initialization

4. **Empirical Results**:
   - With warmup: consistent convergence across runs
   - Without warmup: occasional NaN loss, divergence
   - Prevents the "training becomes unstable" pattern

**Our Configuration**:
- Warmup steps: 100 (can adjust based on dataset size)
- Warmup strategy: Linear (other options: exponential, cosine)
- Starting LR: 1e-5 (very conservative)
- Target LR: 5e-5 (still conservative for Transformers)

**Visual Example**:
```
Learning Rate over Steps:
│     ┌─────────────────────────── (target LR = 5e-5)
│    /
│   /  (warmup phase)
│  /
└────────────── (start LR ≈ 0)
  └─→ warmup_steps (100)
```

---

### 6.4 Gradient Norm Clipping

**What is Gradient Clipping?**

Gradient clipping caps (clips) the gradient norms to prevent them from becoming too large. This is done after the backward pass but before the optimizer step.

**Implementation**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
optimizer.step()
```

**The Problem: Exploding Gradients**

During backpropagation in deep networks (especially Transformers), gradients can grow exponentially:

```
Layer 1 → Layer 2 → Layer 3 → ... → Layer 12
∂L/∂w₁₂ = (∂L/∂out) × ∂out/∂h₁₁ × ∂h₁₁/∂h₁₀ × ... × ∂h₂/∂w₁₂
           (product of 12+ derivatives)
```

If each derivative > 1: gradient explodes exponentially
If each derivative < 1: gradient vanishes exponentially

**Symptoms of Exploding Gradients**:
- Loss becomes `NaN` or `Inf`
- Training diverges (loss → ∞)
- Model parameters become `NaN`
- Occasional batches cause complete training failure

**How Clipping Fixes It**:

```python
# Before clipping:
gradient_norm = torch.norm(gradient)  # e.g., 500.0

# If gradient_norm > max_norm (0.25):
gradient = gradient * (max_norm / gradient_norm)
# After clipping: gradient_norm = 0.25
```

**Why max_norm = 0.25?**

- **Too high (>1.0)**: Still allows moderate gradient explosion
- **Too low (<0.1)**: May overly constrain gradients, slow training
- **0.25**: Empirically works well for Transformers
- **Our aggressive value**: We use 0.25 (not standard 1.0) due to:
  - Class weighting amplifying gradients
  - NaN detection findings in training

**Clipping Strategy Comparison**:

| Strategy | max_norm | Use Case |
|----------|----------|----------|
| Conservative | 0.1 | Very unstable models |
| **Aggressive** (ours) | **0.25** | **Transformers with large class weights** |
| Standard | 0.5-1.0 | RNNs, LSTM |
| Off | None | Well-initialized models with careful LR |

**When Combined with Warmup**:

Warmup + Clipping is the "gold standard" for Transformer training:
1. Warmup prevents early-stage gradient explosion
2. Clipping handles mid-training gradient spikes
3. Together they enable stable, reproducible training

---

### 6.5 Automatic Mixed Precision (Optional)

- **Optional flag**: `--use_amp`
- **Benefits**: 2x faster training, 50% less memory
- **Trade-off**: Slightly lower precision, but negligible for this task

---

## 7. Multi-GPU Support

### 7.1 Current Implementation: DataParallel

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

**Pros**:
- Simple to use
- Works immediately with minimal code changes

**Cons**:
- Only works on single machine
- Synchronization overhead (~20% slower than single GPU)

### 7.2 For Production / Large Scale: DistributedDataParallel

Would require:
```python
torch.distributed.init_process_group("nccl")  # Backend
model = torch.nn.parallel.DistributedDataParallel(model)
```

Benefits:
- Works across multiple machines
- More efficient synchronization
- Better gradual scaling

---

## 8. Performance Metrics

### 8.1 Reported Metrics

**Accuracy**: $(TP + TN) / Total$
- Simple overall correctness
- Misleading with imbalanced data

**Precision**: $TP / (TP + FP)$
- "Of comments flagged as toxic, how many are actually toxic?"
- Important for false positive cost

**Recall**: $TP / (TP + FN)$  
- "Of all toxic comments, how many do we catch?"
- Important for false negative cost

**F1 Score**: $2 \times \frac{Precision \times Recall}{Precision + Recall}$
- Harmonic mean of precision and recall
- Balanced metric; used for model selection

### 8.2 Why F1 for Model Selection?

- Accuracy is misleading with imbalanced data
- F1 naturally balances both false positives and false negatives
- Production typically cares equally about both types of errors
- Alternative: Use business metrics if available (e.g., cost per error type)

---

## 9. Deployment Considerations: Post-Mortem

### Scenario: 10,000 requests/second in production

#### Current Bottlenecks

1. **Single GPU bottleneck**: ~1,000 req/sec per GPU
   - Need 10 GPUs to reach 10k req/sec
   - Cost prohibitive

2. **Model size acceptable**: 5MB is deployable

3. **Latency concerns**: 
   - Per-request: ~1-2ms (acceptable)
   - Cold start: Not an issue for warm model

### Required Changes for 10k req/sec

#### 1. **Batch Inference Pipeline**
- Group requests into batches (32-64)
- Use request queuing system
- Reduces per-request overhead by 10x

#### 2. **Model Quantization**
```python
# Option 1: Post-training quantization
import torch.quantization as quant
quantized_model = quant.quantize_dynamic(model)
# 4x smaller, 2x faster

# Option 2: Quantization-aware training
# Better accuracy; requires retraining
```

#### 3. **Model Export and Optimization**
```python
# TorchScript for faster inference
scripted_model = torch.jit.script(model)

# ONNX for cross-platform optimization
torch.onnx.export(model, dummy_input, "model.onnx")

# NVIDIA TensorRT for 2-3x speedup
# TVM for mobile/embedded deployment
```

#### 5. **Infrastructure**
- **Batch serving**: Use ray serve, KServe, or Triton
- **GPU cluster**: 10+ A100s with load balancing
- **Caching**: Cache predictions for identical/similar inputs
- **Async processing**: Return results via callback/stream

#### 6. **Model Pruning**
```python
# Prune 40-50% of weights
torch.nn.utils.prune.l1_unstructured(linear_layer, 'weight', amount=0.4)
# 2x speedup, < 5% accuracy loss
```

#### 7. **Reduced Architecture**
- embedding_dim: 128 → 64
- num_layers: 3 → 2
- dim_feedforward: 512 → 256
- vocab_size: 10k → 5k

**Impact**: smaller model with faster inference and acceptable accuracy loss

---

## 8. Conclusion

This architecture represents a pragmatic balance between:
- **Simplicity**: From-scratch implementation, no external dependencies for model
- **Performance**: Lightweight yet effective
- **Scalability**: Easy to scale horizontally (batch serving)
- **Control**: Full visibility and customization for future optimizations
- **Robustness**: Handles class imbalance, supports multi-GPU

The design is production-ready for small to medium-scale deployments, and provides clear path to 10k req/sec through well-understood optimization strategies.

#### **Author: Hoda Rajaei**
