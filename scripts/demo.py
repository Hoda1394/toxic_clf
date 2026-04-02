"""Quick demo script to test the model without full training."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ToxicCommentTransformer
from src.data import SimpleTokenizer


def demo():
    """Demo script showing model inference."""
    
    print("=" * 60)
    print("Toxic Comment Classifier - Demo")
    print("=" * 60)
    
    # Initialize model
    print("\n1. Initializing model...")
    model = ToxicCommentTransformer(
        vocab_size=10000,
        embedding_dim=128,
        num_layers=3,
        nhead=4,
        dim_feedforward=512,
        max_seq_len=256,
    )
    
    # Select device: CUDA > Apple Silicon (MPS) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    print(f"   Device: {device}")
    
    # Initialize tokenizer
    print("\n2. Initializing tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=10000)
    
    # Build vocab with sample data
    sample_texts = [
        "This is a nice comment",
        "This is offensive and hateful",
        "Great job everyone",
        "You are terrible",
        "I love this idea",
    ]
    tokenizer.build_vocab(sample_texts)
    print(f"   Vocabulary size: {len(tokenizer.token2id)}")
    
    # Test inference
    print("\n3. Testing inference on sample texts...")
    test_texts = [
        "This is a really great product, I love it!",
        "You are stupid and worthless",
        "I disagree with your opinion, but respect your view",
        "Go away you disgusting person",
    ]
    
    with torch.no_grad():
        for text in test_texts:
            # Tokenize
            input_ids = tokenizer.encode(text)
            input_ids = input_ids[:256]  # Truncate
            attention_mask = [1] * len(input_ids)
            
            # Pad
            padding = 256 - len(input_ids)
            input_ids = input_ids + [0] * padding
            attention_mask = attention_mask + [0] * padding
            
            # Convert to tensors
            input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to(device)
            
            # Inference
            logits = model(input_ids_tensor, attention_mask_tensor)
            prob = torch.sigmoid(logits).item()
            prediction = "TOXIC" if prob >= 0.5 else "CLEAN"
            
            print(f"\n   Text: \"{text}\"")
            print(f"   Prediction: {prediction} (confidence: {prob:.2%})")
    
    # Model info
    print("\n4. Model Information")
    print(f"   Architecture: 3-layer Transformer with pre-LN")
    print(f"   Embedding dim: 128")
    print(f"   Attention heads: 4")
    print(f"   Feedforward dim: 512")
    print(f"   Max sequence: 256 tokens")
    print(f"   Output: Binary (toxic/non-toxic)")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("- Prepare your data with columns: 'text', 'toxicity'")
    print("- Run: python scripts/train.py --data_path data/train.csv")
    print("- Evaluate: python scripts/evaluate.py --checkpoint_path checkpoints/model_epoch_0.pt")
    

if __name__ == "__main__":
    demo()
