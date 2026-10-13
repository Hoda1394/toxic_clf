"""Custom Transformer architecture for toxic comment classification."""

import torch
import torch.nn as nn
from typing import Optional


class TransformerEncoder(nn.Module):
    """Single Transformer encoder layer with pre-layer normalization.
    
    Pre-layer normalization (Pre-LN) is applied before each sub-layer,
    which has shown to provide better training stability than post-LN.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        """Initialize TransformerEncoder.
        
        Args:
            d_model: Embedding dimension
            nhead: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            activation: Activation function ("relu" or "gelu")
        """
        super().__init__()
        
        # Pre-layer normalization before self-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Pre-layer normalization before feedforward
        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Initialize weights for better training stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-layer normalization.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Attention mask of shape (batch, seq_len)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Pre-layer norm + self-attention
        normed = self.norm1(x)
        
        attn_out, _ = self.self_attn(
            normed, normed, normed, key_padding_mask=mask
        )
        
        x = x + self.dropout1(attn_out)
        
        # Pre-layer norm + feedforward
        normed = self.norm2(x)
        
        ff_out = self.feedforward(normed)
        
        x = x + self.dropout2(ff_out)
        return x


class ToxicCommentTransformer(nn.Module):
    """Custom lightweight Transformer for toxic comment classification."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_layers: int = 3,
        nhead: int = 4,
        dim_feedforward: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        num_classes: int = 1,
    ):
        """Initialize ToxicCommentTransformer.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension (d_model)
            num_layers: Number of Transformer encoder layers
            nhead: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            num_classes: Number of output classes (1 for binary with sigmoid)
        """
        super().__init__()
        
        assert embedding_dim % nhead == 0, "embedding_dim must be divisible by nhead"
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Positional encoding (learnable)
        self.positional_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        # Dropout after embedding
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(
                d_model=embedding_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes),
        )
        
        # Initialize weights for better training stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if 'embedding' in name:
                # Initialize embeddings with smaller variance
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)
                           1 for valid tokens, 0 for padding
        
        Returns:
            Logits of shape (batch, num_classes)
        """
        batch_size, seq_len = input_ids.size()
        
        # Create attention mask for padding (True where to mask)
        mask = None
        if attention_mask is not None:
            mask = attention_mask == 0  # True for padding tokens
        
        # Token embedding
        x = self.embedding(input_ids)
        
        # # Clamp embeddings to reasonable range
        # x = torch.clamp(x, -5, 5)
        
        # Add positional embedding
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.positional_embedding(positions)
        x = x + pos_emb
        
        x = self.embedding_dropout(x)
        
        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask=mask)
            
            # # Check for NaN after each layer
            # if torch.isnan(x).any():
            #     # Reset to previous valid state if NaN detected
            #     x = torch.zeros_like(x) + 1e-6  # Small constant to avoid zero
        
        # Global average pooling (ignore padding tokens)
        if attention_mask is not None:
            # Average over valid tokens only
            mask_expanded = attention_mask.unsqueeze(-1).expand(x.shape)
            sum_pooled = (x * mask_expanded).sum(1)
            count = attention_mask.sum(1, keepdim=True)
            x = sum_pooled / count.clamp(min=1)
        else:
            x = x.mean(1)  # Average pooling over sequence
        
        # # Clamp pooled output
        # x = torch.clamp(x, -10, 10)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        # Classification head
        logits = self.classifier(x)
        
        # # Final clamp on logits to prevent extreme values
        # logits = torch.clamp(logits, -10, 10)
        
        return logits
