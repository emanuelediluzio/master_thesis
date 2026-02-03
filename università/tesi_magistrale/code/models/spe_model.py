#!/usr/bin/env python3
"""
Modello SPE (Structured Path Encoder) per SVG
Implementazione del modello encoder per path SVG strutturati.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding per sequenze SVG.
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Crea matrice di positional encoding
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, d_model]
        Returns:
            [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention per SPE.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Scaled dot-product attention.
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        
        return output, attention_weights

class FeedForward(nn.Module):
    """
    Feed-forward network per transformer.
    """
    
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """
    Blocco transformer per SPE.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class SPEModel(nn.Module):
    """
    Structured Path Encoder (SPE) per SVG.
    Modello transformer specializzato per encoding di path SVG.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 1024,
        num_heads: int = 16,
        num_layers: int = 12,
        d_ff: int = 4096,
        max_seq_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Pooling strategies
        self.pooling_strategy = "mean"  # "mean", "max", "cls", "last"
        
        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Initialize weights
        self._init_weights()
        
        print(f"✅ SPEModel inizializzato:")
        print(f"   - vocab_size: {vocab_size}")
        print(f"   - d_model: {d_model}")
        print(f"   - num_heads: {num_heads}")
        print(f"   - num_layers: {num_layers}")
        print(f"   - max_seq_length: {max_seq_length}")
    
    def _init_weights(self):
        """Inizializzazione dei pesi."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def create_padding_mask(self, input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
        """
        Crea maschera per padding.
        
        Args:
            input_ids: [batch_size, seq_len]
            pad_token_id: ID del token di padding
            
        Returns:
            torch.Tensor: [batch_size, 1, 1, seq_len]
        """
        mask = (input_ids != pad_token_id).unsqueeze(1).unsqueeze(1)
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False
    ) -> torch.Tensor:
        """
        Forward pass del modello SPE.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] optional
            return_all_hidden_states: Se restituire tutti gli stati nascosti
            
        Returns:
            torch.Tensor: [batch_size, seq_len, d_model] o [batch_size, d_model] se pooled
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        embeddings = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        
        # Add CLS token if using CLS pooling
        if self.pooling_strategy == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat([cls_tokens, embeddings], dim=1)
            seq_len += 1
        
        # Scale embeddings
        embeddings = embeddings * math.sqrt(self.d_model)
        
        # Positional encoding
        embeddings = embeddings.transpose(0, 1)  # [seq_len, batch_size, d_model]
        embeddings = self.positional_encoding(embeddings)
        embeddings = embeddings.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.create_padding_mask(input_ids)
        else:
            if self.pooling_strategy == "cls":
                # Add mask for CLS token
                cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
                attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        
        # Transformer blocks
        hidden_states = embeddings
        all_hidden_states = []
        
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, attention_mask)
            if return_all_hidden_states:
                all_hidden_states.append(hidden_states)
        
        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        
        return hidden_states
    
    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode input e applica pooling per ottenere rappresentazione fissa.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] optional
            
        Returns:
            torch.Tensor: [batch_size, d_model]
        """
        hidden_states = self.forward(input_ids, attention_mask)
        
        # Apply pooling
        if self.pooling_strategy == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                pooled = hidden_states.mean(dim=1)
        
        elif self.pooling_strategy == "max":
            pooled, _ = hidden_states.max(dim=1)
        
        elif self.pooling_strategy == "cls":
            pooled = hidden_states[:, 0]  # CLS token
        
        elif self.pooling_strategy == "last":
            if attention_mask is not None:
                # Last non-padded token
                seq_lengths = attention_mask.sum(dim=1) - 1
                pooled = hidden_states[torch.arange(hidden_states.size(0)), seq_lengths]
            else:
                pooled = hidden_states[:, -1]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return pooled
    
    def set_pooling_strategy(self, strategy: str):
        """Imposta strategia di pooling."""
        valid_strategies = ["mean", "max", "cls", "last"]
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
        self.pooling_strategy = strategy
        print(f"📊 Pooling strategy set to: {strategy}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Restituisce informazioni sul modello."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "SPE (Structured Path Encoder)",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "max_seq_length": self.max_seq_length,
            "num_layers": len(self.transformer_blocks),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "pooling_strategy": self.pooling_strategy,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def save_pretrained(self, save_path: str):
        """Salva il modello."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Salva state dict
        torch.save(self.state_dict(), os.path.join(save_path, "spe_model.pth"))
        
        # Salva config
        config = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "max_seq_length": self.max_seq_length,
            "pooling_strategy": self.pooling_strategy
        }
        
        import json
        with open(os.path.join(save_path, "spe_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 SPE model saved to: {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str):
        """Carica il modello."""
        import os
        import json
        
        # Carica config
        config_path = os.path.join(load_path, "spe_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Crea modello
        model = cls(**config)
        
        # Carica weights
        weights_path = os.path.join(load_path, "spe_model.pth")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        
        print(f"📂 SPE model loaded from: {load_path}")
        return model

# Funzioni di utilità
def create_spe_model_small():
    """Crea un modello SPE piccolo per test."""
    return SPEModel(
        vocab_size=8000,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_length=256
    )

def create_spe_model_base():
    """Crea un modello SPE base."""
    return SPEModel(
        vocab_size=32000,
        d_model=1024,
        num_heads=16,
        num_layers=12,
        d_ff=4096,
        max_seq_length=512
    )

def create_spe_model_large():
    """Crea un modello SPE large."""
    return SPEModel(
        vocab_size=50000,
        d_model=1536,
        num_heads=24,
        num_layers=24,
        d_ff=6144,
        max_seq_length=1024
    )