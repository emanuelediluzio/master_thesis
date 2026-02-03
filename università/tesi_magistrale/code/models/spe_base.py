import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class FlashMHA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.batch_first = True
        self._qkv_same_embed_dim = True 
        self.in_proj_bias = None
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False, is_causal=False):
        # query: (batch, seq, embed_dim)
        B, S, D = query.shape
        qkv = self.qkv_proj(query)  # (B, S, 3*D)
        qkv = qkv.reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3, B, num_heads, S, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, num_heads, S, head_dim)
        
        # Standard scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, num_heads, S, S)
        
        if is_causal:
            causal_mask = torch.triu(torch.ones(S, S, device=query.device), diagonal=1).bool()
            scores.masked_fill_(causal_mask, float('-inf'))
        
        if attn_mask is not None:
            scores += attn_mask
            
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # (B, num_heads, S, head_dim)
        
        out = out.transpose(1, 2).reshape(B, S, D)  # (B, S, D)
        out = self.out_proj(out)
        
        if need_weights:
            return out, attn_weights
        return out

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PathEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, proj=False, num_tokens = 1024, dim_feedforward = 2048, use_flash=False, cls_mode='cls', **kwargs):
        super(PathEncoder, self).__init__()
        self.d_model = d_model
        self.num_tokens = num_tokens
        self.cls_mode = cls_mode
        self.proj = proj
        if proj:
            self.projection = nn.Linear(d_model, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        if use_flash:
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
            encoder_layer.self_attn = FlashMHA(d_model, nhead)
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        if cls_mode == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        elif cls_mode == 'mean':
            pass
        else:
            raise ValueError(f"Unknown cls_mode: {cls_mode}")

    def forward(self, x, mask=None, src_key_padding_mask=None, is_causal=None): # x : bs, seq_len, d_model
        if self.proj:
            x = self.projection(x)
        if self.cls_mode == 'cls':
            batch_size = x.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, seq_len + 1, d_model)
            if src_key_padding_mask is not None:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
                src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model) for pos_encoder
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # back to (batch_size, seq_len, d_model)
        
        output = self.transformer_encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
        
        if self.cls_mode == 'cls':
            cls_output = output[:, 0, :]  # Extract CLS token representation
            return cls_output
        elif self.cls_mode == 'mean':
            if src_key_padding_mask is not None:
                # Mask out padded positions
                mask_expanded = src_key_padding_mask.unsqueeze(-1).expand_as(output)
                output = output.masked_fill(mask_expanded, 0.0)
                # Compute mean over non-padded positions
                lengths = (~src_key_padding_mask).sum(dim=1, keepdim=True).float()
                mean_output = output.sum(dim=1) / lengths
            else:
                mean_output = output.mean(dim=1)
            return mean_output
        else:
            return output

class PathDecoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, num_tokens = 1024, dim_feedforward=2048, use_flash=False, **kwargs):
        super(PathDecoder, self).__init__()
        self.d_model = d_model
        self.num_tokens = num_tokens
        self.pos_encoder = PositionalEncoding(d_model)
        if use_flash:
            decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
            decoder_layer.self_attn = FlashMHA(d_model, nhead)
            decoder_layer.multihead_attn = FlashMHA(d_model, nhead)
        else:
            decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
    def forward(self, cls_vec, src, mask, src_key_padding_mask, is_causal=True):
        # cls_vec: (batch_size, d_model)
        # src: (batch_size, seq_len, d_model)
        
        # Expand cls_vec to match src sequence length
        batch_size, seq_len, d_model = src.shape
        cls_vec_expanded = cls_vec.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, d_model)
        
        # Apply positional encoding
        cls_vec_expanded = cls_vec_expanded.transpose(0, 1)  # (seq_len, batch_size, d_model)
        cls_vec_expanded = self.pos_encoder(cls_vec_expanded)
        cls_vec_expanded = cls_vec_expanded.transpose(0, 1)  # back to (batch_size, seq_len, d_model)
        
        # Decoder forward pass
        output = self.transformer_decoder(
            tgt=cls_vec_expanded,
            memory=src,
            tgt_mask=mask,
            tgt_key_padding_mask=src_key_padding_mask
        )
        
        return output