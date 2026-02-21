#!/usr/bin/env python3
"""
🤖 MODELLO MULTIMODALE
Integra encoder immagini con LLM per input multimodali
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DimensionalityAdapter(nn.Module):
    """Adatta dimensionalità encoder → LLM"""
    
    def __init__(self, encoder_dim: int, llm_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        
        # Projection layer
        self.projection = nn.Linear(encoder_dim, llm_dim)
        self.layer_norm = nn.LayerNorm(llm_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Activation
        self.activation = nn.GELU()
        
    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embeddings: [batch_size, seq_len, encoder_dim]
        Returns:
            projected_embeddings: [batch_size, seq_len, llm_dim]
        """
        # Project to LLM dimension
        projected = self.projection(image_embeddings)
        projected = self.activation(projected)
        projected = self.layer_norm(projected)
        projected = self.dropout(projected)
        
        return projected

class MultimodalLLM(nn.Module):
    """Modello LLM multimodale con encoder immagini"""
    
    def __init__(
        self,
        llm_model_name: str,
        encoder_dim: int,
        adapter_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        # Load LLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Get LLM hidden dimension
        self.llm_dim = self.llm.config.hidden_size
        
        # Dimensionality adapter
        adapter_config = adapter_config or {}
        self.adapter = DimensionalityAdapter(
            encoder_dim=encoder_dim,
            llm_dim=self.llm_dim,
            dropout=adapter_config.get('dropout', 0.1)
        )
        
        # Special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Add special tokens for multimodal
        special_tokens = {
            "additional_special_tokens": ["<image>", "</image>", "<visual>", "</visual>"]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
        logger.info(f"Initialized MultimodalLLM:")
        logger.info(f"  LLM: {llm_model_name}")
        logger.info(f"  LLM dim: {self.llm_dim}")
        logger.info(f"  Encoder dim: {encoder_dim}")
        logger.info(f"  Vocab size: {len(self.tokenizer)}")
    
    def forward(
        self,
        image_embeddings: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass multimodale
        
        Args:
            image_embeddings: [batch_size, img_seq_len, encoder_dim]
            input_ids: [batch_size, text_seq_len]
            attention_mask: [batch_size, total_seq_len]
            labels: [batch_size, total_seq_len] for training
        """
        
        if image_embeddings is not None:
            # Project image embeddings to LLM space
            visual_embeddings = self.adapter(image_embeddings)  # [B, img_seq, llm_dim]
            
            if input_ids is not None:
                # Get text embeddings
                text_embeddings = self.llm.get_input_embeddings()(input_ids)  # [B, text_seq, llm_dim]
                
                # Concatenate visual + text embeddings
                combined_embeddings = torch.cat([visual_embeddings, text_embeddings], dim=1)
                
                # Update attention mask
                batch_size, img_seq_len = visual_embeddings.shape[:2]
                visual_attention = torch.ones(batch_size, img_seq_len, device=visual_embeddings.device)
                
                if attention_mask is not None:
                    attention_mask = torch.cat([visual_attention, attention_mask], dim=1)
                else:
                    text_seq_len = text_embeddings.shape[1]
                    text_attention = torch.ones(batch_size, text_seq_len, device=visual_embeddings.device)
                    attention_mask = torch.cat([visual_attention, text_attention], dim=1)
                
                # Update labels for training
                if labels is not None:
                    # Pad labels for visual tokens (ignore in loss)
                    visual_labels = torch.full(
                        (batch_size, img_seq_len), 
                        fill_value=-100, 
                        device=labels.device
                    )
                    labels = torch.cat([visual_labels, labels], dim=1)
            
            else:
                # Only visual input
                combined_embeddings = visual_embeddings
                batch_size, seq_len = visual_embeddings.shape[:2]
                attention_mask = torch.ones(batch_size, seq_len, device=visual_embeddings.device)
        
        else:
            # Only text input (fallback)
            if input_ids is None:
                raise ValueError("Either image_embeddings or input_ids must be provided")
            
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        
        # Forward through LLM with combined embeddings
        return self.llm(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def generate(
        self,
        image_embeddings: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        **generation_kwargs
    ):
        """Generate text from multimodal input"""
        
        if image_embeddings is not None:
            # Project image embeddings
            visual_embeddings = self.adapter(image_embeddings)
            
            if input_ids is not None:
                # Combine visual + text
                text_embeddings = self.llm.get_input_embeddings()(input_ids)
                combined_embeddings = torch.cat([visual_embeddings, text_embeddings], dim=1)
                
                # Update attention mask
                batch_size, img_seq_len = visual_embeddings.shape[:2]
                visual_attention = torch.ones(batch_size, img_seq_len, device=visual_embeddings.device)
                
                if attention_mask is not None:
                    attention_mask = torch.cat([visual_attention, attention_mask], dim=1)
                else:
                    text_seq_len = text_embeddings.shape[1]
                    text_attention = torch.ones(batch_size, text_seq_len, device=visual_embeddings.device)
                    attention_mask = torch.cat([visual_attention, text_attention], dim=1)
            else:
                combined_embeddings = visual_embeddings
                batch_size, seq_len = visual_embeddings.shape[:2]
                attention_mask = torch.ones(batch_size, seq_len, device=visual_embeddings.device)
            
            # Generate with combined embeddings
            return self.llm.generate(
                inputs_embeds=combined_embeddings,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **generation_kwargs
            )
        
        else:
            # Text-only generation
            return self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **generation_kwargs
            )
    
    def save_adapter(self, save_path: str):
        """Salva solo l'adapter (non tutto l'LLM)"""
        torch.save({
            'adapter_state_dict': self.adapter.state_dict(),
            'encoder_dim': self.adapter.encoder_dim,
            'llm_dim': self.adapter.llm_dim,
        }, save_path)
        logger.info(f"Adapter saved to {save_path}")
    
    def load_adapter(self, load_path: str):
        """Carica adapter pre-trained"""
        checkpoint = torch.load(load_path, map_location='cpu')
        self.adapter.load_state_dict(checkpoint['adapter_state_dict'])
        logger.info(f"Adapter loaded from {load_path}")

# Factory function
def create_multimodal_model(
    llm_model_name: str,
    encoder_dim: int,
    adapter_config: Optional[Dict[str, Any]] = None
) -> MultimodalLLM:
    """Factory per creare modello multimodale"""
    
    return MultimodalLLM(
        llm_model_name=llm_model_name,
        encoder_dim=encoder_dim,
        adapter_config=adapter_config
    )
