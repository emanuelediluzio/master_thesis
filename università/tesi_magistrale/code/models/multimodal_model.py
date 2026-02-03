#!/usr/bin/env python3
"""
Modello Multimodale: SPE + Llama-3.1-8B
Integrazione completa con Linear Adapter e supporto per generazione.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from transformers import GenerationConfig

class LinearAdapter(nn.Module):
    """
    Adapter lineare per mappare embeddings SPE a dimensioni LLM.
    """
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Layer di proiezione
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Inizializzazione Xavier
        self._init_weights()
    
    def _init_weights(self):
        """Inizializzazione dei pesi."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, spe_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Proietta embeddings SPE a dimensioni LLM.
        
        Args:
            spe_embeddings: [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: [batch_size, seq_len, output_dim]
        """
        return self.projection(spe_embeddings)

class MultimodalModel(nn.Module):
    """
    Modello multimodale che combina SPE encoder e LLM.
    """
    
    def __init__(
        self,
        spe_encoder: nn.Module,
        llm_model: nn.Module,
        spe_dim: int,
        llm_dim: int,
        adapter_dropout: float = 0.1
    ):
        super().__init__()
        
        self.spe_encoder = spe_encoder
        self.llm_model = llm_model
        self.spe_dim = spe_dim
        self.llm_dim = llm_dim
        
        # Linear adapter per mappare SPE → LLM
        self.linear_adapter = LinearAdapter(
            input_dim=spe_dim,
            output_dim=llm_dim,
            dropout=adapter_dropout
        )
        
        # Token speciale per separare SVG da testo
        self.svg_separator_embedding = nn.Parameter(
            torch.randn(1, 1, llm_dim) * 0.02
        )
        
        print(f"✅ MultimodalModel inizializzato:")
        print(f"   - SPE dim: {spe_dim}")
        print(f"   - LLM dim: {llm_dim}")
        print(f"   - Adapter dropout: {adapter_dropout}")
    
    def encode_svg(self, svg_tokens: torch.Tensor) -> torch.Tensor:
        """
        Codifica tokens SVG usando SPE encoder.
        
        Args:
            svg_tokens: [batch_size, seq_len]
            
        Returns:
            torch.Tensor: [batch_size, seq_len, llm_dim]
        """
        # Encode con SPE
        spe_embeddings = self.spe_encoder(svg_tokens)
        
        # Proietta a dimensioni LLM
        llm_embeddings = self.linear_adapter(spe_embeddings)
        
        return llm_embeddings
    
    def forward(
        self,
        svg_tokens: torch.Tensor,
        text_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass del modello multimodale.
        
        Args:
            svg_tokens: [batch_size, svg_seq_len]
            text_input_ids: [batch_size, text_seq_len]
            attention_mask: [batch_size, total_seq_len]
            labels: [batch_size, total_seq_len] per training
            
        Returns:
            Dict con logits, loss, etc.
        """
        batch_size = svg_tokens.size(0)
        
        # 1. Encode SVG
        svg_embeddings = self.encode_svg(svg_tokens)  # [batch, svg_len, llm_dim]
        
        # 2. Get text embeddings
        text_embeddings = self.llm_model.get_input_embeddings()(text_input_ids)
        
        # 3. Add separator token
        separator = self.svg_separator_embedding.expand(batch_size, -1, -1)
        
        # 4. Concatena: SVG + separator + text
        combined_embeddings = torch.cat([
            svg_embeddings,
            separator,
            text_embeddings
        ], dim=1)
        
        # 5. Crea attention mask se non fornita
        if attention_mask is None:
            total_len = combined_embeddings.size(1)
            attention_mask = torch.ones(
                batch_size, total_len,
                device=combined_embeddings.device,
                dtype=torch.long
            )
        
        # 6. Forward attraverso LLM
        outputs = self.llm_model(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate(
        self,
        svg_tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
        max_length: int = 150,
        temperature: float = 0.7,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Genera testo condizionato su SVG.
        
        Args:
            svg_tokens: [batch_size, svg_seq_len]
            prompt_tokens: [batch_size, prompt_len]
            max_length: Lunghezza massima generazione
            temperature: Temperatura per sampling
            do_sample: Se usare sampling
            pad_token_id: ID del token di padding
            
        Returns:
            torch.Tensor: [batch_size, generated_len]
        """
        batch_size = svg_tokens.size(0)
        device = svg_tokens.device
        
        # 1. Encode SVG
        svg_embeddings = self.encode_svg(svg_tokens)
        
        # 2. Get prompt embeddings
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_tokens)
        
        # 3. Add separator
        separator = self.svg_separator_embedding.expand(batch_size, -1, -1)
        
        # 4. Combine: SVG + separator + prompt
        initial_embeddings = torch.cat([
            svg_embeddings,
            separator,
            prompt_embeddings
        ], dim=1)
        
        # 5. Crea input_ids iniziali (dummy, useremo inputs_embeds)
        initial_length = initial_embeddings.size(1)
        input_ids = torch.zeros(
            batch_size, initial_length,
            device=device, dtype=torch.long
        )
        
        # 6. Attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # 7. Genera usando il metodo del LLM
        # Nota: Questo è un approccio semplificato
        # In pratica, dovresti implementare una generazione più sofisticata
        
        with torch.no_grad():
            # Forward iniziale
            outputs = self.llm_model(
                inputs_embeds=initial_embeddings,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Prendi logits dell'ultimo token
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Applica temperatura
            if temperature != 1.0:
                logits = logits / temperature
            
            # Sampling o greedy
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Inizia con prompt + primo token generato
            generated_ids = torch.cat([prompt_tokens, next_token], dim=1)
            
            # Genera tokens rimanenti
            for _ in range(max_length - generated_ids.size(1)):
                # Crea embeddings per sequenza corrente
                current_text_embeddings = self.llm_model.get_input_embeddings()(generated_ids)
                current_embeddings = torch.cat([
                    svg_embeddings,
                    separator,
                    current_text_embeddings
                ], dim=1)
                
                # Forward
                outputs = self.llm_model(
                    inputs_embeds=current_embeddings,
                    return_dict=True
                )
                
                # Next token
                logits = outputs.logits[:, -1, :]
                if temperature != 1.0:
                    logits = logits / temperature
                
                if do_sample:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Aggiungi token
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Stop se EOS
                if pad_token_id is not None and (next_token == pad_token_id).all():
                    break
        
        return generated_ids
    
    def get_trainable_parameters(self):
        """Restituisce parametri trainable (solo adapter)."""
        trainable_params = []
        
        # Linear adapter
        for param in self.linear_adapter.parameters():
            trainable_params.append(param)
        
        # Separator embedding
        trainable_params.append(self.svg_separator_embedding)
        
        return trainable_params
    
    def freeze_base_models(self):
        """Congela i modelli base (SPE e LLM)."""
        # Congela SPE
        for param in self.spe_encoder.parameters():
            param.requires_grad = False
        
        # Congela LLM (eccetto adapter LoRA se presente)
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
        print("🔒 Modelli base congelati. Solo adapter trainable.")
    
    def unfreeze_all(self):
        """Scongela tutti i parametri."""
        for param in self.parameters():
            param.requires_grad = True
        
        print("🔓 Tutti i parametri scongelati.")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Restituisce informazioni sul modello."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "spe_dim": self.spe_dim,
            "llm_dim": self.llm_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (trainable_params / total_params) * 100,
            "adapter_parameters": sum(p.numel() for p in self.linear_adapter.parameters()),
            "has_lora": hasattr(self.llm_model, 'peft_config')
        }