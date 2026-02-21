import torch
import json
import os
from typing import List, Union, Optional

class MySPTokenizer:
    """Local implementation of SPE tokenizer to avoid external dependencies"""
    
    def __init__(self, vocab_file: str = None, vocab_size: int = 1024):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        
        # Special tokens
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # Initialize basic vocabulary
        self._init_vocab()
        
        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
    
    def _init_vocab(self):
        """Initialize basic vocabulary with special tokens"""
        self.vocab = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id
        }
        
        # Add basic SVG tokens
        svg_tokens = [
            'M', 'L', 'C', 'Q', 'A', 'Z', 'H', 'V', 'S', 'T',
            'm', 'l', 'c', 'q', 'a', 'z', 'h', 'v', 's', 't'
        ]
        
        # Add numbers 0-9 and common separators
        for i in range(10):
            svg_tokens.append(str(i))
        
        svg_tokens.extend(['.', ',', ' ', '-', '+'])
        
        # Add these to vocabulary
        for i, token in enumerate(svg_tokens):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        # Fill remaining vocabulary slots with placeholder tokens
        while len(self.vocab) < self.vocab_size:
            placeholder = f'<token_{len(self.vocab)}>'
            self.vocab[placeholder] = len(self.vocab)
        
        # Create inverse vocabulary
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def load_vocab(self, vocab_file: str):
        """Load vocabulary from file"""
        try:
            with open(vocab_file, 'r') as f:
                self.vocab = json.load(f)
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        except Exception as e:
            print(f"Warning: Could not load vocab from {vocab_file}: {e}")
            print("Using default vocabulary")
    
    def save_vocab(self, vocab_file: str):
        """Save vocabulary to file"""
        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f, indent=2)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into list of tokens"""
        if not text:
            return []
        
        tokens = []
        i = 0
        while i < len(text):
            # Try to match longest possible token
            matched = False
            for length in range(min(10, len(text) - i), 0, -1):
                candidate = text[i:i+length]
                if candidate in self.vocab:
                    tokens.append(candidate)
                    i += length
                    matched = True
                    break
            
            if not matched:
                # Character not in vocabulary, use unk token
                tokens.append(self.unk_token)
                i += 1
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to list of token IDs"""
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.unk_token_id)
        
        return token_ids
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                if skip_special_tokens and token in [self.pad_token, self.bos_token, self.eos_token]:
                    continue
                tokens.append(token)
            else:
                if not skip_special_tokens:
                    tokens.append(self.unk_token)
        
        return ''.join(tokens)
    
    def __call__(self, text: Union[str, List[str]], 
                 padding: bool = False, 
                 truncation: bool = False, 
                 max_length: Optional[int] = None,
                 return_tensors: Optional[str] = None) -> dict:
        """Tokenize and encode text(s)"""
        
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Encode all texts
        all_token_ids = []
        for t in texts:
            token_ids = self.encode(t, add_special_tokens=True)
            
            if truncation and max_length and len(token_ids) > max_length:
                token_ids = token_ids[:max_length-1] + [self.eos_token_id]
            
            all_token_ids.append(token_ids)
        
        # Padding
        if padding and len(all_token_ids) > 1:
            max_len = max(len(ids) for ids in all_token_ids)
            if max_length:
                max_len = min(max_len, max_length)
            
            for i, token_ids in enumerate(all_token_ids):
                if len(token_ids) < max_len:
                    all_token_ids[i] = token_ids + [self.pad_token_id] * (max_len - len(token_ids))
        
        # Create attention masks
        attention_masks = []
        for token_ids in all_token_ids:
            mask = [1 if tid != self.pad_token_id else 0 for tid in token_ids]
            attention_masks.append(mask)
        
        result = {
            'input_ids': all_token_ids,
            'attention_mask': attention_masks
        }
        
        # Convert to tensors if requested
        if return_tensors == 'pt':
            try:
                result['input_ids'] = torch.tensor(result['input_ids'])
                result['attention_mask'] = torch.tensor(result['attention_mask'])
            except ValueError as e:
                # Handle irregular sequences by padding to max length
                max_len = max(len(ids) for ids in all_token_ids)
                padded_input_ids = []
                padded_attention_masks = []
                
                for i, token_ids in enumerate(all_token_ids):
                    if len(token_ids) < max_len:
                        padded_ids = token_ids + [self.pad_token_id] * (max_len - len(token_ids))
                        padded_mask = attention_masks[i] + [0] * (max_len - len(attention_masks[i]))
                    else:
                        padded_ids = token_ids
                        padded_mask = attention_masks[i]
                    
                    padded_input_ids.append(padded_ids)
                    padded_attention_masks.append(padded_mask)
                
                result['input_ids'] = torch.tensor(padded_input_ids)
                result['attention_mask'] = torch.tensor(padded_attention_masks)
        
        # If single text input, remove batch dimension
        if isinstance(text, str):
            if return_tensors == 'pt':
                result['input_ids'] = result['input_ids'].squeeze(0)
                result['attention_mask'] = result['attention_mask'].squeeze(0)
            else:
                result['input_ids'] = result['input_ids'][0]
                result['attention_mask'] = result['attention_mask'][0]
        
        return result

# Factory function to create tokenizer
def create_spe_tokenizer(vocab_file: str = None, vocab_size: int = 1024) -> MySPTokenizer:
    """Create SPE tokenizer instance"""
    return MySPTokenizer(vocab_file=vocab_file, vocab_size=vocab_size)