#!/usr/bin/env python3
"""
Modelli per l'integrazione multimodale SPE + LLM.
"""

from .spe_model import SPEModel, create_spe_model_small, create_spe_model_base, create_spe_model_large

# Import multimodal_model solo quando necessario per evitare dipendenze
try:
    from .multimodal_model import MultimodalModel, LinearAdapter
    _multimodal_available = True
except ImportError:
    _multimodal_available = False
    MultimodalModel = None
    LinearAdapter = None

__all__ = [
    'SPEModel',
    'create_spe_model_small',
    'create_spe_model_base', 
    'create_spe_model_large'
]

if _multimodal_available:
    __all__.extend(['MultimodalModel', 'LinearAdapter'])