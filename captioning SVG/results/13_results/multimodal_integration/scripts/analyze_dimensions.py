#!/usr/bin/env python3
"""
📊 ANALISI DIMENSIONALITÀ
Analizza dimensioni encoder, embedding e LLM per pianificare integrazione
"""

import torch
import pickle
import json
import logging
from pathlib import Path
from transformers import AutoConfig
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_llm_dimensions():
    """Analizza dimensioni dei modelli LLM trained"""
    
    logger.info("🤖 ANALISI DIMENSIONI LLM")
    logger.info("=" * 50)
    
    models = {
        'gemma-2-9b-it': 'google/gemma-2-9b-it',
        'llama-3.1-8b-instruct': 'meta-llama/Llama-3.1-8B-Instruct'
    }
    
    llm_info = {}
    
    for model_name, model_path in models.items():
        try:
            config = AutoConfig.from_pretrained(model_path)
            
            info = {
                'hidden_size': config.hidden_size,
                'num_attention_heads': config.num_attention_heads,
                'num_hidden_layers': config.num_hidden_layers,
                'vocab_size': config.vocab_size,
                'max_position_embeddings': getattr(config, 'max_position_embeddings', 'N/A'),
                'intermediate_size': getattr(config, 'intermediate_size', 'N/A')
            }
            
            llm_info[model_name] = info
            
            logger.info(f"📋 {model_name.upper()}:")
            logger.info(f"   Hidden Size: {info['hidden_size']}")
            logger.info(f"   Attention Heads: {info['num_attention_heads']}")
            logger.info(f"   Layers: {info['num_hidden_layers']}")
            logger.info(f"   Vocab Size: {info['vocab_size']}")
            logger.info(f"   Max Position: {info['max_position_embeddings']}")
            logger.info(f"   Intermediate: {info['intermediate_size']}")
            logger.info("")
            
        except Exception as e:
            logger.error(f"❌ Errore analisi {model_name}: {e}")
    
    return llm_info

def analyze_encoder_weights(encoder_weights_dir):
    """Analizza pesi encoder da Leonardo"""
    
    logger.info("🎨 ANALISI ENCODER WEIGHTS")
    logger.info("=" * 50)
    
    encoder_weights_path = Path(encoder_weights_dir)
    encoder_info = {}
    
    if not encoder_weights_path.exists():
        logger.warning("⚠️ Directory encoder_weights non trovata")
        logger.info("📥 Attendere materiali da Leonardo:")
        logger.info("   - image_encoder.pth")
        logger.info("   - projection_layer.pth") 
        logger.info("   - config.json")
        return encoder_info
    
    # Analizza config.json
    config_file = encoder_weights_path / "config.json"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            encoder_info['config'] = config
            logger.info("✅ Config encoder trovato:")
            for key, value in config.items():
                logger.info(f"   {key}: {value}")
            logger.info("")
            
        except Exception as e:
            logger.error(f"❌ Errore lettura config: {e}")
    
    # Analizza pesi encoder
    encoder_file = encoder_weights_path / "image_encoder.pth"
    if encoder_file.exists():
        try:
            encoder_weights = torch.load(encoder_file, map_location='cpu')
            
            logger.info("✅ Pesi encoder trovati:")
            logger.info(f"   Tipo: {type(encoder_weights)}")
            
            if isinstance(encoder_weights, dict):
                logger.info("   Chiavi:")
                for key in encoder_weights.keys():
                    if isinstance(encoder_weights[key], torch.Tensor):
                        shape = encoder_weights[key].shape
                        logger.info(f"     {key}: {shape}")
                    else:
                        logger.info(f"     {key}: {type(encoder_weights[key])}")
            
            encoder_info['encoder_weights'] = {
                'type': str(type(encoder_weights)),
                'keys': list(encoder_weights.keys()) if isinstance(encoder_weights, dict) else None
            }
            
        except Exception as e:
            logger.error(f"❌ Errore caricamento encoder: {e}")
    
    # Analizza projection layer
    projection_file = encoder_weights_path / "projection_layer.pth"
    if projection_file.exists():
        try:
            projection_weights = torch.load(projection_file, map_location='cpu')
            
            logger.info("✅ Projection layer trovato:")
            logger.info(f"   Tipo: {type(projection_weights)}")
            
            if isinstance(projection_weights, dict):
                for key in projection_weights.keys():
                    if isinstance(projection_weights[key], torch.Tensor):
                        shape = projection_weights[key].shape
                        logger.info(f"     {key}: {shape}")
            
            encoder_info['projection_weights'] = {
                'type': str(type(projection_weights)),
                'keys': list(projection_weights.keys()) if isinstance(projection_weights, dict) else None
            }
            
        except Exception as e:
            logger.error(f"❌ Errore caricamento projection: {e}")
    
    return encoder_info

def analyze_embeddings(embeddings_dir):
    """Analizza embedding pre-calcolati da Leonardo"""
    
    logger.info("🎯 ANALISI EMBEDDINGS")
    logger.info("=" * 50)
    
    embeddings_path = Path(embeddings_dir)
    embeddings_info = {}
    
    if not embeddings_path.exists():
        logger.warning("⚠️ Directory embeddings non trovata")
        logger.info("📥 Attendere materiali da Leonardo:")
        logger.info("   - train_embeddings.pkl")
        logger.info("   - test_embeddings.pkl")
        logger.info("   - embedding_metadata.json")
        return embeddings_info
    
    # Analizza metadata
    metadata_file = embeddings_path / "embedding_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            embeddings_info['metadata'] = metadata
            logger.info("✅ Metadata embedding trovati:")
            for key, value in metadata.items():
                logger.info(f"   {key}: {value}")
            logger.info("")
            
        except Exception as e:
            logger.error(f"❌ Errore lettura metadata: {e}")
    
    # Analizza embedding files
    embedding_files = ['train_embeddings.pkl', 'test_embeddings.pkl']
    
    for filename in embedding_files:
        filepath = embeddings_path / filename
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    embeddings = pickle.load(f)
                
                logger.info(f"✅ {filename} trovato:")
                logger.info(f"   Tipo: {type(embeddings)}")
                
                if isinstance(embeddings, (list, tuple)):
                    logger.info(f"   Lunghezza: {len(embeddings)}")
                    if len(embeddings) > 0:
                        first_item = embeddings[0]
                        if isinstance(first_item, torch.Tensor):
                            logger.info(f"   Shape primo elemento: {first_item.shape}")
                        elif isinstance(first_item, np.ndarray):
                            logger.info(f"   Shape primo elemento: {first_item.shape}")
                
                elif isinstance(embeddings, dict):
                    logger.info("   Chiavi:")
                    for key in list(embeddings.keys())[:5]:  # Prime 5 chiavi
                        value = embeddings[key]
                        if isinstance(value, (torch.Tensor, np.ndarray)):
                            logger.info(f"     {key}: {value.shape}")
                        else:
                            logger.info(f"     {key}: {type(value)}")
                
                elif isinstance(embeddings, (torch.Tensor, np.ndarray)):
                    logger.info(f"   Shape: {embeddings.shape}")
                    logger.info(f"   Dtype: {embeddings.dtype}")
                
                embeddings_info[filename] = {
                    'type': str(type(embeddings)),
                    'length': len(embeddings) if hasattr(embeddings, '__len__') else None,
                    'shape': embeddings.shape if hasattr(embeddings, 'shape') else None
                }
                
                logger.info("")
                
            except Exception as e:
                logger.error(f"❌ Errore caricamento {filename}: {e}")
    
    return embeddings_info

def create_integration_plan(llm_info, encoder_info, embeddings_info):
    """Crea piano di integrazione basato su analisi dimensionalità"""
    
    logger.info("🎯 PIANO INTEGRAZIONE")
    logger.info("=" * 50)
    
    plan = {
        'llm_dimensions': llm_info,
        'encoder_info': encoder_info,
        'embeddings_info': embeddings_info,
        'integration_strategy': {},
        'adapter_requirements': {}
    }
    
    # Strategia per ogni LLM
    for llm_name, llm_config in llm_info.items():
        llm_dim = llm_config['hidden_size']
        
        logger.info(f"📋 STRATEGIA {llm_name.upper()}:")
        logger.info(f"   LLM Hidden Dim: {llm_dim}")
        
        # TODO: Determinare encoder_dim quando disponibile
        encoder_dim = "TBD"  # Da Leonardo
        
        strategy = {
            'llm_dim': llm_dim,
            'encoder_dim': encoder_dim,
            'adapter_type': 'linear_projection',  # Default
            'training_strategy': 'freeze_llm_train_adapter',
            'integration_method': 'prepend_visual_tokens'
        }
        
        plan['integration_strategy'][llm_name] = strategy
        
        logger.info(f"   Encoder Dim: {encoder_dim}")
        logger.info(f"   Adapter: Linear projection {encoder_dim} → {llm_dim}")
        logger.info(f"   Training: Freeze LLM, train adapter")
        logger.info(f"   Integration: Prepend visual tokens")
        logger.info("")
    
    return plan

def save_analysis_report(analysis_data, output_file):
    """Salva report analisi"""
    
    with open(output_file, 'w') as f:
        json.dump(analysis_data, f, indent=2, default=str)
    
    logger.info(f"📄 Report salvato: {output_file}")

def main():
    """Main analysis function"""
    
    logger.info("📊 ANALISI DIMENSIONALITÀ MULTIMODALE")
    logger.info("=" * 60)
    
    # Paths
    base_dir = Path("multimodal_integration")
    encoder_weights_dir = base_dir / "encoder_weights"
    embeddings_dir = base_dir / "embeddings"
    
    # Analisi
    llm_info = analyze_llm_dimensions()
    encoder_info = analyze_encoder_weights(encoder_weights_dir)
    embeddings_info = analyze_embeddings(embeddings_dir)
    
    # Piano integrazione
    integration_plan = create_integration_plan(llm_info, encoder_info, embeddings_info)
    
    # Salva report
    output_file = base_dir / "experiments" / "dimension_analysis" / "analysis_report.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_analysis_report(integration_plan, output_file)
    
    logger.info("=" * 60)
    logger.info("🎉 ANALISI COMPLETATA!")
    logger.info("=" * 60)
    
    return integration_plan

if __name__ == "__main__":
    main()
