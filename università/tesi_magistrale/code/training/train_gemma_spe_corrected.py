#!/usr/bin/env python3
"""
Script di training corretto per SPE + Gemma-9B
Basato sulla configurazione funzionante di Qwen2-7B + SPE
"""

import os
import sys
import torch
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Setup environment
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['HF_DATASETS_CACHE'] = '/tmp/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'

# Add project paths
sys.path.append('/work/tesi_ediluzio')
sys.path.append('/work/tesi_ediluzio/scripts/training')

# Import the robust training launcher
from robust_training_launcher import main as launch_training

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_corrected_gemma_config():
    """Crea la configurazione corretta per SPE + Gemma-9B"""
    
    config = {
        "model_name": "google/gemma-2-9b-it",
        "use_lora": True,
        "dataset_path": "/work/tesi_ediluzio/data/jsonl/qwen2_svg_train.jsonl",
        "output_dir": "/work/tesi_ediluzio/outputs/gemma_9b_lora_spe_corrected",
        "training_args": {
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 16,
            "num_train_epochs": 5,
            "learning_rate": 2e-5,  # Stesso learning rate di Qwen
            "warmup_steps": 500,
            "logging_steps": 25,
            "save_steps": 500,
            "eval_steps": 500,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "fp16": True,
            "gradient_checkpointing": True,
            "dataloader_pin_memory": False,
            "remove_unused_columns": False
        },
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        "spe_config": {
            "enabled": True,
            "encoder_path": "/work/tesi_ediluzio/SPE/SPE_31/checkpoint-360000",
            "projection_dim": 4096
        },
        "wandb_config": {
            "project": "svg_captioning_multimod",
            "entity": "337543-unimore",
            "run_name": "gemma_9b_lora_spe_corrected_training"
        },
        "report_to": "wandb"
    }
    
    return config

def save_config(config, config_path):
    """Salva la configurazione in un file JSON"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configurazione salvata in: {config_path}")

def main():
    """Main function per avviare il training corretto"""
    
    logger.info("🚀 Avvio training corretto SPE + Gemma-9B")
    
    # Crea la configurazione corretta
    config = create_corrected_gemma_config()
    
    # Salva la configurazione
    config_dir = Path("/work/tesi_ediluzio/configs")
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "gemma_9b_spe_corrected_config.json"
    save_config(config, config_path)
    
    # Crea la directory di output
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("📋 Configurazione corretta:")
    logger.info(f"  - Modello: {config['model_name']}")
    logger.info(f"  - LoRA: r={config['lora_config']['r']}, alpha={config['lora_config']['lora_alpha']}")
    logger.info(f"  - Target modules: {config['lora_config']['target_modules']}")
    logger.info(f"  - SPE encoder: {config['spe_config']['encoder_path']}")
    logger.info(f"  - Output dir: {config['output_dir']}")
    logger.info(f"  - Learning rate: {config['training_args']['learning_rate']}")
    
    # Prepara gli argomenti per il launcher
    class Args:
        def __init__(self, config_path, output_dir):
            self.config = str(config_path)
            self.output_dir = str(output_dir)
    
    args = Args(config_path, config["output_dir"])
    
    # Modifica sys.argv per il launcher
    original_argv = sys.argv.copy()
    sys.argv = [
        'train_gemma_spe_corrected.py',
        '--config', str(config_path),
        '--output_dir', config["output_dir"]
    ]
    
    try:
        logger.info("🎯 Avvio del training con configurazione corretta...")
        launch_training()
        logger.info("✅ Training completato con successo!")
        
    except Exception as e:
        logger.error(f"❌ Errore durante il training: {e}")
        raise
    finally:
        # Ripristina sys.argv
        sys.argv = original_argv

if __name__ == "__main__":
    main()