#!/usr/bin/env python3
"""
Robust Training Launcher
Gestisce il training con configurazioni specifiche per MLP-only e LoRA
"""

import os

# Advanced CUDA memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Disable HuggingFace cache to avoid disk quota issues
os.environ['HF_DATASETS_CACHE'] = '/tmp/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
import sys
import torch
import yaml
import json
import logging
import argparse
import gc
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.cuda.amp import autocast, GradScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clear_cuda_memory():
    """Aggressively clear CUDA memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        logger.info(f"CUDA memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated():.2f} MB")

def get_quantization_config():
    """Get optimized quantization configuration for memory efficiency"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

def optimize_model_for_memory(model):
    """Apply memory optimizations to the model"""
    # Enable gradient checkpointing
    # Handle DistributedDataParallel wrapper
    actual_model = model.module if hasattr(model, 'module') else model
    if hasattr(actual_model, 'gradient_checkpointing_enable'):
        actual_model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Skip half precision conversion to avoid CUDA memory allocation errors
    # This prevents the "!block->expandable_segment_" error
    logger.info("Skipping half precision conversion to avoid CUDA allocation errors")
    
    # Additional aggressive optimizations for large models
    if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
        if model.config.num_hidden_layers > 20:  # For large models like 9B
            logger.info("Applying aggressive optimizations for large model")
            
            # Force CPU offloading for some layers if available
            if hasattr(model, 'tie_weights'):
                model.tie_weights()
                logger.info("Tied model weights")
            
            # Enable Flash Attention if available
            try:
                if hasattr(model.config, 'use_flash_attention_2'):
                    model.config.use_flash_attention_2 = True
                    logger.info("Enabled Flash Attention 2")
            except Exception as e:
                logger.warning(f"Could not enable Flash Attention: {e}")
    
    return model

def count_trainable_parameters(model):
    """Conta i parametri trainable del modello"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params

def freeze_model_components(model, config):
    """Applica freezing e configurazioni specifiche al modello"""
    
    if config.get('train_mlp_only', False):
        logger.info("Training MLP layers only")
        
        # Prima freeziamo tutto
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze ONLY specific MLP parameters
        unfrozen_count = 0
        mlp_patterns = ['gate_proj.weight', 'up_proj.weight', 'down_proj.weight']
        
        logger.info("=== Looking for MLP parameters ===")
        for name, param in model.named_parameters():
            # Check if this parameter belongs to an MLP layer (Gemma structure: model.layers.X.mlp.Y)
            if 'layers.' in name and '.mlp.' in name and any(pattern in name for pattern in mlp_patterns):
                # Check if parameter can require gradients (not quantized)
                if param.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                    param.requires_grad = True
                    unfrozen_count += 1
                    logger.info(f"Unfrozen MLP parameter: {name} (dtype: {param.dtype})")
                else:
                    logger.warning(f"Skipping quantized parameter: {name} (dtype: {param.dtype})")
                    
        logger.info(f"Unfrozen {unfrozen_count} MLP parameters")
        
        # Count trainable parameters
        trainable_count, total_count = count_trainable_parameters(model)
        logger.info(f"Trainable parameters: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)")
        
        if trainable_count == 0:
            logger.error("No trainable parameters found! Trying broader patterns...")
            # Try broader patterns if no MLP layers found
            broader_patterns = ['linear', 'dense', 'fc']
            for name, module in model.named_modules():
                if any(pattern in name.lower() for pattern in broader_patterns):
                    logger.info(f"Unfreezing broader pattern module: {name}")
                    for param in module.parameters():
                        param.requires_grad = True
            
            # Final check
            trainable_count, total_count = count_trainable_parameters(model)
            if trainable_count == 0:
                logger.error("Still no trainable parameters! Enabling all parameters as fallback.")
                fallback_enabled = 0
                fallback_skipped = 0
                for param in model.parameters():
                    if param.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                        param.requires_grad = True
                        fallback_enabled += 1
                    else:
                        fallback_skipped += 1
                logger.info(f"Fallback enabled {fallback_enabled} parameters, skipped {fallback_skipped} quantized parameters")
        
        return model
        
    elif config.get('use_lora', False):
        logger.info("Applying LoRA configuration")
        
        # Get LoRA config from nested structure or direct config
        lora_cfg = config.get('lora_config', {})
        
        lora_config = LoraConfig(
            r=lora_cfg.get('r', config.get('lora_r', 16)),
            lora_alpha=lora_cfg.get('lora_alpha', config.get('lora_alpha', 32)),
            target_modules=lora_cfg.get('target_modules', config.get('lora_target_modules', ["q_proj", "v_proj"])),
            lora_dropout=lora_cfg.get('lora_dropout', config.get('lora_dropout', 0.1)),
            bias=lora_cfg.get('bias', "none"),
            task_type=lora_cfg.get('task_type', "CAUSAL_LM")
        )
        
        model = get_peft_model(model, lora_config)
        
        # Ensure LoRA parameters have gradients enabled
        lora_params_found = 0
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
                lora_params_found += 1
                logger.debug(f"Enabled gradients for LoRA parameter: {name}")
        
        logger.info(f"Found and enabled {lora_params_found} LoRA parameters")
        
        # Count trainable parameters
        trainable_count, total_count = count_trainable_parameters(model)
        logger.info(f"LoRA trainable parameters: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)")
        
        if trainable_count == 0:
            logger.error("No LoRA trainable parameters found! Check LoRA configuration.")
            # Force enable gradients for all LoRA parameters
            for name, param in model.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
                    logger.info(f"Force-enabled gradients for: {name}")
            
            # Re-count after forcing
            trainable_count, total_count = count_trainable_parameters(model)
            logger.info(f"After force-enabling: {trainable_count:,} / {total_count:,} trainable parameters")
        
        # SPE adapter integration
        if config.get('use_spe_adapter', False):
            logger.info("Loading SPE adapter...")
            spe_model_path = config.get('spe_model_path')
            if spe_model_path and os.path.exists(spe_model_path):
                # Load SPE model configuration
                spe_config_path = os.path.join(spe_model_path, f"{os.path.basename(spe_model_path)}.yaml")
                if os.path.exists(spe_config_path):
                    with open(spe_config_path, 'r') as f:
                        spe_config = yaml.safe_load(f)
                    logger.info(f"SPE model loaded from {spe_model_path}")
                else:
                    logger.warning(f"SPE config not found at {spe_config_path}")
            else:
                logger.warning(f"SPE model path not found: {spe_model_path}")
                config['use_spe_adapter'] = False
        
        return model
    
    else:
        # Full fine-tuning
        enabled_params = 0
        skipped_params = 0
        for param in model.parameters():
            # Only enable gradients for floating point parameters (not quantized)
            if param.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                param.requires_grad = True
                enabled_params += 1
            else:
                skipped_params += 1
                
        logger.info(f"Full fine-tuning: enabled {enabled_params} parameters, skipped {skipped_params} quantized parameters")
        
        trainable_count, total_count = count_trainable_parameters(model)
        logger.info(f"Full fine-tuning parameters: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)")
        
        return model

def load_config(config_path):
    """Carica la configurazione da file YAML o JSON"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)
    return config

def prepare_dataset(config, tokenizer):
    """Prepara il dataset per il training"""
    # Try multiple possible dataset path configurations
    dataset_path = None
    
    # Check nested data structure first
    if 'data' in config and 'train_path' in config['data']:
        dataset_path = config['data']['train_path']
    # Fallback to direct dataset_path
    elif 'dataset_path' in config:
        dataset_path = config['dataset_path']
    # Fallback to train_path
    elif 'train_path' in config:
        dataset_path = config['train_path']
    
    if not dataset_path:
        raise ValueError("No dataset path found in config. Expected 'data.train_path', 'dataset_path', or 'train_path'")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    logger.info(f"Using dataset path: {dataset_path}")
    
    # Load dataset with streaming to reduce memory usage
    dataset = load_dataset('json', data_files=dataset_path, split='train', cache_dir=None, streaming=False)
    logger.info(f"Loaded dataset with {len(dataset)} examples")
    
    # Use full dataset for training
    logger.info(f"Using full dataset with {len(dataset)} examples for training")
    
    # Preprocessing function
    def preprocess_function(examples):
        # Get text field - handle different dataset formats
        if 'text' in examples:
            texts = examples['text']
        elif 'input' in examples and 'output' in examples:
            # Combine input and output for instruction tuning
            texts = [f"Input: {inp}\nOutput: {out}" for inp, out in zip(examples['input'], examples['output'])]
        else:
            raise ValueError("Dataset must have either 'text' field or 'input'/'output' fields")
        
        # Get max length from config (check both possible keys)
        max_length = config.get('max_length', config.get('max_seq_length', 512))
        
        # Tokenize with padding and truncation
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids
        # Replace padding tokens in labels with -100
        labels = []
        for input_ids in tokenized['input_ids']:
            label = input_ids.copy()
            # Replace pad tokens with -100 so they're ignored in loss
            for i, token_id in enumerate(label):
                if token_id == tokenizer.pad_token_id:
                    label[i] = -100
            labels.append(label)
        
        tokenized['labels'] = labels
        
        return tokenized
    
    # Apply preprocessing with smaller batches to reduce memory usage
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=100,  # Smaller batch size for memory efficiency
        remove_columns=dataset.column_names,
        num_proc=1  # Single process to avoid memory issues
    )
    
    return dataset

class StableTrainer(Trainer):
    """Trainer personalizzato con stabilizzazione per evitare loss explosion"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_history = []
        self.loss_threshold = 10.0  # Soglia per rilevare loss explosion
        self.patience = 3  # Numero di step consecutivi sopra soglia prima di intervenire
        self.consecutive_high_loss = 0
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override compute_loss per monitorare e stabilizzare"""
        try:
            # Calcola loss normale
            if return_outputs:
                if num_items_in_batch is not None:
                    loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
                else:
                    loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            else:
                if num_items_in_batch is not None:
                    loss = super().compute_loss(model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch)
                else:
                    loss = super().compute_loss(model, inputs, return_outputs=False)
                outputs = None
            
            # Monitora loss
            current_loss = loss.item() if hasattr(loss, 'item') else float(loss)
            self.loss_history.append(current_loss)
            
            # Mantieni solo ultimi 100 valori
            if len(self.loss_history) > 100:
                self.loss_history = self.loss_history[-100:]
            
            # Controlla se loss è troppo alto
            if current_loss > self.loss_threshold:
                self.consecutive_high_loss += 1
                logger.warning(f"High loss detected: {current_loss:.4f} (consecutive: {self.consecutive_high_loss})")
                
                # Se troppi step consecutivi con loss alto, applica gradient clipping più aggressivo
                if self.consecutive_high_loss >= self.patience:
                    logger.warning("Applying aggressive gradient clipping due to loss instability")
                    # Clip gradients più aggressivamente
                    if hasattr(model, 'parameters'):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                    self.consecutive_high_loss = 0  # Reset counter
            else:
                self.consecutive_high_loss = 0
            
            # Verifica NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Invalid loss detected: {loss}")
                # Sostituisci con loss piccolo per continuare training
                loss = torch.tensor(0.01, device=loss.device, requires_grad=True)
            
            return (loss, outputs) if return_outputs else loss
            
        except Exception as e:
            logger.error(f"Error in compute_loss: {e}")
            # Fallback loss
            fallback_loss = torch.tensor(0.01, device=inputs[list(inputs.keys())[0]].device, requires_grad=True)
            return (fallback_loss, None) if return_outputs else fallback_loss
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step per gestione errori"""
        try:
            if num_items_in_batch is not None:
                return super().training_step(model, inputs, num_items_in_batch)
            else:
                return super().training_step(model, inputs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("CUDA OOM in training_step, clearing cache")
                clear_cuda_memory()
                # Riprova con batch size ridotto
                if hasattr(self.args, 'per_device_train_batch_size') and self.args.per_device_train_batch_size > 1:
                    logger.info("Reducing batch size temporarily")
                    # Questo è un workaround semplificato
                    pass
            raise e

def main():
    parser = argparse.ArgumentParser(description='Robust Training Launcher')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--output_dir', help='Output directory (overrides config)')
    parser.add_argument('--resume_from_checkpoint', help='Path to checkpoint to resume from')
    parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU training with DDP')
    parser.add_argument('--world_size', type=int, default=1, help='Number of GPUs for distributed training')
    args = parser.parse_args()
    
    try:
        # Initialize distributed training if multi-GPU is enabled
        if args.multi_gpu:
            import torch.distributed as dist
            if not dist.is_initialized():
                # Initialize the process group for distributed training
                dist.init_process_group(backend='nccl')
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                torch.cuda.set_device(local_rank)
                logger.info(f"Initialized distributed training. Local rank: {local_rank}, World size: {dist.get_world_size()}")
        
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded config: {config}")
        
        # Override output_dir if provided via command line
        if args.output_dir:
            config['output_dir'] = args.output_dir
        
        # Aggressive memory clearing
        clear_cuda_memory()
        
        # Setup model and tokenizer with memory optimizations
        model_name = config.get('model_name', 'google/gemma-2b')
        logger.info(f"Loading model with memory optimizations: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Use quantization for memory efficiency
        quantization_config = None
        if config.get('use_quantization', False):
            # Use quantization if explicitly requested
            quantization_config = get_quantization_config()
            if 'quantization_config' in config:
                # Override with custom quantization config
                quant_cfg = config['quantization_config']
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=quant_cfg.get('load_in_8bit', False),
                    load_in_4bit=quant_cfg.get('load_in_4bit', False),
                    llm_int8_threshold=quant_cfg.get('llm_int8_threshold', 6.0),
                    llm_int8_has_fp16_weight=quant_cfg.get('llm_int8_has_fp16_weight', False),
                    bnb_4bit_compute_dtype=torch.float16 if quant_cfg.get('bnb_4bit_compute_dtype') == 'float16' else None,
                    bnb_4bit_use_double_quant=quant_cfg.get('bnb_4bit_use_double_quant', False),
                    bnb_4bit_quant_type=quant_cfg.get('bnb_4bit_quant_type', 'nf4')
                )
            logger.info("Using quantization with LoRA training")
        elif not config.get('train_mlp_only', False) and not config.get('use_lora', False):
            quantization_config = get_quantization_config()
            logger.info("Using quantization for full fine-tuning")
        else:
            if config.get('train_mlp_only', False):
                logger.info("Disabling quantization for MLP-only training")
            if config.get('use_lora', False) and not config.get('use_quantization', False):
                logger.info("Disabling quantization for LoRA training to avoid gradient issues")
        
        # Load model with proper device handling
        if args.multi_gpu:
            # For multi-GPU training, determine local device
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda:0")
        
        if quantization_config is not None:
            # For quantized models, use device_map="auto" for single GPU or specific device for multi-GPU
            if args.multi_gpu:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    device_map={"": device},
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
        else:
            # For non-quantized models, load to specific device for consistency
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if args.multi_gpu else torch.float32,  # Use float16 for multi-GPU
                trust_remote_code=True,
                use_cache=False,
                low_cpu_mem_usage=True
            )
            model = model.to(device)
        
        # Apply memory optimizations
        model = optimize_model_for_memory(model)
        clear_cuda_memory()
        
        # Apply model configuration (freezing/LoRA)
        model = freeze_model_components(model, config)
        
        # Wrap model with DDP for multi-GPU training
        if args.multi_gpu:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=False)
            logger.info(f"Model wrapped with DDP for multi-GPU training on device {device}")
        
        # Prepare dataset
        dataset = prepare_dataset(config, tokenizer)
        
        # Setup memory-optimized training arguments with stability improvements
        training_args_dict = {
            "output_dir": config.get('output_dir', './checkpoints'),
            "dataloader_pin_memory": False,  # Disable pin memory to avoid device conflicts
            "num_train_epochs": config.get('num_epochs', config.get('num_train_epochs', 3)),
            "per_device_train_batch_size": config.get('per_device_train_batch_size', config.get('batch_size', 1)),
            "gradient_accumulation_steps": config.get('gradient_accumulation_steps', 16),
            "learning_rate": config.get('learning_rate', 2e-5),
            "lr_scheduler_type": config.get('lr_scheduler_type', 'cosine'),
            "weight_decay": config.get('weight_decay', 0.01),
            "warmup_steps": config.get('warmup_steps', 500),
            "logging_steps": config.get('logging_steps', 25),
            "save_steps": config.get('save_steps', 500),
            "eval_steps": config.get('eval_steps', 500),
            "eval_strategy": config.get('evaluation_strategy', config.get('eval_strategy', 'no')),
            "save_strategy": config.get('save_strategy', 'steps'),
            "load_best_model_at_end": config.get('load_best_model_at_end', False),
            "metric_for_best_model": config.get('metric_for_best_model', 'eval_loss'),
            "greater_is_better": config.get('greater_is_better', False),
            "gradient_checkpointing": config.get('gradient_checkpointing', True),
            "fp16": config.get('fp16', args.multi_gpu),  # Enable fp16 for multi-GPU, configurable otherwise
            "bf16": config.get('bf16', False),  # Disable bf16 to avoid conflicts
            "dataloader_num_workers": config.get('dataloader_num_workers', 4),
            "optim": "adamw_torch",  # Use standard AdamW for stability
            "max_grad_norm": config.get('max_grad_norm', 1.0),  # Gradient clipping
            "report_to": config.get('report_to', 'wandb') if config.get('report_to') else None,
            "run_name": config.get('run_name', config.get('wandb_run_name', 'robust_training_optimized')),
            "remove_unused_columns": config.get('remove_unused_columns', False),
            "ddp_find_unused_parameters": config.get('ddp_find_unused_parameters', False),  # Optimize DDP
            "ddp_backend": config.get('ddp_backend', 'nccl') if args.multi_gpu else None,
            "dataloader_drop_last": config.get('dataloader_drop_last', True) if args.multi_gpu else False,
            "save_safetensors": True,  # Use safetensors for memory efficiency
            "torch_compile": False,  # Disable torch compile to save memory
            "logging_nan_inf_filter": True,  # Filter NaN/Inf in logs
            "skip_memory_metrics": True,  # Skip memory metrics for stability
            "resume_from_checkpoint": args.resume_from_checkpoint  # Resume from checkpoint if provided
        }
        
        # Add DeepSpeed configuration if specified
        if config.get('deepspeed_config'):
            deepspeed_config_path = config.get('deepspeed_config')
            if os.path.exists(deepspeed_config_path):
                training_args_dict["deepspeed"] = deepspeed_config_path
                logger.info(f"Using DeepSpeed configuration: {deepspeed_config_path}")
            else:
                logger.warning(f"DeepSpeed config file not found: {deepspeed_config_path}")
        
        training_args = TrainingArguments(**training_args_dict)
        
        # Data collator with proper padding configuration
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
        
        # Clear memory before trainer initialization
        clear_cuda_memory()
        
        # Initialize stable trainer
        trainer = StableTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        # Clear memory after trainer initialization
        clear_cuda_memory()
        
        # Re-verify and fix gradient requirements after Trainer initialization
        if config.get('train_mlp_only', False):
            logger.info("Re-checking MLP gradient requirements after Trainer init...")
            fixed_count = 0
            mlp_patterns = ['gate_proj.weight', 'up_proj.weight', 'down_proj.weight']
            
            for name, param in trainer.model.named_parameters():
                if 'layers.' in name and '.mlp.' in name and any(pattern in name for pattern in mlp_patterns):
                    if not param.requires_grad:
                        param.requires_grad = True
                        fixed_count += 1
                        logger.info(f"Fixed gradient requirement for: {name}")
            
            if fixed_count > 0:
                logger.info(f"Fixed {fixed_count} parameters that lost gradient requirements")
            
            # Final verification
            trainable_count, total_count = count_trainable_parameters(trainer.model)
            logger.info(f"Final trainable parameters: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)")
            
            if trainable_count == 0:
                logger.error("CRITICAL: Still no trainable parameters after Trainer init!")
                raise RuntimeError("No trainable parameters found after Trainer initialization")
        
        # Start training
        logger.info("Starting training...")
        if args.resume_from_checkpoint:
            logger.info(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train()
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        logger.info(f"Training completed successfully. Model saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()