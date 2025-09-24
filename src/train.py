#!/usr/bin/env python3
"""
MECO Model Training Script with HuggingFace Dataset Support
"""

from datasets import load_dataset, load_from_disk

def download_hf_dataset_if_needed(config):
    """Download HuggingFace dataset to local directory if not exists"""
    if 'hf_dataset_name' not in config:
        return  # No HF dataset specified, use existing local path
    
    hf_dataset_name = config['hf_dataset_name']
    local_path = config['meco_dataset_path']
    
    log = logging.getLogger(__name__)
    
    # Check if dataset already exists locally
    if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, 'dataset_dict.json')):
        log.info(f"📁 Using existing local dataset: {local_path}")
        return
    
    log.info(f"📥 Downloading dataset: {hf_dataset_name}")
    log.info(f"📁 Local path: {local_path}")
    
    try:
        # Create local directory
        os.makedirs(local_path, exist_ok=True)
        
        # Download dataset
        dataset = load_dataset(hf_dataset_name, token=True)
        
        # Save to local directory
        dataset.save_to_disk(local_path)
        
        log.info(f"✅ Dataset downloaded successfully")
        log.info(f"   Splits: {list(dataset.keys())}")
        for split, data in dataset.items():
            log.info(f"   {split}: {len(data)} samples")
        
    except Exception as e:
        log.error(f"❌ Failed to download dataset: {e}")
        raise RuntimeError(f"Failed to download dataset: {hf_dataset_name}")

#!/usr/bin/env python3
"""
MECO Training Script for Quick LLaMA - Single File Implementation

Usage:
    python train_meco.py --meco-dataset-path /path/to/dataset --experiment-name test --max-steps 100
"""

import os
import sys
import argparse
import logging
import json
import gc
import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import yaml

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Add Quick LLaMA to path
quick_llama_root = Path(__file__).parent.parent / "quick_llama"
quick_llama_src = quick_llama_root / "src"
sys.path.insert(0, str(quick_llama_src))

# Import and register all Quick LLaMA modules
import quick_llama.cache as cache_module
import quick_llama.packer_batcher as packer_batcher_module
sys.modules['cache'] = cache_module
sys.modules['packer_batcher'] = packer_batcher_module

# Override cache directory
cache_dir = os.path.join(os.path.dirname(__file__), "..", "training_data", "tokenized_cache")
os.makedirs(cache_dir, exist_ok=True)
cache_module.TMP_DIR = cache_dir
log.info(f"📁 Tokenized datasets will be cached in: {cache_dir}")

# Quick LLaMA imports
try:
    import torch
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
    from accelerate.utils.dataclasses import DDPCommunicationHookType
    from quick_llama.models import model_utils
    from datasets import Dataset, DatasetDict
    
    log.info("✅ Quick LLaMA imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Quick LLaMA: {e}")
    sys.exit(1)

# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

def load_yaml_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "meco_config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_meco_config(
    meco_dataset_path: str,
    experiment_name: str,
    config_overrides: Dict[str, Any] = None,
    config_file: str = None
) -> Dict[str, Any]:
    """Get MECO configuration with CLI overrides"""
    
    # Load base config from YAML
    config = load_yaml_config(config_file)
    
    # Ensure numeric types are correct
    numeric_fields = {
        'sequence_length': int,
        'batch_size_per_device': int,
        'minimum_sequence_length': int,
        'learning_rate': float,
        'weight_decay': float,
        'adam_epsilon': float,
        'num_warmup_steps_ratio': float,
        'clip_grad_norm': float,
        'max_steps': int,
        'gradient_accumulation_steps': int,
        'save_steps': int,
        'logging_steps': int,
        'save_limit': int,
        'steps_between_evals': int,
        'max_eval_batches': int,
        'memory_cleanup_interval': int,
        'gpu_monitor_interval': int,
        'performance_monitor_interval': int,
        'communication_timeout_seconds': int,
        'seed': int,
        'meco_split_ratio': float,
    }
    
    # Convert types
    for field, field_type in numeric_fields.items():
        if field in config:
            config[field] = field_type(config[field])
    
    # Add required paths and names
    config.update({
        "meco_dataset_path": meco_dataset_path,
        "experiment_name": experiment_name,
        "dataset_name": f"meco_{experiment_name}",
        "run_name": f"meco_{experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "model_name": f"meta-llama/Llama-3.2-{config['model_size'].upper()}",
        "output_dir": config["output_dir_template"].format(experiment_name=experiment_name),
        "checkpoint_dir": config["checkpoint_dir_template"].format(experiment_name=experiment_name),
        "project_name": experiment_name,
    })
    
    # Apply CLI overrides with type conversion
    if config_overrides:
        for key, value in config_overrides.items():
            if key in numeric_fields:
                config[key] = numeric_fields[key](value)
            else:
                config[key] = value
    
    # Derived values - don't set max_steps here, let it be calculated from epochs
    config["batch_size"] = config["batch_size_per_device"]
    config["lr"] = config["learning_rate"]  # Add lr alias for validation
    
    # Validate configuration
    _validate_meco_config(config)
    
    # Debug LR scaling (log the math)
    if config.get('_debug_config_lr'):
        grad_accum = config.get('gradient_accumulation_steps', 1)
        effective_lr = config['_debug_config_lr'] / grad_accum
        log.info(f"[LR DEBUG] config_lr={config['_debug_config_lr']:.6g}, "
                f"effective_peak_lr={effective_lr:.6g}, "
                f"tokens_per_step={config.get('_debug_tokens_per_step', 'unknown')}, "
                f"grad_accum={grad_accum}")
    
    return config

def _validate_meco_config(config: Dict[str, Any]):
    """Validate MECO configuration"""
    required_keys = ["experiment_name", "sequence_length", "batch_size_per_device"]
    
    # Only require meco_dataset_path if not using LMSYS chat
    if not (config.get('training_mode') == 'instruction_tuning' and config.get('use_lmsys_chat', False)):
        required_keys.append("meco_dataset_path")
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    # Only validate dataset path if we're using it AND no HF dataset is specified
    if ("meco_dataset_path" in config and 
        "hf_dataset_name" not in config and 
        not os.path.exists(config["meco_dataset_path"])):
        raise FileNotFoundError(f"MECO dataset path not found: {config['meco_dataset_path']}")

# =============================================================================
# DATA LOADING
# =============================================================================

def load_lmsys_chat_dataset(config: Dict) -> DatasetDict:
    """Load LMSYS Chat-1M dataset for instruction tuning with caching"""
    from datasets import load_dataset, Dataset
    
    # Setup cache directory
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "training_data", "instruction_tuning")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "lmsys_chat_100k")
    
    # Check if cached version exists
    if os.path.exists(cache_path):
        log.info(f"Loading cached LMSYS dataset from: {cache_path}")
        dataset = Dataset.load_from_disk(cache_path)
    else:
        log.info("Downloading and processing LMSYS Chat-1M dataset...")
        
        # Load full dataset
        full_dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")
        log.info(f"Loaded {len(full_dataset):,} total samples")
        
        # Filter for English, non-redacted samples
        filtered_dataset = full_dataset.filter(
            lambda x: x['language'] == 'English' and x['redacted'] == False
        )
        log.info(f"Filtered to {len(filtered_dataset):,} English, non-redacted samples")
        
        # Sample 100k rows
        if len(filtered_dataset) > 100000:
            dataset = filtered_dataset.shuffle(seed=config["seed"]).select(range(100000))
        else:
            dataset = filtered_dataset
        log.info(f"Selected {len(dataset):,} samples for training")
        
        # Format conversations using LLaMA chat format
        def format_conversation(example):
            conversation = example['conversation']
            user_content = None
            assistant_content = None
            
            # Find first user-assistant pair
            for turn in conversation:
                if turn['role'] == 'user' and user_content is None:
                    user_content = turn['content']
                elif turn['role'] == 'assistant' and user_content is not None and assistant_content is None:
                    assistant_content = turn['content']
                    break
            
            if user_content and assistant_content:
                # Use proper LLaMA chat format with system message
                text = f"<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant_content}<|eot_id|>"
                return {"text": text}
            return {"text": ""}
        
        # Apply formatting and filter empty
        dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)
        dataset = dataset.filter(lambda x: len(x['text']) > 0)
        
        # Cache the processed dataset
        dataset.save_to_disk(cache_path)
        log.info(f"Cached processed dataset to: {cache_path}")
    
    log.info(f"Final dataset size: {len(dataset):,} samples")
    
    # Split into train/validation
    train_val_split = dataset.train_test_split(
        test_size=1.0 - config["meco_split_ratio"],
        seed=config["seed"]
    )
    
    return DatasetDict({
        "train": train_val_split["train"],
        "validation": train_val_split["test"]
    })

def upload_meco_dataset_to_hf(dataset_path: str, hf_dataset_name: str = None):
    """Upload MECO dataset splits to HuggingFace"""
    from datasets import Dataset, DatasetDict
    
    dataset_path = Path(dataset_path)
    if hf_dataset_name is None:
        hf_dataset_name = dataset_path.name
    
    log.info(f"Uploading dataset from {dataset_path} to HuggingFace as {hf_dataset_name}")
    
    # Load all splits
    splits = {}
    for split_dir in dataset_path.iterdir():
        if split_dir.is_dir():
            split_name = split_dir.name
            splits[split_name] = Dataset.load_from_disk(str(split_dir))
            log.info(f"Loaded {split_name} split: {len(splits[split_name]):,} samples")
    
    # Create DatasetDict and push to hub
    dataset_dict = DatasetDict(splits)
    try:
        dataset_dict.push_to_hub(hf_dataset_name)
        log.info(f"✅ Uploaded dataset to HuggingFace: {hf_dataset_name}")
    except Exception as e:
        log.error(f"❌ Failed to upload to HuggingFace: {e}")
        log.error("Make sure you're logged in with 'huggingface-cli login'")
        raise

def load_meco_dataset_for_training(config: Dict) -> DatasetDict:
    """Load MECO dataset for training from local or HuggingFace"""
    # Check if we should use LMSYS chat dataset for instruction tuning
    if config.get('training_mode') == 'instruction_tuning' and config.get('use_lmsys_chat', False):
        return load_lmsys_chat_dataset(config)
    
    # If HF dataset is specified, download it first
    if 'hf_dataset_name' in config:
        download_hf_dataset_if_needed(config)
    
    dataset_path = Path(config["meco_dataset_path"])
    
    # Try loading from local first
    if dataset_path.exists():
        log.info(f"Loading MECO dataset from local path: {dataset_path}")
        
        # New structure: dataset_path points to with_metadata or without_metadata directory
        train_path = dataset_path / "train"
        validation_path = dataset_path / "validation"  # Updated from "reserved_test"
        
        if train_path.exists():
            train_dataset = Dataset.load_from_disk(str(train_path))
            log.info(f"Loaded MECO dataset split 'train' with {len(train_dataset):,} samples")
        else:
            raise FileNotFoundError(f"Train split not found at: {train_path}")
        
        # Load validation split if it exists
        if validation_path.exists():
            validation_dataset = Dataset.load_from_disk(str(validation_path))
            log.info(f"Loaded MECO dataset split 'validation' with {len(validation_dataset):,} samples")
            
            return DatasetDict({
                "train": train_dataset,
                "validation": validation_dataset
            })
        else:
            log.info("No validation split found, creating train/validation split from train data")
            # Split train into train/validation
            train_val_split = train_dataset.train_test_split(
                test_size=1.0 - config["meco_split_ratio"],
                seed=config["seed"]
            )
            
            return DatasetDict({
                "train": train_val_split["train"],
                "validation": train_val_split["test"]
            })
    else:
        # Try loading from HuggingFace
        hf_dataset_name = dataset_path.name
        log.info(f"Local path not found, trying HuggingFace dataset: {hf_dataset_name}")
        
        try:
            from datasets import load_dataset
            full_dataset = load_dataset(hf_dataset_name)
            train_dataset = full_dataset['train']
            log.info(f"Loaded MECO dataset from HuggingFace with {len(train_dataset):,} samples")
            
            # Split train into train/validation
            train_val_split = train_dataset.train_test_split(
                test_size=1.0 - config["meco_split_ratio"],
                seed=config["seed"]
            )
            
            return DatasetDict({
                "train": train_val_split["train"],
                "validation": train_val_split["test"]
            })
        except Exception as e:
            raise FileNotFoundError(f"Dataset not found locally or on HuggingFace: {dataset_path} / {hf_dataset_name}. Error: {e}")

# =============================================================================
# DATASET AND TRAINING COMPONENTS
# =============================================================================

class MECODataset(torch.utils.data.Dataset):
    """Streaming MECO dataset that yields batches on demand"""
    
    def __init__(self, hf_dataset, config):
        self.dataset = hf_dataset
        self.config = config
        self.sequence_length = config['sequence_length']
        
        # Move imports to initialization to avoid hot path overhead
        import quick_llama.data_utils as data_utils
        self.tokenizer = data_utils
        
        # Initialize tokenizer with correct model name
        data_utils.initialize_tokenizer(config.get('model_name', 'meta-llama/Llama-3.2-1B'))
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            # Get text from dataset
            text = self.dataset[idx]['text']
            
            # Tokenize using Quick LLaMA's tokenizer
            tokens = self.tokenizer.tokenize_text(text)
            
            # Handle sequence length - pad or truncate
            original_length = len(tokens)
            if len(tokens) > self.sequence_length:
                tokens = tokens[:self.sequence_length]
                original_length = self.sequence_length
            elif len(tokens) < self.sequence_length:
                # Pad with zeros
                pad_length = self.sequence_length - len(tokens)
                tokens = np.concatenate([tokens, np.zeros(pad_length, dtype=tokens.dtype)])
            
            # Create labels based on training mode
            if self.config.get('training_mode', 'pretraining') in ['pretraining', 'continued_pretraining']:
                # MECO pretraining with BOS/EOS: <|begin_of_text|> -> 10, 10 -> 20, ..., 40 -> <|end_of_text|>
                bos_token = 128000  # <|begin_of_text|>
                eos_token = 128001  # <|end_of_text|>
                
                # Truncate if needed to leave room for BOS/EOS
                if len(tokens) > self.sequence_length - 2:
                    tokens = tokens[:self.sequence_length - 2]
                    original_length = self.sequence_length - 2
                
                # Add BOS/EOS tokens
                tokens_with_special = np.concatenate([[bos_token], tokens, [eos_token]])
                
                # Pad if needed
                if len(tokens_with_special) < self.sequence_length:
                    pad_length = self.sequence_length - len(tokens_with_special)
                    tokens_with_special = np.concatenate([tokens_with_special, np.zeros(pad_length, dtype=tokens.dtype)])
                
                # Labels: predict next token for all positions except last
                labels = np.concatenate([tokens_with_special[1:], [-100]])
                
                segment_ids = np.zeros(self.sequence_length, dtype=np.int32)
                attention_mask = np.ones(original_length + 2, dtype=np.int32)  # +2 for BOS/EOS
                
                # Mask padding in attention
                if original_length + 2 < self.sequence_length:
                    attention_mask = np.concatenate([attention_mask, np.zeros(self.sequence_length - original_length - 2, dtype=np.int32)])
                
                # Use the tokens with special tokens
                tokens = tokens_with_special
            else:
                # Instruction tuning: mask input tokens, predict output tokens
                bos_token = 128000  # <|begin_of_text|>
                eot_token = 128009  # <|eot_id|>
                
                # Get original text to find input/output boundary
                text = self.dataset[idx]['text']
                assistant_start = text.find('<|start_header_id|>assistant<|end_header_id|>')
                
                if assistant_start != -1:
                    # Tokenize input portion to find boundary
                    input_text = text[:assistant_start + len('<|start_header_id|>assistant<|end_header_id|>\n\n')]
                    input_tokens = self.tokenizer.tokenize_text(input_text)
                    input_length = len(input_tokens)
                else:
                    # Only LMSYS format supported for instruction tuning
                    raise ValueError("Instruction tuning only supports LMSYS chat format with proper role headers")
                
                # Add BOS and truncate if needed
                if len(tokens) > self.sequence_length - 1:
                    tokens = tokens[:self.sequence_length - 1]
                    original_length = self.sequence_length - 1
                    input_length = min(input_length, original_length)
                
                tokens_with_bos = np.concatenate([[bos_token], tokens])
                
                # Pad if needed
                if len(tokens_with_bos) < self.sequence_length:
                    pad_length = self.sequence_length - len(tokens_with_bos)
                    tokens_with_bos = np.concatenate([tokens_with_bos, np.zeros(pad_length, dtype=tokens.dtype)])
                
                # Labels: predict next token, but mask input portion
                labels = np.concatenate([tokens_with_bos[1:], [-100]])
                labels[:input_length] = -100  # Mask BOS + input tokens
                
                # Segment IDs: 0 for input, 1 for output  
                segment_ids = np.zeros(self.sequence_length, dtype=np.int32)
                segment_ids[input_length:original_length + 1] = 1  # Output segment
                
                # Attention mask: attend to real tokens, ignore padding
                attention_mask = np.ones(original_length + 1, dtype=np.int32)
                if original_length + 1 < self.sequence_length:
                    attention_mask = np.concatenate([attention_mask, np.zeros(self.sequence_length - original_length - 1, dtype=np.int32)])
                
                tokens = tokens_with_bos
            
            return {
                'input_ids': torch.tensor(tokens, dtype=torch.long),
                'label_ids': torch.tensor(labels, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'segment_ids': torch.tensor(segment_ids, dtype=torch.long)
            }
            
        except Exception as e:
            log.error(f"Error processing sample {idx}: {e}")
            raise

def create_streaming_dataloader(dataset, config, accelerator):
    """Create streaming dataloader that doesn't load all data into memory"""
    meco_dataset = MECODataset(dataset, config)
    
    # Create dataloader with proper distributed sampling
    sampler = None
    if accelerator.num_processes > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(meco_dataset, shuffle=True)
    
    dataloader = DataLoader(
        meco_dataset,
        batch_size=config["batch_size_per_device"],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=int(config.get("dataloader_num_workers", 0)),
        pin_memory=bool(config.get("dataloader_pin_memory", True)),
        drop_last=True,
        persistent_workers=(int(config.get("dataloader_num_workers", 0)) > 0)
    )
    
    return dataloader

def validate_environment(config):
    """Validate training environment and configuration"""
    # Check CUDA availability
    if config.get('mixed_precision') in ['fp16', 'bf16'] and not torch.cuda.is_available():
        raise RuntimeError("Mixed precision training requires CUDA")
    
    # Validate model config compatibility
    required_keys = ['sequence_length', 'batch_size_per_device', 'learning_rate']
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    
    # Check memory requirements
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 16 and config['sequence_length'] > 2048:
            log.warning("Long sequences may cause OOM on GPUs with <16GB memory")

class TrainingMetrics:
    """Centralized training metrics collection and reporting"""
    
    def __init__(self, config):
        self.config = config
        self.reset()
        
    def reset(self):
        self.train_loss_sum = 0
        self.train_steps = 0
        self.best_val_loss = float('inf')
        
    def update_train_loss(self, loss):
        self.train_loss_sum += loss
        self.train_steps += 1
        
    def get_avg_train_loss(self):
        return self.train_loss_sum / self.train_steps if self.train_steps > 0 else 0
        
    def should_log(self, step):
        return step % self.config["logging_steps"] == 0
        
    def should_evaluate(self, step, max_steps):
        eval_steps = self.config["steps_between_evals"]
        return step % eval_steps == 0
        
    def should_save_checkpoint(self, step):
        return step % self.config["save_steps"] == 0

class ModelTrainer:
    """Handles model training operations"""
    
    def __init__(self, model, optimizer, lr_scheduler, accelerator, config):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.config = config
        
    def train_step(self, batch):
        """Execute single training step"""
        with self.accelerator.accumulate(self.model):
            # Use Quick LLaMA model interface
            loss = self.model(
                input_ids=batch['input_ids'], 
                label_ids=batch['label_ids'],
                attention_mask=batch['attention_mask'],
                segment_ids=batch['segment_ids']
            )
            
            if torch.isnan(loss):
                raise ValueError("NaN loss detected")
            
            self.accelerator.backward(loss)
            
            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config["clip_grad_norm"]
                )
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                if grad_norm > 10.0:
                    log.warning(f"High gradient norm: {grad_norm:.2f}")
                    
        return loss
    
    def eval_step(self, batch):
        """Execute single evaluation step"""
        with torch.no_grad():
            # Use Quick LLaMA model interface
            loss = self.model(
                input_ids=batch['input_ids'], 
                label_ids=batch['label_ids'],
                attention_mask=batch['attention_mask'],
                segment_ids=batch['segment_ids']
            )
            if torch.isnan(loss):
                raise ValueError("NaN validation loss")
            return loss

class CheckpointManager:
    """Handles model checkpointing operations"""
    
    def __init__(self, accelerator, config):
        self.accelerator = accelerator
        self.config = config
        
    def save_checkpoint(self, model, optimizer, lr_scheduler, step):
        """Save model checkpoint with proper synchronization"""
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            try:
                model_utils.save_checkpoint(
                    self.accelerator, model, optimizer, lr_scheduler,
                    step, self.config["output_dir"], self.config,
                    self.config["save_limit"]
                )
                log.info(f"Saved checkpoint at step {step}")
            except Exception as e:
                log.error(f"Failed to save checkpoint: {e}")
        
        # Remove the problematic final wait_for_everyone()
        # self.accelerator.wait_for_everyone()

class SystemMonitor:
    """Handles system monitoring and logging"""
    
    def __init__(self, config):
        self.config = config
        self.gpu_monitor_interval = config['gpu_monitor_interval']
        self.performance_monitor_interval = config['performance_monitor_interval']
        
    def should_monitor_gpu(self, step):
        return step % self.gpu_monitor_interval == 0
        
    def should_monitor_performance(self, step):
        return step % self.performance_monitor_interval == 0
        
    def log_training_metrics(self, step, avg_loss, lr, max_steps, include_gpu=False, accelerator=None):
        """Log training metrics with optional GPU monitoring"""
        # Calculate perplexity from training loss
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        metrics_dict = {
            "train/loss": avg_loss,
            "train/perplexity": perplexity,
            "train/learning_rate": lr,
            "train/effective_learning_rate": lr * accelerator.gradient_accumulation_steps * accelerator.num_processes * self.config["batch_size_per_device"],
            "train/step": step,
            "train/progress": step / max_steps * 100  # Progress percentage
        }
        
        if include_gpu and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            gpu_cached = torch.cuda.memory_reserved() / 1e9
            
            metrics_dict.update({
                "system/gpu_memory_gb": gpu_memory,
                "system/gpu_cached_gb": gpu_cached
            })
            log.info(f"Step {step}/{max_steps}: train_loss={avg_loss:.4f}, perplexity={perplexity:.2f}, lr={lr:.2e}, "
                    f"GPU_mem={gpu_memory:.1f}GB, GPU_cached={gpu_cached:.1f}GB")
        else:
            log.info(f"Step {step}/{max_steps}: train_loss={avg_loss:.4f}, perplexity={perplexity:.2f}, lr={lr:.2e}")
        
        # Log to WandB via accelerator
        if accelerator and accelerator.is_main_process:
            accelerator.log(metrics_dict, step=step)
            
    def log_dataset_stats(self, dataset, step):
        """Log dataset statistics periodically"""
        if hasattr(dataset, 'get_statistics') and step % (self.config["logging_steps"] * 10) == 0:
            stats = dataset.get_statistics()
            if stats:
                log.info(f"Dataset stats: truncation_rate={stats.get('truncation_rate', 0):.3f}, "
                        f"padding_rate={stats.get('padding_rate', 0):.3f}, "
                        f"significant_truncations={stats.get('significant_truncation_rate', 0):.3f}")

class ValidationRunner:
    """Handles model validation operations"""
    
    def __init__(self, trainer, accelerator, config):
        self.trainer = trainer
        self.accelerator = accelerator
        self.config = config
        
    def run_validation(self, val_dataloader, step, max_steps):
        """Run validation and return average loss"""
        if not val_dataloader:
            return None
            
        if self.accelerator.is_main_process:
            log.info(f"Running evaluation at step {step}")
        
        self.trainer.model.eval()
        val_loss_sum = 0
        val_steps = 0
        total_tokens = 0
        max_eval_batches = self.config["max_eval_batches"]
        
        try:
            for val_batch in val_dataloader:
                if val_steps >= max_eval_batches:
                    break
                    
                val_loss = self.trainer.eval_step(val_batch)
                val_loss_tensor = self.accelerator.gather(val_loss.detach())
                val_loss_sum += val_loss_tensor.mean().item()
                
                # Count tokens for perplexity calculation
                if 'attention_mask' in val_batch:
                    tokens_in_batch = val_batch['attention_mask'].sum().item()
                else:
                    tokens_in_batch = val_batch['input_ids'].numel()
                total_tokens += tokens_in_batch
                
                val_steps += 1
                
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if self.accelerator.is_main_process:
                log.error(f"Validation error at step {step}: {e}")
            return None
        finally:
            self.trainer.model.train()
            
        if val_steps > 0:
            avg_val_loss = val_loss_sum / val_steps
            perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
            
            if self.accelerator.is_main_process:
                log.info(f"Step {step}/{max_steps}: val_loss={avg_val_loss:.4f}, perplexity={perplexity:.2f}")
                
                # Log comprehensive validation metrics to WandB
                val_metrics = {
                    "val/loss": avg_val_loss,
                    "val/perplexity": perplexity,
                    "val/step": step,
                    "val/tokens_evaluated": total_tokens,
                    "val/batches_evaluated": val_steps
                }
                
                # Add learning rate for reference
                current_lr = self.trainer.lr_scheduler.get_last_lr()[0]
                val_metrics["val/learning_rate"] = current_lr
                
                self.accelerator.log(val_metrics, step=step)
                
            return avg_val_loss
            
        return None

@contextmanager
def cuda_event_manager():
    """Context manager for proper CUDA event lifecycle"""
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        try:
            yield start_event, end_event
        finally:
            del start_event, end_event
            torch.cuda.empty_cache()
    else:
        yield None, None

def verify_model_size(model, expected_size_str):
    """Verify that the loaded model has the expected parameter count"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Expected parameter counts based on official LLaMA 3.2 configs
    expected_counts = {
        "1b": 1_235_814_400,  # 16 layers, 2048 hidden, 32 heads, 128256 vocab
    }
    
    expected_count = expected_counts.get(expected_size_str.lower())
    if expected_count:
        # Allow 20% tolerance
        tolerance = 0.2
        min_expected = expected_count * (1 - tolerance)
        max_expected = expected_count * (1 + tolerance)
        
        if not (min_expected <= total_params <= max_expected):
            log.warning(f"⚠️  Model size mismatch!")
            log.warning(f"   Expected: ~{expected_count:,} parameters ({expected_size_str})")
            log.warning(f"   Actual: {total_params:,} parameters")
        else:
            log.info(f"✅ Model size verified: {total_params:,} parameters (~{expected_size_str})")
    else:
        log.info(f"📊 Model loaded with {total_params:,} parameters")
    
    log.info(f"   Total parameters: {total_params:,}")
    log.info(f"   Trainable parameters: {trainable_params:,}")
    log.info(f"   Model size: ~{total_params / 1e9:.1f}B parameters")
    
    return total_params

def setup_training_components(config, dataset, accelerator):
    """Setup all training components - testable factory function"""
    # Use config's model configuration directly
    model_config = {
        "model_name": config["model_name"],
        "dtype": config["dtype"],
        "sequence_length": config["sequence_length"],
        "batch_size": config["batch_size_per_device"],
        "minimum_sequence_length": config["minimum_sequence_length"],
        "num_training_steps": config.get("num_training_steps", 10000),  # Use get() with default
    }
    
    log.info(f"Loading model with config: {model_config}")
    
    # Import load_model here to avoid circular imports
    from quick_llama.models.model_arch import load_model
    
    # Determine loading strategy based on training mode
    training_mode = config.get('training_mode', 'pretraining')
    checkpoint_path = config.get('checkpoint_path')
    
    if training_mode == 'pretraining':
        # Pretraining from scratch - random initialization
        model = load_model(config=model_config, load_pretrained=False)
    elif training_mode == 'continued_pretraining':
        # Continued pretraining - load from HuggingFace or checkpoint
        if checkpoint_path:
            model = load_model(config=model_config, checkpoint_path=checkpoint_path)
        else:
            model = load_model(config=model_config, load_pretrained=True, instruct=False)
    elif training_mode == 'instruction_tuning':
        # Instruction tuning - load from our pretrained checkpoint or HuggingFace
        if config.get('use_hf_model'):
            model = load_model(config=model_config, load_pretrained=True, instruct=False)
        elif checkpoint_path:
            model = load_model(config=model_config, checkpoint_path=checkpoint_path)
        else:
            raise ValueError("Instruction tuning requires --checkpoint-path or --use-hf-model")
    else:
        raise ValueError(f"Unknown training mode: {training_mode}")
    
    # Verify model size matches expected configuration
    verify_model_size(model, config["model_size"])
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        betas=(config.get('adam_beta1', 0.9), config.get('adam_beta2', 0.95)),  # Paper values
        weight_decay=config['weight_decay'],
        eps=config['adam_epsilon']
    )
    
    # Always calculate max_steps from epochs when num_train_epochs is specified
    if 'num_train_epochs' in config:
        num_epochs = config.get('num_train_epochs', 1.0)
        gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        
        # Create temporary dataloader to get actual batch count
        temp_dataloader = create_streaming_dataloader(dataset['train'], config, accelerator)
        num_batches = len(temp_dataloader)
        num_update_steps_per_epoch = num_batches // gradient_accumulation_steps
        max_steps = int(num_update_steps_per_epoch * num_epochs)
        
        log.info(f"Calculated max_steps: {max_steps} ({num_epochs} epochs × {num_update_steps_per_epoch} update steps/epoch)")
        del temp_dataloader  # Clean up
    else:
        # Use max_steps if specified
        max_steps = config.get('max_steps', 10000)
    
    config['max_steps'] = max_steps  # Store for later use
    config["num_training_steps"] = max_steps  # Update this too
    
    log.info(f"Updated max steps: {max_steps} for {config.get('num_train_epochs', 1.0)} epochs")
    
    # Create dataloaders
    train_dataloader = create_streaming_dataloader(dataset['train'], config, accelerator)
    val_dataloader = create_streaming_dataloader(dataset['validation'], config, accelerator) if 'validation' in dataset else None
    
    # Setup scheduler - use linear scheduler with warmup
    warmup_ratio = config.get('warmup_ratio', 0.05)  # Default 5%
    warmup_steps = int(max_steps * warmup_ratio)
    scheduler_type = config.get('scheduler_type', 'linear')
    min_lr_ratio = config.get('min_lr_ratio', 0.1)  # Decay to 10%
    
    log.info(f"Scheduler: {scheduler_type}, warmup_steps: {warmup_steps} ({warmup_ratio*100}% of {max_steps})")
    
    if scheduler_type == 'cosine':
        import math
        base_lr = config['learning_rate']
        min_ratio = config.get('min_lr_ratio', 0.1)
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / max(1, warmup_steps)
            progress = float(current_step - warmup_steps) / max(1, max_steps - warmup_steps)
            # cosine from 1.0 -> 0.0, then lift by min_ratio
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif scheduler_type == 'linear':
        from transformers import get_linear_schedule_with_warmup
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps
        )
    else:
        # Fallback to cosine
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    
    # Prepare with accelerator
    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader
    )
    
    if val_dataloader:
        val_dataloader = accelerator.prepare(val_dataloader)
    
    # Resume from checkpoint if specified
    start_step = 0
    if config.get('resume_from_checkpoint') and config.get('checkpoint_path'):
        checkpoint_path = config['checkpoint_path']
        if os.path.exists(checkpoint_path):
            log.info(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_step = checkpoint['step']
            log.info(f"Resumed from step {start_step}")
        else:
            log.warning(f"Checkpoint not found: {checkpoint_path}")
    
    config['start_step'] = start_step
    
    return {
        'model': model,
        'optimizer': optimizer, 
        'lr_scheduler': lr_scheduler,
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'max_steps': max_steps
    }

def run_training_epoch(trainer, train_dataloader, metrics, monitor, checkpoint_mgr, 
                      val_runner, val_dataloader, step, max_steps, config):
    """Run single training epoch - testable unit"""
    
    for batch in train_dataloader:
        # Performance monitoring
        with cuda_event_manager() as (start_event, end_event):
            if start_event and monitor.should_monitor_performance(step):
                start_event.record()
            
            # Training step
            try:
                loss = trainer.train_step(batch)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                log.error(f"Training error at step {step}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
            
            # Performance timing
            if end_event and monitor.should_monitor_performance(step):
                end_event.record()
                torch.cuda.synchronize()
        
        # Update metrics
        if trainer.accelerator.num_processes > 1:
            # Multi-GPU: gather with timeout handling
            try:
                loss_tensor = trainer.accelerator.gather(loss.detach())
                metrics.update_train_loss(loss_tensor.mean().item())
            except Exception as e:
                # Fallback: use local loss if gather fails
                log.warning(f"Loss gather failed, using local loss: {e}")
                metrics.update_train_loss(loss.detach().item())
        else:
            # Single GPU: use loss directly
            metrics.update_train_loss(loss.detach().item())
        
        # Only increment step on actual optimizer updates (after gradient accumulation)
        if trainer.accelerator.sync_gradients:
            step += 1
            
            # Check if we've reached max steps
            if step >= max_steps:
                return step
            
            # Logging
            if metrics.should_log(step) and trainer.accelerator.is_main_process:
                avg_loss = metrics.get_avg_train_loss()
                current_lr = trainer.lr_scheduler.get_last_lr()[0]
                
                include_gpu = monitor.should_monitor_gpu(step)
                monitor.log_training_metrics(step, avg_loss, current_lr, max_steps, include_gpu, trainer.accelerator)
                monitor.log_dataset_stats(train_dataloader.dataset, step)
                
                metrics.reset()
            
            # Validation
            if metrics.should_evaluate(step, max_steps) and val_dataloader:
                avg_val_loss = val_runner.run_validation(val_dataloader, step, max_steps)
                if avg_val_loss and avg_val_loss < metrics.best_val_loss:
                    metrics.best_val_loss = avg_val_loss
                    if trainer.accelerator.is_main_process:
                        log.info(f"New best validation loss: {metrics.best_val_loss:.4f}")
            
            # Checkpointing - remove sync to avoid hangs
            if metrics.should_save_checkpoint(step):
                if trainer.accelerator.is_main_process:
                    try:
                        # Save without accelerator sync
                        torch.save({
                            'model_state_dict': trainer.model.state_dict(),
                            'optimizer_state_dict': trainer.optimizer.state_dict(),
                            'scheduler_state_dict': trainer.lr_scheduler.state_dict(),
                            'step': step,
                            'config': config
                        }, os.path.join(config["output_dir"], f"checkpoint_step_{step}.pt"))
                        log.info(f"Saved simple checkpoint at step {step}")
                    except Exception as e:
                        log.error(f"Failed to save checkpoint: {e}")
                # No sync - continue immediately
        
        # Memory cleanup
        if step % config["memory_cleanup_interval"] == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return step

def train_meco_model(config, dataset=None):
    """Main training function - now modular and testable"""
    log.info("Starting MECO model training...")
    
    # Download HF dataset if needed
    download_hf_dataset_if_needed(config)
    
    # Load dataset if not provided
    if dataset is None:
        dataset = load_meco_dataset(config)
    
    # Validate environment
    validate_environment(config)
    
    # Setup accelerator with config values
    kwargs_handlers = [
        DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.FP16),
        InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=config['communication_timeout_seconds'])),
    ]
    
    accelerator = Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        log_with=config["log_with"],
        project_dir=config["output_dir"],
        kwargs_handlers=kwargs_handlers,
        device_placement=True,
    )
    accelerator.init_trackers(config['experiment_name'])
    
    # Create output directory and save config
    os.makedirs(config["output_dir"], exist_ok=True)
    if accelerator.is_main_process:
        save_config(config)
    
    # Setup all training components
    components = setup_training_components(config, dataset, accelerator)
    
    # Initialize training modules
    trainer = ModelTrainer(
        components['model'], components['optimizer'], 
        components['lr_scheduler'], accelerator, config
    )
    metrics = TrainingMetrics(config)
    monitor = SystemMonitor(config)
    checkpoint_mgr = CheckpointManager(accelerator, config)
    val_runner = ValidationRunner(trainer, accelerator, config)
    
    if accelerator.is_main_process:
        log.info(f"Model parameter count: {sum(p.numel() for p in components['model'].parameters()):,}")
        log.info(f"Training on {accelerator.num_processes} processes")
        log.info(f"Train batches: {len(components['train_dataloader'])}")
        if components['val_dataloader']:
            log.info(f"Validation batches: {len(components['val_dataloader'])}")
    
    # Training loop
    step = config.get('start_step', 0)
    for epoch in range(config.get("num_epochs", 1)):
        if accelerator.is_main_process:
            log.info(f"Starting epoch {epoch + 1}")
        
        # Set epoch for distributed sampler
        if hasattr(components['train_dataloader'].sampler, 'set_epoch'):
            components['train_dataloader'].sampler.set_epoch(epoch)
        
        components['model'].train()
        
        # Run training epoch
        step = run_training_epoch(
            trainer, components['train_dataloader'], metrics, monitor,
            checkpoint_mgr, val_runner, components['val_dataloader'],
            step, components['max_steps'], config
        )
        
        if step >= components['max_steps']:
            break
    
    # Final checkpoint
    checkpoint_mgr.save_checkpoint(
        components['model'], components['optimizer'], 
        components['lr_scheduler'], step
    )
    
    if accelerator.is_main_process:
        log.info(f"Training completed! Best validation loss: {metrics.best_val_loss:.4f}")

def save_config(config):
    """Save configuration to file - testable utility"""
    config_path = os.path.join(config["output_dir"], "meco_config.json")
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)
    
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    log.info(f"Saved config to: {config_path}")

def parse_meco_args():
    """Parse MECO-specific command line arguments"""
    parser = argparse.ArgumentParser(description="Train LLaMA models on MECO datasets")
    
    # Required arguments
    parser.add_argument("--meco-dataset-path", required=True,
                       help="Path to MECO dataset")
    parser.add_argument("--experiment-name", required=True,
                       help="Name for this experiment")
    
    # Model configuration
    parser.add_argument("--model-size", default=None, choices=["1b"],
                       help="Model size to train")
    parser.add_argument("--sequence-length", type=int, default=None,
                       help="Maximum sequence length")
    
    # Training configuration
    parser.add_argument("--batch-size-per-device", type=int, default=None,
                       help="Batch size per GPU")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Maximum training steps")
    parser.add_argument("--save-steps", type=int, default=None,
                       help="Steps between checkpoints")
    
    # Training modes
    parser.add_argument("--training-mode", default="pretraining", 
                       choices=["pretraining", "continued_pretraining", "instruction_tuning"],
                       help="Training mode: pretraining, continued_pretraining, or instruction_tuning")
    parser.add_argument("--use-lmsys-chat", action="store_true",
                       help="Use LMSYS Chat-1M dataset for instruction tuning")
    parser.add_argument("--checkpoint-path", type=str,
                       help="Path to checkpoint for continued pretraining or instruction tuning")
    parser.add_argument("--use-hf-model", action="store_true",
                       help="Use HuggingFace model directly for instruction tuning")
    parser.add_argument("--config-file", default=None,
                       help="Path to config YAML file (default: meco_config.yaml)")
    
    parser.add_argument("--upload-dataset", action="store_true",
                       help="Upload local MECO dataset to HuggingFace and exit")
    parser.add_argument("--hf-dataset-name", default=None,
                       help="HuggingFace dataset name (default: folder name)")
    
    # Output configuration
    parser.add_argument("--output-dir", default=None,
                       help="Output directory (default: logs/meco_{experiment_name})")
    
    return parser.parse_args()

def main():
    """Main training function"""
    # Parse arguments
    args = parse_meco_args()
    
    # Handle dataset upload
    if args.upload_dataset:
        if not args.meco_dataset_path:
            log.error("--meco-dataset-path required for upload")
            sys.exit(1)
        
        try:
            upload_meco_dataset_to_hf(args.meco_dataset_path, args.hf_dataset_name)
            log.info("🎉 Dataset upload completed successfully!")
            sys.exit(0)
        except Exception as e:
            log.error(f"❌ Upload failed: {e}")
            sys.exit(1)
    
    # Generate MECO configuration
    log.info("Generating MECO training configuration...")
    
    # Collect CLI overrides
    config_overrides = {}
    if args.model_size:
        config_overrides['model_size'] = args.model_size
    if args.sequence_length:
        config_overrides['sequence_length'] = args.sequence_length
    if args.batch_size_per_device:
        config_overrides['batch_size_per_device'] = args.batch_size_per_device
    if args.gradient_accumulation_steps:
        config_overrides['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    if args.learning_rate:
        config_overrides['learning_rate'] = args.learning_rate
    if args.training_mode:
        config_overrides['training_mode'] = args.training_mode
    if args.checkpoint_path:
        config_overrides['checkpoint_path'] = args.checkpoint_path
    if args.use_hf_model:
        config_overrides['use_hf_model'] = args.use_hf_model
    if args.max_steps:
        config_overrides['max_steps'] = args.max_steps
    if args.save_steps:
        config_overrides['save_steps'] = args.save_steps
    if args.output_dir:
        config_overrides['output_dir'] = args.output_dir
    if args.training_mode:
        config_overrides['training_mode'] = args.training_mode
    if args.use_lmsys_chat:
        config_overrides['use_lmsys_chat'] = args.use_lmsys_chat
    
    config = get_meco_config(
        meco_dataset_path=args.meco_dataset_path,
        experiment_name=args.experiment_name,
        config_overrides=config_overrides,
        config_file=args.config_file
    )
    
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    log.info(f"Experiment: {config['experiment_name']}")
    log.info(f"Dataset: {config['meco_dataset_path']}")
    log.info(f"Model: {config['model_name']}")
    log.info(f"Epochs: {config.get('num_train_epochs', 1.0)}")
    log.info(f"Output: {config['output_dir']}")
    
    # Validate dataset exists (only for MECO datasets without HF dataset)
    if not (config.get('training_mode') == 'instruction_tuning' and config.get('use_lmsys_chat', False)):
        if 'hf_dataset_name' not in config and not os.path.exists(config['meco_dataset_path']):
            log.error(f"MECO dataset not found: {config['meco_dataset_path']}")
            sys.exit(1)
    
    # Load dataset
    log.info("Loading dataset...")
    try:
        dataset = load_meco_dataset_for_training(config)
        log.info(f"✅ Dataset loaded successfully:")
        log.info(f"   Train samples: {len(dataset['train']):,}")
        log.info(f"   Validation samples: {len(dataset['validation']):,}")
    except Exception as e:
        log.error(f"❌ Failed to load dataset: {e}")
        sys.exit(1)
    
    # Train model
    try:
        train_meco_model(config, dataset)
        log.info("🎉 MECO training completed successfully!")
    except Exception as e:
        log.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
