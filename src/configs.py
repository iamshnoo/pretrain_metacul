#!/usr/bin/env python3
"""
Generate training configuration files for all MECO splits and training modes.

This script creates YAML config files for:
- All splits from meco_splits.txt
- Both with_metadata and without_metadata variants
- 4 training modes: pretraining, continued_pretraining, pretrain+instruct, continued_pretrain+instruct

Directory structure:
train_configs/
├── continents/
│   ├── africa/
│   │   ├── with_metadata/
│   │   │   ├── pretraining.yaml
│   │   │   ├── continued_pretraining.yaml
│   │   │   ├── pretrain_instruct.yaml
│   │   │   └── continued_pretrain_instruct.yaml
│   │   └── without_metadata/
│   │       └── ... (same 4 files)
│   └── ... (other continents)
└── ... (other split types)
"""

import os
import yaml
from pathlib import Path

def parse_splits_file(splits_file):
    """Parse meco_splits.txt to extract split configurations."""
    splits = []
    with open(splits_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Parse: --split-type continents --split-name africa
                parts = line.split()
                split_type = None
                split_name = None

                for i, part in enumerate(parts):
                    if part == '--split-type' and i + 1 < len(parts):
                        split_type = parts[i + 1]
                    elif part == '--split-name' and i + 1 < len(parts):
                        split_name = parts[i + 1]

                if split_type and split_name:
                    splits.append((split_type, split_name))

    return splits

def create_base_config():
    """Create base configuration common to all training modes."""
    return {
        'model_size': '1b',
        'model_name': 'meta-llama/Llama-3.2-1B',
        'sequence_length': 1024,
        'minimum_sequence_length': 64,  # Minimum sequence length for training
        'batch_size_per_device': 4,
        'gradient_accumulation_steps': 8,   # Global batch = 4×8×2 = 64 (same as original)
        'learning_rate': 1e-5,  # 2x higher for 4 GPUs (global batch = 128)
        'adam_beta1': 0.9,      # Paper value
        'adam_beta2': 0.95,     # Paper value (not default 0.999)
        'weight_decay': 0.033,  # Paper value
        'adam_epsilon': 1e-8,   # Adam optimizer epsilon
        'warmup_ratio': 0.01,   # 1% warmup for faster training
        'num_train_epochs': 1.0,   # FIXED EPOCHS - same data exposure for all splits
        'scheduler_type': 'cosine',  # Cosine decay like paper
        'min_lr_ratio': 0.1,    # Decay to 10% of peak like paper
        'save_steps': 1000,        # Save every 1k steps
        'eval_steps': 5000,        # Evaluate every 5k steps
        'steps_between_evals': 5000,  # Steps between evaluations
        'logging_steps': 10,       # Log every 10 update steps to reduce WandB overhead
        'dtype': 'bfloat16',
        'seed': 42,
        'dataloader_num_workers': 0,  # Disable multiprocessing for speed
        'remove_unused_columns': False,
        'dataloader_pin_memory': True,
        'gradient_checkpointing': True,  # Keep enabled for 40GB GPUs
        'use_wandb': True,
        'wandb_project': 'meco-training',
        'communication_timeout_seconds': 1800,  # 30 minutes timeout for distributed training
        'mixed_precision': 'bf16',
        'clip_grad_norm': 1.0,
        'save_limit': 3,
        'log_with': 'wandb',  # Logging backend for Accelerator
        'gpu_monitor_interval': 1800,  # Monitor GPU every 30 minutes
        'performance_monitor_interval': 600,  # Performance monitoring less frequent
        'memory_cleanup_interval': 1000,  # Memory cleanup interval in steps
        'max_eval_batches': 100  # 100 validation samples
    }

def create_pretraining_config(split_type, split_name, metadata_variant):
    """Create pretraining config."""
    config = create_base_config()

    # Convert split_type format for path
    split_type_path = split_type.replace('-', '_')

    # Get project paths relative to this script
    src_dir = Path(__file__).parent
    project_root = src_dir.parent if src_dir.name == 'src' else src_dir

    config.update({
        'training_mode': 'pretraining',
        'hf_dataset_name': f'iamshnoo/meco-{split_name.replace("_", "-")}-{"with-metadata" if metadata_variant == "with_metadata" else "without-metadata"}',
        'meco_dataset_path': str(project_root / 'training_data' / 'downloaded_datasets' / split_type_path / split_name / metadata_variant),
        'experiment_name': f'{split_name}_{metadata_variant}_pretraining',
        'output_dir_template': str(src_dir / 'logs' / f'pretraining_{split_type_path}_{split_name}_{metadata_variant}'),
        'checkpoint_dir_template': str(src_dir / 'logs' / f'pretraining_{split_type_path}_{split_name}_{metadata_variant}' / 'checkpoints')
    })

    return config

def create_continued_pretraining_config(split_type, split_name, metadata_variant):
    """Create continued pretraining config."""
    config = create_base_config()

    split_type_path = split_type.replace('-', '_')

    # Get project paths relative to this script
    src_dir = Path(__file__).parent
    project_root = src_dir.parent if src_dir.name == 'src' else src_dir

    config.update({
        'training_mode': 'continued_pretraining',
        'hf_dataset_name': f'your-username/meco-{split_name.replace("_", "-")}-{"with-metadata" if metadata_variant == "with_metadata" else "without-metadata"}',
        'meco_dataset_path': str(project_root / 'training_data' / 'downloaded_datasets' / split_type_path / split_name / metadata_variant),
        'experiment_name': f'{split_name}_{metadata_variant}_continued_pretraining',
        'output_dir_template': str(src_dir / 'logs' / f'continued_pretraining_{split_type_path}_{split_name}_{metadata_variant}'),
        'checkpoint_dir_template': str(src_dir / 'logs' / f'continued_pretraining_{split_type_path}_{split_name}_{metadata_variant}' / 'checkpoints'),
        'num_train_epochs': 0.5  # Shorter for continued pretraining
    })

    return config

def create_pretrain_instruct_config(split_type, split_name, metadata_variant):
    """Create pretraining + instruction tuning config."""
    config = create_base_config()

    split_type_path = split_type.replace('-', '_')

    config.update({
        'training_mode': 'instruction_tuning',
        'use_lmsys_chat': True,
        'checkpoint_path': f'logs/pretraining_{split_type_path}_{split_name}_{metadata_variant}/final_checkpoint',
        'experiment_name': f'{split_name}_{metadata_variant}_pretrain_instruct',
        'output_dir_template': f'logs/pretrain_instruct_{split_type_path}_{split_name}_{metadata_variant}',
        'checkpoint_dir_template': f'logs/pretrain_instruct_{split_type_path}_{split_name}_{metadata_variant}/checkpoints',
        'learning_rate': 5e-5,  # Lower LR for instruction tuning
        'num_train_epochs': 3.0,  # Multiple epochs for instruction tuning
        'save_steps': 1000,
        'eval_steps': 500
    })

    return config

def create_continued_pretrain_instruct_config(split_type, split_name, metadata_variant):
    """Create continued pretraining + instruction tuning config."""
    config = create_base_config()

    split_type_path = split_type.replace('-', '_')

    config.update({
        'training_mode': 'instruction_tuning',
        'use_lmsys_chat': True,
        'checkpoint_path': f'logs/continued_pretraining_{split_type_path}_{split_name}_{metadata_variant}/final_checkpoint',
        'experiment_name': f'{split_name}_{metadata_variant}_continued_pretrain_instruct',
        'output_dir_template': f'logs/continued_pretrain_instruct_{split_type_path}_{split_name}_{metadata_variant}',
        'checkpoint_dir_template': f'logs/continued_pretrain_instruct_{split_type_path}_{split_name}_{metadata_variant}/checkpoints',
        'learning_rate': 5e-5,  # Lower LR for instruction tuning
        'num_train_epochs': 3.0,  # Multiple epochs for instruction tuning
        'save_steps': 1000,
        'eval_steps': 500
    })

    return config

def main():
    # Paths
    src_dir = Path(__file__).parent
    metacul_dir = src_dir.parent
    train_configs_dir = metacul_dir / 'train_configs'
    splits_file = src_dir / 'splits.txt'

    # Create train_configs directory
    train_configs_dir.mkdir(exist_ok=True)

    # Parse splits
    splits = parse_splits_file(splits_file)
    print(f"Found {len(splits)} splits to process")

    # Training modes and their config generators
    training_modes = {
        'pretraining': create_pretraining_config,
        # 'continued_pretraining': create_continued_pretraining_config,
        # 'pretrain_instruct': create_pretrain_instruct_config,
        # 'continued_pretrain_instruct': create_continued_pretrain_instruct_config
    }

    # Metadata variants
    metadata_variants = ['with_metadata', 'without_metadata']

    total_configs = 0

    for split_type, split_name in splits:
        print(f"\nProcessing {split_type}/{split_name}")

        # Create directory structure
        split_dir = train_configs_dir / split_type / split_name

        for metadata_variant in metadata_variants:
            variant_dir = split_dir / metadata_variant
            variant_dir.mkdir(parents=True, exist_ok=True)

            for mode_name, config_generator in training_modes.items():
                config = config_generator(split_type, split_name, metadata_variant)

                # Save config file
                config_file = variant_dir / f'{mode_name}.yaml'
                with open(config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                total_configs += 1
                print(f"  Created: {config_file.relative_to(train_configs_dir)}")

    # Save all config paths to text file for SLURM array jobs
    os.makedirs(src_dir / 'logs', exist_ok=True)
    config_paths_file = src_dir / 'logs' / 'train_config_paths.txt'
    all_config_paths = []

    for split_type, split_name in splits:
        split_dir = train_configs_dir / split_type / split_name
        for metadata_variant in metadata_variants:
            variant_dir = split_dir / metadata_variant
            for mode_name in training_modes.keys():
                config_file = variant_dir / f'{mode_name}.yaml'
                # Store relative path from metacul root
                relative_path = config_file.relative_to(metacul_dir)
                all_config_paths.append(str(relative_path))

    # Write config paths file
    with open(config_paths_file, 'w') as f:
        for path in all_config_paths:
            f.write(f"{path}\n")

    print(f"\n✅ Generated {total_configs} configuration files in {train_configs_dir}")
    print(f"✅ Saved config paths to {config_paths_file} for SLURM array jobs")
    print(f"\nDirectory structure:")
    print(f"{train_configs_dir}/")
    for split_type, split_name in splits[:2]:  # Show first 2 as examples
        print(f"├── {split_type}/")
        print(f"│   ├── {split_name}/")
        print(f"│   │   ├── with_metadata/")
        for mode in training_modes.keys():
            print(f"│   │   │   ├── {mode}.yaml")
        print(f"│   │   └── without_metadata/")
        for mode in training_modes.keys():
            print(f"│   │       ├── {mode}.yaml")
    print("│   └── ...")
    print("└── ...")

    print(f"\nSLURM array job usage:")
    print(f"sbatch --array=1-{len(all_config_paths)} scripts/train_meco.slurm")

if __name__ == '__main__':
    main()
