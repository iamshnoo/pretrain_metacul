#!/bin/bash
#SBATCH --job-name=meco_asia
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --mem=80G
#SBATCH --time=5-00:00:00
#SBATCH --output=/scratch/${USER}/pretrain_metacul/src/logs/meco_train_%A_%a.out.txt
#SBATCH --error=/scratch/${USER}/pretrain_metacul/src/logs/meco_train_%A_%a.err.txt

# MECO Training Script - Supports both single job and array job modes
#
# Single job usage: sbatch train_meco.slurm <config_file> [num_gpus]
# Array job usage: sbatch --array=1-N train_meco.slurm [num_gpus]
#
# Examples:
#   # Single job
#   sbatch train_meco.slurm train_configs/continents/africa/with_metadata/pretraining.yaml
#   sbatch train_meco.slurm train_configs/continents/africa/with_metadata/pretraining.yaml 2
#
#   # Array job (runs all configs)
#   sbatch --array=1-416 train_meco.slurm
#   sbatch --array=1-416 train_meco.slurm 4
#
#   # Array job subset
#   sbatch --array=1-10 train_meco.slurm

echo "=== MECO Training Job ==="
echo "Job ID: $SLURM_JOB_ID"
if [ -n "$SLURM_ARRAY_JOB_ID" ]; then
    echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
    echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
fi
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Determine if this is an array job or single job
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # Array job mode - read config from train_config_paths.txt
    CONFIG_PATHS_FILE="asia_pretraining_configs.txt"

    if [ ! -f "$CONFIG_PATHS_FILE" ]; then
        echo "Error: Config paths file not found: $CONFIG_PATHS_FILE"
        echo "Run generate_train_configs.py first to create this file"
        exit 1
    fi

    # Read the config file for this array task
    RELATIVE_CONFIG_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONFIG_PATHS_FILE")

    # Convert to absolute path based on parent directory
    PARENT_DIR=$(dirname "$(pwd)")
    CONFIG_FILE="$PARENT_DIR/$RELATIVE_CONFIG_FILE"

    NUM_GPUS="${1:-1}"

    if [ -z "$CONFIG_FILE" ]; then
        echo "Error: No config found for array task ID $SLURM_ARRAY_TASK_ID"
        exit 1
    fi

    echo "Array job mode"
    echo "Config paths file: $CONFIG_PATHS_FILE"
else
    # Single job mode - config file passed as argument
    CONFIG_FILE="$1"
    NUM_GPUS="${2:-4}"

    if [ -z "$CONFIG_FILE" ]; then
        echo "Error: Config file not specified"
        echo "Usage: sbatch train_meco.slurm <config_file> [num_gpus]"
        echo "Example: sbatch train_meco.slurm train_configs/continents/asia/with_metadata/pretraining.yaml"
        exit 1
    fi

    echo "Single job mode"
fi

# Validate config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Config file: $CONFIG_FILE"
echo "Number of GPUs: $NUM_GPUS"

# Update SLURM resource allocation for multi-GPU jobs
if [ "$NUM_GPUS" -gt 4 ]; then
    echo "Note: Default is 4 GPUs. For more GPUs, ensure SLURM job was submitted with:"
    echo "  --gres=gpu:A100.80gb:$NUM_GPUS (or H100.80gb:$NUM_GPUS)"
    echo "  May need different partition for >4 GPUs"
elif [ "$NUM_GPUS" -lt 4 ]; then
    echo "Note: Using fewer than default 4 GPUs. Override with:"
    echo "  --gres=gpu:A100.80gb:$NUM_GPUS"
fi

# Set up environment
source /home/${USER}/pretrain-env/bin/activate

# NCCL debugging and timeout settings
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Network settings for better stability
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=0

# Performance optimizations
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0

# Verify environment
echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Set CUDA device visibility based on requested GPUs and available GPUs
AVAILABLE_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
echo "Available GPUs: $AVAILABLE_GPUS"

if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    echo "Using all available GPUs: $AVAILABLE_GPUS"
    NUM_GPUS=$AVAILABLE_GPUS
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ "$NUM_GPUS" -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES=0,1
elif [ "$NUM_GPUS" -eq 4 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
elif [ "$NUM_GPUS" -eq 8 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
else
    echo "Error: Unsupported number of GPUs: $NUM_GPUS"
    echo "Supported: 1, 2, 4, 8"
    exit 1
fi

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Change to src directory
cd /scratch/${USER}/pretrain_metacul/src

# Verify training script exists
if [ ! -f "train.py" ]; then
    echo "Error: Training script not found: train.py"
    exit 1
fi

# Extract required arguments from config file
MECO_DATASET_PATH=$(python -c "
import yaml
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    print(config.get('meco_dataset_path', ''))
except:
    print('')
")

EXPERIMENT_NAME=$(python -c "
import yaml
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    print(config.get('experiment_name', ''))
except:
    print('')
")

if [ -z "$MECO_DATASET_PATH" ] || [ -z "$EXPERIMENT_NAME" ]; then
    echo "Error: Could not extract meco_dataset_path or experiment_name from config file"
    exit 1
fi

echo "Dataset path: $MECO_DATASET_PATH"
echo "Experiment name: $EXPERIMENT_NAME"
echo "=========================================="

# Run training with appropriate launcher
if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU training
    echo "Running single GPU training..."
    python train.py --meco-dataset-path "$MECO_DATASET_PATH" --experiment-name "$EXPERIMENT_NAME" --config-file "$CONFIG_FILE"
else
    # Multi-GPU training with accelerate
    echo "Running multi-GPU training with accelerate..."

    # Create persistent accelerate config directory
    ACCELERATE_CONFIG_DIR="/scratch/${USER}/pretrain_metacul/src/logs/accelerate_configs"
    mkdir -p "$ACCELERATE_CONFIG_DIR"

    # Create accelerate config for multi-GPU
    ACCELERATE_CONFIG_FILE="$ACCELERATE_CONFIG_DIR/multi_gpu_${NUM_GPUS}_${SLURM_JOB_ID}.yaml"

    # Generate random port to avoid conflicts
    MASTER_PORT=$((29500 + RANDOM % 1000))

    cat > "$ACCELERATE_CONFIG_FILE" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: $NUM_GPUS
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
main_process_port: $MASTER_PORT
EOF

    echo "Created accelerate config: $ACCELERATE_CONFIG_FILE"

    # Launch with accelerate
    accelerate launch --config_file "$ACCELERATE_CONFIG_FILE" \
        train.py --meco-dataset-path "$MECO_DATASET_PATH" --experiment-name "$EXPERIMENT_NAME" --config-file "$CONFIG_FILE"
fi

# Capture exit code
EXIT_CODE=$?

echo "=========================================="
echo "Training completed for: $EXPERIMENT_INFO"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

# Log completion
COMPLETION_LOG="/scratch/${USER}/pretrain_metacul/src/logs/meco_training_completion.log"
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    echo "$(date): $SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID: $CONFIG_FILE: $EXPERIMENT_INFO: EXIT_CODE=$EXIT_CODE" >> "$COMPLETION_LOG"
else
    echo "$(date): $SLURM_JOB_ID: $CONFIG_FILE: $EXPERIMENT_INFO: EXIT_CODE=$EXIT_CODE" >> "$COMPLETION_LOG"
fi

# Show final model info if successful
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training successful! Checking outputs..."

    # Extract output directory from config
    OUTPUT_DIR=$(python -c "
import yaml
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    print(config.get('output_dir_template', 'logs'))
except:
    print('logs')
")

    if [ -d "$OUTPUT_DIR" ]; then
        echo "Output directory: $OUTPUT_DIR"
        echo "Checkpoints:"
        find "$OUTPUT_DIR" -name "step_*" -type d | tail -5

        echo "Model files:"
        find "$OUTPUT_DIR" -name "*.safetensors" | head -3
    fi
else
    echo "Training failed with exit code: $EXIT_CODE"
    echo "Check error logs for details"
fi

exit $EXIT_CODE
