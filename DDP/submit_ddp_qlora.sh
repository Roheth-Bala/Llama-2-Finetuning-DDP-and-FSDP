#!/bin/bash
#SBATCH --job-name=ddp-qlora-llama
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1

# --- Log file paths ---
# We use %j here, which is the SLURM_JOB_ID
#SBATCH --output=/home/e/e1415353/my_projects/Assignment7Folder/Assignment7/logs_alpaca/ddp_qlora_%j.out
#SBATCH --error=/home/e/e1415353/my_projects/Assignment7Folder/Assignment7/logs_alpaca/ddp_qlora_%j.err

# --- Environment Setup ---
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export NCCL_TIMEOUT=1800


# --- Activate existing Conda environment 'ml' ---
source ~/.bashrc

# Initialize Conda for non-interactive shells
echo "Initializing Conda..."
eval "$(conda shell.bash hook)"

# Activate your environment
echo "Activating Conda environment: ml"
conda activate ml
echo "Conda environment 'ml' activated."

# --- Define and create output directories 
# $SLURM_JOB_ID is available after the job starts.

# Directory for SLURM .out/.err files 
LOG_DIR="/home/e/e1415353/my_projects/Assignment7Folder/Assignment7/logs_alpaca"

# Directory for Python/model results
RESULTS_DIR="/home/e/e1415353/my_projects/Assignment7Folder/Assignment7/results/ddp_qlora_${SLURM_JOB_ID}"

echo "Creating directories..."
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

echo "Log directory: $LOG_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Starting DDP training..."


srun python train_ddp_qlora.py \
    --model_id meta-llama/Llama-2-7b-hf \
    --dataset_name databricks/databricks-dolly-15k \
    --output_dir "$RESULTS_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --max_seq_length 1024 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --logging_steps 100

echo "DDP training script finished. Results saved to $RESULTS_DIR"