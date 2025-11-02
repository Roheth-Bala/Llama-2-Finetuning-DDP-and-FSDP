#!/bin/bash
#SBATCH --job-name=alpaca_pair_both
#SBATCH --output=logs/alpacapair_%j.out
#SBATCH --error=logs/alpacapair_%j.err
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00 # Increased time for two model evaluations

set -euo pipefail

# --- 1. Activate Environment ---
# Make sure to activate your conda env!
source "/home/e/e1415353/my_projects/miniconda3/bin/activate" ml
echo "Conda env 'ml' activated."
echo "Using Python: $(which python)"

# --- 2. Set Variables ---
FINETUNED_MODEL_PATH="/home/e/e1415353/my_projects/Assignment7Folder/Assignment7/results_235293/dolly_lora_llama2_7b_smooth_integrated_fixed_v3/final"
BASELINE_MODEL_PATH="meta-llama/Llama-2-7b-hf" 
NUM_INSTRUCTIONS=800 
MAIN_OUTPUT_DIR="./pairwise_alpacaeval_results_${SLURM_JOB_ID}"
JUDGE_MODEL="prometheus-eval/prometheus-7b-v2.0"

echo "==========================================="
echo "Starting AlpacaEval Pairwise Job $SLURM_JOB_ID"
echo "Judge Model: $JUDGE_MODEL"
echo "Main Output Directory: $MAIN_OUTPUT_DIR"
echo "==========================================="

# Create the main output directory
mkdir -p "$MAIN_OUTPUT_DIR"

# --- 3. Evaluate Fine-tuned Model vs text-davinci-003 ---
echo "--- Evaluating Fine-tuned Model ---"
echo "Model Path (A): $FINETUNED_MODEL_PATH"
echo "Baseline (B): text-davinci-003"
FINETUNED_OUTPUT_DIR="$MAIN_OUTPUT_DIR/finetuned_model_results"

python alpaca_pairwise.py \
    --your_model_path "$FINETUNED_MODEL_PATH" \
    --num_instructions $NUM_INSTRUCTIONS \
    --output_dir "$FINETUNED_OUTPUT_DIR" \
    --judge_model "$JUDGE_MODEL"

echo "Fine-tuned model evaluation complete. Results in: $FINETUNED_OUTPUT_DIR"
echo "-------------------------------------------"


# --- 4. Evaluate Baseline Model vs text-davinci-003 ---
echo "--- Evaluating Baseline Model ---"
echo "Model Path (A): $BASELINE_MODEL_PATH"
echo "Baseline (B): text-davinci-003"
BASELINE_OUTPUT_DIR="$MAIN_OUTPUT_DIR/baseline_model_results"

python alpaca_pairwise.py \
    --your_model_path "$BASELINE_MODEL_PATH" \
    --num_instructions $NUM_INSTRUCTIONS \
    --output_dir "$BASELINE_OUTPUT_DIR" \
    --judge_model "$JUDGE_MODEL"

echo "Baseline model evaluation complete. Results in: $BASELINE_OUTPUT_DIR"
echo "-------------------------------------------"


# --- 5. Finalization ---
echo "Job completed successfully."
echo "Final results are located in: $MAIN_OUTPUT_DIR"
echo "==========================================="
