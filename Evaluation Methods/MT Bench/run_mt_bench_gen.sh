#!/bin/bash

#SBATCH --job-name=mt_bench_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/tmp/%u_%j_mt_bench_gen.out
#SBATCH --error=/tmp/%u_%j_mt_bench_gen.err

echo "==========================================="
echo "MT-Bench Answer Generation Job Started: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "==========================================="

# --- 1. Environment Setup ---
JOB_TMP_DIR=$(mktemp -d /tmp/${USER}_job${SLURM_JOB_ID}_mt_bench_gen_XXXX)
echo "Created temporary directory: $JOB_TMP_DIR"

# Copy FastChat directory (assuming it's in the project dir)
cp -r /home/e/e1415353/my_projects/Assignment7Folder/Assignment7/FastChat $JOB_TMP_DIR/
cd $JOB_TMP_DIR/FastChat

# Use absolute path to conda Python
PYTHON_EXEC="/home/e/e1415353/my_projects/miniconda3/envs/ml/bin/python"
echo "Using Python: $($PYTHON_EXEC --version)"

# Log directory path
LOG_DIR="/home/e/e1415353/my_projects/Assignment7Folder/Assignment7/logs_mt_bench"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/mt_bench_gen_${SLURM_JOB_ID}.log"


# --- 2. Run Generation for Fine-tuned Model ---
echo "=== Generating Answers for Fine-tuned Model ==="
$PYTHON_EXEC -m fastchat.llm_judge.gen_model_answer \
    --model-path /home/e/e1415353/my_projects/Assignment7Folder/Assignment7/results_241919/dolly_lora_llama2_7b_tuned_v1/final \
    --model-id Llama2-7B-Dolly-QLoRA-Finetuned \
    --answer-file data/mt_bench/model_answer/Llama2-7B-Dolly-QLoRA-Finetuned.jsonl \
    --num-gpus-per-model 1 \
    --num-gpus-total 2 \
    --max-new-token 1024 2>&1 | tee "$LOG_FILE.finetuned_gen"

# --- 3. Run Generation for Base Model ---
echo "=== Generating Answers for Base Model ==="
$PYTHON_EXEC -m fastchat.llm_judge.gen_model_answer \
    --model-path meta-llama/Llama-2-7b-hf \
    --model-id Llama2-7B-HF-Base \
    --answer-file data/mt_bench/model_answer/Llama2-7B-HF-Base.jsonl \
    --num-gpus-per-model 1 \
    --num-gpus-total 2 \
    --max-new-token 1024 2>&1 | tee "$LOG_FILE.base_gen"

# --- 4. Copy Results Back ---
RESULTS_DIR="/home/e/e1415353/my_projects/Assignment7Folder/Assignment7/mt_bench_answers_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"
cp -r data/mt_bench/model_answer/* "$RESULTS_DIR/"
echo "Answers copied to: $RESULTS_DIR"

# Copy SLURM logs
cp "/tmp/${USER}_${SLURM_JOB_ID}_mt_bench_gen.out" "$LOG_DIR/mt_bench_gen_${SLURM_JOB_ID}.slurm.out"
cp "/tmp/${USER}_${SLURM_JOB_ID}_mt_bench_gen.err" "$LOG_DIR/mt_bench_gen_${SLURM_JOB_ID}.slurm.err"

# Cleanup
rm -rf "$JOB_TMP_DIR"
echo "MT-Bench Answer Generation Job Finished: $(date)"
echo "==========================================="