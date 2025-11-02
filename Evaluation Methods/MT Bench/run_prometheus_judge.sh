#!/bin/bash
#SBATCH --job-name=mtbench_prometheus
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00



# Setting this to  our project's *FastChat* directory 

BASE_DIR="/home/e/e1415353/my_projects/Assignment7Folder/Assignment7/FastChat/fastchat/llm_judge"

# Setting the absolute path to your generated answers ---
ANSWERS_DIR="/home/e/e1415353/my_projects/Assignment7Folder/Assignment7/mt_bench_answers_241073"

#  Set the models to test 
MODELS_TO_TEST=(
    "Llama2-7B-HF-Base"
    "Llama2-7B-Dolly-QLoRA-Finetuned"
)

# --- Set the Judge Model ---
JUDGE_MODEL="prometheus-eval/prometheus-7b-v2.0"
PYTHON_SCRIPT="./judge_mtbench_local.py"
QUESTIONS_FILE="data/mt_bench/question.jsonl"

LOG_DIR="/home/e/e1415353/my_projects/Assignment7Folder/Assignment7/logs_mt_bench"
mkdir -p "$LOG_DIR"

# --- Set SLURM output paths ---
#SBATCH --output=${LOG_DIR}/mtbench_prometheus_%j.out
#SBATCH --error=${LOG_DIR}/mtbench_prometheus_%j.err


set -euo pipefail
cd "$BASE_DIR"
echo "Changed directory to: $(pwd)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Judge Model: $JUDGE_MODEL"
echo "Answers Source: $ANSWERS_DIR"
echo "---"

# Run Evaluation Loop 
for MODEL_ID in "${MODELS_TO_TEST[@]}"; do
    echo "======================================================="
    echo "Running Prometheus Judge on: $MODEL_ID"
    echo "======================================================="

    
    ANSWERS_FILE="${ANSWERS_DIR}/${MODEL_ID}.jsonl"
    
    OUT_FILE="data/mt_bench/model_judgment/prometheus-7b-v2.0_${MODEL_ID}.jsonl"
    RAW_FILE="data/mt_bench/model_judgment/prometheus-7b-v2.0_${MODEL_ID}_raw.jsonl"

    if [ ! -f "$ANSWERS_FILE" ]; then
        echo "[ERROR] Answer file not found, skipping: $ANSWERS_FILE"
        continue
    fi

    

    python3 "$PYTHON_SCRIPT" \
        --judge_model "$JUDGE_MODEL" \
        --questions_file "$QUESTIONS_FILE" \
        --model_answers_file "$ANSWERS_FILE" \
        --out_file "$OUT_FILE" \
        --first_n 80 \
        --dtype bfloat16 \
        --raw_dump "$RAW_FILE"

    EXIT_CODE=$?
    echo "Script finished for $MODEL_ID with code: $EXIT_CODE"
    ls -lh "$OUT_FILE" "$RAW_FILE"
    echo "---"
done

echo "=== ALL EVALUATIONS COMPLETE ==="
echo "Final results are in: data/mt_bench/model_judgment/"
ls -lh data/mt_bench/model_judgment/prometheus-7b-v2.0_*.jsonl