#!/bin/bash

#SBATCH --job-name=eval_extended_v4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1  # Or h100-47:1, depending on availability and your preference
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G # Adjust memory based on requirements, might need more
#SBATCH --time=24:00:00 # Adjust time based on expected runtime (could be very long)
#SBATCH --output=/tmp/%u_%j_eval_ext_v4.out
#SBATCH --error=/tmp/%u_%j_eval_ext_v4.err

echo "Evaluation Job Started: $(date)"
echo "Running on node: $(hostname)"

# 1. Temporary working directory 
JOB_TMP_DIR=$(mktemp -d /tmp/${USER}_job${SLURM_JOB_ID}_eval_ext_v4_XXXX)
echo "Created temporary directory: $JOB_TMP_DIR"

# 2. Copy script and potentially other files (e.g., results folder if needed locally)
cp run_extended_eval.py $JOB_TMP_DIR/ 

cd $JOB_TMP_DIR

# 3. Use absolute path to conda Python
PYTHON_EXEC="/home/e/e1415353/my_projects/miniconda3/envs/ml/bin/python"
echo "Using Python: $($PYTHON_EXEC --version)"

# 4. Log directory
LOG_DIR="/home/e/e1415353/my_projects/Assignment7Folder/Assignment7/" 
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_extended_${SLURM_JOB_ID}.log"

# 5. Run evaluation 
echo "Executing run_extended_eval.py..."
$PYTHON_EXEC run_extended_eval.py 2>&1 | tee "$LOG_FILE"



if [ -d "./evaluation_results_extended_v4" ]; then
    RESULT_DIR="$LOG_DIR/evaluation_results_extended_${SLURM_JOB_ID}"
    mkdir -p "$RESULT_DIR"
    cp -r ./evaluation_results_extended/* "$RESULT_DIR/"
    echo "Evaluation results copied to: $RESULT_DIR"
fi

# 7. Copy SLURM logs
cp "/tmp/${USER}_${SLURM_JOB_ID}_eval_ext.out" "$LOG_DIR/eval_extended_${SLURM_JOB_ID}.slurm.out"
cp "/tmp/${USER}_${SLURM_JOB_ID}_eval_ext.err" "$LOG_DIR/eval_extended_${SLURM_JOB_ID}.slurm.err"

# 8. Cleanup
rm -rf "$JOB_TMP_DIR"
echo "Evaluation Job Finished: $(date)"
