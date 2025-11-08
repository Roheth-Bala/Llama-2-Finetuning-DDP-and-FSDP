# Assignment 7: Comparative Analysis of LLaMA-2-7B Fine-Tuning
**Author: Roheth Balamurugan (e1415353)**

This repository contains the code and analysis for a comparative study on fine-tuning the `meta-llama/Llama-2-7b-hf` model. This project explores two main axes of research:
1.  **LoRA Rank Comparison:** Analyzing the impact of LoRA adapter capacity (Rank 8 vs. Rank 32) on a single GPU.
2.  **Distributed Strategy Comparison:** Benchmarking Distributed Data Parallel (DDP) with QLoRA against Fully Sharded Data Parallel (FSDP) with `bf16` precision.

The final, detailed findings are compiled in the NNDL_Assignment7_Report.pdf.
---

📈 Outputs and Results
All generated outputs, including result curves, are located in their respective folders.

🚀 Running the Training
The steps to replicate the training process are provided in the .sh scripts within each folder.

Important Environment Note:

These scripts are sbatch files configured for the NUS SOC (School of Computing) Cluster.

You will need to adapt the commands within these .sh files to match your specific hardware or cluster environment.

⚙️ Configuration
The adapter configuration file (adapter_config.json and safetensors) is located in the Finetuning/rank_8/ directory.

## 🎯 Project Objective

Fine-tune the **meta-llama/Llama-2-7b** model using **Parameter-Efficient Fine-Tuning (PEFT, via QLoRA)** on the **databricks/databricks-dolly-15k** dataset. The goal is to demonstrate loss convergence and measure performance improvements on **AlpacaEval 2** and **MT-Bench**, while also comparing the trade-offs of different LoRA ranks and distributed training strategies.

---

## 📂 Repository Structure

```
Llama-2-Finetuning-DDP-and-FSDP/
│
├── Finetuning/
│   ├── rank_8/                 # Files for Rank 8 Single-GPU training
│   └── rank_32/                # Files for Rank 32 Single-GPU training
│       └── train_dolly_lora.py
│
├── DDP/
│   └── train_ddp_qlora.py      # DDP + QLoRA (multi-GPU) training script
│
├── FSDP/
│   └── train_fsdp_bf16.py      # FSDP + bf16 (multi-GPU) training script
│
├── Evaluation Methods/
│   ├── Alpaca Eval/
│   │   └── alpaca_pairwise.py    # AlpacaEval 2 script
│   ├── MT Bench/
│   │   ├── run_mt_bench_gen.sh   # Script to generate MT-Bench answers
│   │   └── judge_mtbench_local.py# Script to judge answers w/ Prometheus
│   └── Extended Evaluation/
│       └── run_extended_eval.py  # MMLU, ARC, GSM8K eval script
│
├── results/
│   ├── loss_curve_rank_8.png
│   ├── loss_curve_rank_32.png
│   ├── validation_perplexity_rank_32.png
│   ├── learning_rate_schedule_rank_32.png
│   ├── training_summary_plots_ddp.png
│   └── fsdp_bf16_run_plot.png
│
├── assignment7_report.tex      # LaTeX source for the final report
├── assignment7_report.cls      # LaTeX class file
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🏁 Getting Started

### 1. Setup Environment

First, clone the repository:
```bash
git clone [https://github.com/Roheth-Bala/Llama-2-Finetuning-DDP-and-FSDP.git](https://github.com/Roheth-Bala/Llama-2-Finetuning-DDP-and-FSDP.git)
cd Llama-2-Finetuning-DDP-and-FSDP
```

It is highly recommended to use a virtual environment (e.g., conda or venv):

```bash
conda create -n llama-peft python=3.10
conda activate llama-peft
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Hugging Face Authentication

You will need a Hugging Face token to access the LLaMA-2 models.
```bash
huggingface-cli login
```
Paste your Hugging Face token when prompted.

---

## 🚀 Running the Experiments

All training scripts are designed to be run on a system with one or more NVIDIA GPUs.

### 1. Single-GPU Fine-Tuning (Rank 32)

This script trains a QLoRA model with Rank 32 adapters on a single GPU.

```bash
python Finetuning/rank_32/train_dolly_lora.py
```
*(Note: The script for Rank 8 training follows the same command structure.)*

### 2. Distributed Fine-Tuning (DDP)


You will need to adapt the commands within these .sh files to match your specific hardware or cluster environment.

### 3. Distributed Fine-Tuning (FSDP)


You will need to adapt the commands within these .sh files to match your specific hardware or cluster environment.


---

## 📊 Running the Evaluation

After training, you can evaluate the models on the standard benchmarks.

### 1. AlpacaEval 2

This script evaluates a fine-tuned model against a baseline using `prometheus-7b-v2.0` as the judge.

```bash
python "Evaluation Methods/Alpaca Eval/alpaca_pairwise.py" \
    --model_name "path/to/your/finetuned/model" \
    --baseline_model "meta-llama/Llama-2-7b-hf" \
    --judge_model "prometheus-eval/prometheus-7b-v2.0"
```

### 2. MT-Bench

MT-Bench is a two-step process: answer generation and judging.

```bash
# 1. Generate model answers using the fastchat script
#    (You may need to edit paths inside the .sh file first)
bash "Evaluation Methods/MT Bench/run_mt_bench_gen.sh"

# 2. Judge the generated answers locally using Prometheus
python "Evaluation Methods/MT Bench/judge_mtbench_local.py" \
    --model_answers_file "path/to/generated_answers.jsonl" \
    --judge_model "prometheus-eval/prometheus-7b-v2.0" \
    --out_file "path/to/results.jsonl"
```

### 3. Extended Benchmarks (MMLU, ARC, GSM8K)

This script uses `lm-eval` to run the extended academic benchmarks.

```bash
python "Evaluation Methods/Extended Evaluation/run_extended_eval.py" \
    --model_path "path/to/your/finetuned/model" \
    --output_dir "path/to/evaluation/results"
```

---

## 📈 Summary of Results

### Analysis 1: LoRA Rank Comparison (Single-GPU)

This analysis shows the trade-off between adapter capacity and performance.

**Instruction-Following (AlpacaEval 2)**
| Experiment | Model | Win Rate (%) | Loss Rate (%) | Tie Rate (%) | Avg. Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Rank 8** | Llama-2-7B (Base) | 79.00 | 14.75 | 6.25 | 4.09 |
| | **Llama-2-7B + LoRA (R=8)** | **85.00** | **9.88** | **5.13** | **4.21** |
| **Rank 32** | Llama-2-7B (Base) | 79.75 | 14.37 | 5.88 | 4.11 |
| | **Llama-2-7B + LoRA (R=32)** | **83.75** | **10.75** | **5.50** | **4.19** |

**Conversational Ability (MT-Bench)**
| Model | Mean Score (out of 10) | 95\% CI |
| :--- | :---: | :---: |
| Llama-2-7B (Base) | 5.550 | [5.055, 6.045] |
| **Llama-2-7B + LoRA (R=8)** | **6.037** | **[5.618, 6.457]** |
| **Llama-2-7B + LoRA (R=32)** | **6.037** | **[5.618, 6.457]** |

**General Knowledge & Reasoning (Extended Benchmarks)**
| Benchmark | Metric | Base Model | Tuned (R=8) | Tuned (R=32) |
| :--- | :---: | :---: | :---: | :---: |
| MMLU | acc | 0.4084 | 0.4127 | **0.4231** |
| ARC Challenge | acc_norm | 0.4616 | 0.4454 | **0.4846** |
| ARC Easy | acc_norm | 0.7458 | 0.7399 | **0.7715** |
| GSM8K | flexible_em | **0.0584** | 0.0546 | 0.0478 |

**Insight:** Rank 8 is slightly better for the specific instruction-following task (AlpacaEval), but **Rank 32 is significantly better at retaining general knowledge** (MMLU, ARC), suggesting it suffers less from catastrophic forgetting.

### Analysis 2: Distributed Strategy Comparison (Multi-GPU)

This analysis shows the trade-off between training speed and final model precision.

| Metric | DDP (4-bit QLoRA) | FSDP (bf16) |
| :--- | :--- | :--- |
| Hardware | 2x NVIDIA H100 | 2x NVIDIA H100 |
| **Total Training Time** | **~48 min** | ~146 min |
| **Speedup** | **~3.05x Faster** | 1x (Baseline) |
| Final Train Loss | 1.268 | 1.255 |
| **Final Eval Loss** | ~1.31 | **~1.29** |
| Train Samples/sec | **12.519** | 4.104 |
| Total Steps | 564 | 564 |

**Insight:** DDP with QLoRA provides a massive **3x speedup** with a negligible impact on model quality. FSDP (`bf16`) is much slower but produces a marginally better model (lower validation loss) by training in full precision.
