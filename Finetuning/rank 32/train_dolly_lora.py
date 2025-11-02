import os
import json
import torch
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login


logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)


set_seed(42)

# Load and Preprocess Dataset 
logger.info("Loading Dolly-15k dataset...")
dataset = load_dataset("databricks/databricks-dolly-15k")

def format_dolly(sample):
    """Formats a Dolly sample into a single text string."""
    instr = sample["instruction"]
    ctx = sample["context"]
    resp = sample["response"]
    # Filter out samples with empty responses
    if not resp or not resp.strip():
        return None
    if ctx and ctx.strip():
        text_output = f"### Instruction:\n{instr}\n\n### Context:\n{ctx}\n\n### Response:\n{resp}"
    else:
        text_output = f"### Instruction:\n{instr}\n\n### Response:\n{resp}"
    return {"text": text_output}


formatted_dataset = dataset["train"].map(format_dolly).filter(lambda x: x is not None)


train_val_split = formatted_dataset.train_test_split(test_size=0.2, seed=42)
test_val_split = train_val_split['test'].train_test_split(test_size=0.5, seed=42)
final_dataset = {
    'train': train_val_split['train'],
    'validation': test_val_split['test'],
    'test': test_val_split['train']
}

logger.info(f"Dataset splits created. Train: {len(final_dataset['train'])}, Val: {len(final_dataset['validation'])}, Test: {len(final_dataset['test'])}")

# Model & Tokenizer 
model_id = "meta-llama/Llama-2-7b-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

logger.info(f"✓ Model {model_id} loaded in 4-bit (QLoRA)")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#  Prepare Model for k-bit Training 
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
logger.info("Model prepared for k-bit training.")

# LoRA Config
peft_config = LoraConfig(
    r=32,          
    lora_alpha=64,   
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

# Apply PEFT 
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
logger.info("Model wrapped with PEFT LoRA.")

# SFTConfig 
sft_config = SFTConfig(
    output_dir="./results/dolly_lora_llama2_7b_tuned_v1",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    
    learning_rate=1e-4,
    
    weight_decay=0.0,
    warmup_ratio=0.03,
    max_steps=-1,
    
   
    logging_strategy="steps",
    logging_steps=100,      
    eval_strategy="steps",
    eval_steps=100,        
    save_strategy="steps",
    save_steps=100,        
    

    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    report_to="none",
    bf16=True,
    dataloader_num_workers=2,
    remove_unused_columns=False,
    
    
    group_by_length=True,
    
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    dataset_text_field="text",
    max_seq_length=1024,
    packing=False,
)

# SFTTrainer 
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=final_dataset["train"],
    eval_dataset=final_dataset["validation"],
    processing_class=tokenizer,
)

# Start Training 
logger.info("\nStarting QLoRA fine-tuning...")
start_time = time.time()

trainer.train()

end_time = time.time()
training_duration_minutes = (end_time - start_time) / 60
logger.info(f"Training finished in: {training_duration_minutes:.2f} minutes")

# Save Final Model 
final_save_dir = os.path.join(sft_config.output_dir, "final")
trainer.save_model(final_save_dir)
tokenizer.save_pretrained(final_save_dir)
logger.info(f"Final model and tokenizer saved to: {final_save_dir}")

# Plot Training Results
log_history = trainer.state.log_history
log_save_path = os.path.join(sft_config.output_dir, "training_logs.json")
with open(log_save_path, "w") as f:
    json.dump(log_history, f, indent=2)
logger.info(f"Training logs saved to: {log_save_path}")


train_logs = [entry for entry in log_history if "loss" in entry]
eval_logs = [entry for entry in log_history if "eval_loss" in entry]

train_steps = [entry["step"] for entry in train_logs]
train_loss = [entry["loss"] for entry in train_logs]

train_lr = [entry.get("learning_rate", 0) for entry in train_logs]

eval_steps = [entry["step"] for entry in eval_logs]
eval_loss = [entry["eval_loss"] for entry in eval_logs]

eval_perplexity = [np.exp(loss) for loss in eval_loss]


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
fig.suptitle('Training & Validation Summary', fontsize=16, y=1.02)
plt.subplots_adjust(hspace=0.1)

# Plot 1: Loss
ax1.plot(train_steps, train_loss, label="Train Loss", marker='o', markersize=2)
if eval_steps:
    ax1.plot(eval_steps, eval_loss, label="Eval Loss", marker='s', markersize=3)
ax1.set_ylabel("Loss")
ax1.set_title("Loss (Train vs. Eval)")
ax1.legend()
ax1.grid(True)
ax1.tick_params(labelbottom=False) 

# Plot 2: Perplexity
if eval_perplexity:
    ax2.plot(eval_steps, eval_perplexity, label="Eval Perplexity", color='green', marker='s', markersize=3)
ax2.set_ylabel("Perplexity")
ax2.set_title("Validation Perplexity")
ax2.legend()
ax2.grid(True)
ax2.tick_params(labelbottom=False) 

# Plot 3: Learning Rate
ax3.plot(train_steps, train_lr, label="Learning Rate", color='red', marker='o', markersize=2)
ax3.set_xlabel("Step")
ax3.set_ylabel("Learning Rate")
ax3.set_title("Learning Rate Schedule")
ax3.legend()
ax3.grid(True)

# Save the combined plot
plot_path = os.path.join(sft_config.output_dir, "training_summary_plots.png")
plt.savefig(plot_path, bbox_inches='tight')
logger.info(f"All training plots saved to: {plot_path}")

# Log Hardware Info
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
logger.info(f"GPU used: {gpu_name}")
logger.info(f"Estimated VRAM usage (Peak allocated MB): {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

logger.info("Script completed successfully.")