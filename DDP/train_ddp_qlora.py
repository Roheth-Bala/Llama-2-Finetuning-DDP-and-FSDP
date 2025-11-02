#!/usr/bin/env python3
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
import argparse
import torch.distributed as dist


logger = logging.getLogger(__name__)



def setup_distributed():
    
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
        
        if 'SLURM_JOB_NODELIST' in os.environ:
             os.environ['MASTER_ADDR'] = 'localhost'
        else:
             os.environ['MASTER_ADDR'] = 'localhost' 
             
        os.environ['MASTER_PORT'] = '12355' 
        
        
        torch.cuda.set_device(local_rank)
        
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        print(f"[Rank {rank}] Running on SLURM cluster. world_size: {world_size}, local_rank: {local_rank}")
        
        return rank, world_size, local_rank
    else:
        print("Not running on SLURM cluster, running as single process.")
        return 0, 1, 0 

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def format_dolly(sample):
    """Formatting a Dolly sample into a single text string."""
    instr = sample["instruction"]
    ctx = sample["context"]
    resp = sample["response"]
    # Filtering out samples that have empty responses
    if not resp or not resp.strip():
        return None
    if ctx and ctx.strip():
        text_output = f"### Instruction:\n{instr}\n\n### Context:\n{ctx}\n\n### Response:\n{resp}"
    else:
        text_output = f"### Instruction:\n{instr}\n\n### Response:\n{resp}"
    return {"text": text_output}

def main():
    parser = argparse.ArgumentParser(description="DDP QLoRA Fine-tuning")
    # Having the Model/Data arguments
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset_name", type=str, default="databricks/databricks-dolly-15k")
    parser.add_argument("--output_dir", type=str, default="./results/dolly_lora_llama2_7b_ddp")
    # Having the Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--logging_steps", type=int, default=100)
    # Having the LoRA arguments
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    args = parser.parse_args()

    rank, world_size, local_rank = 0, 1, 0
    
    try:
        
        rank, world_size, local_rank = setup_distributed()
        
       
        logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING)
        
        
        set_seed(42)

        
        if rank == 0:
            logger.info("Loading Dolly-15k dataset...")
        
       
        dataset = load_dataset(args.dataset_name)

        
        formatted_dataset = dataset["train"].map(format_dolly).filter(lambda x: x is not None)

        
        train_val_split = formatted_dataset.train_test_split(test_size=0.2, seed=42)
        test_val_split = train_val_split['test'].train_test_split(test_size=0.5, seed=42)
        final_dataset = {
            'train': train_val_split['train'],
            'validation': test_val_split['test'],
            'test': test_val_split['train']
        }
        
        if rank == 0:
            logger.info(f"Dataset splits created. Train: {len(final_dataset['train'])}, Val: {len(final_dataset['validation'])}, Test: {len(final_dataset['test'])}")

       
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map={"": local_rank}, 
        )

        if rank == 0:
            logger.info(f"✓ Model {args.model_id} loaded in 4-bit (QLoRA) on Rank 0")

        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

      
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        
        if rank == 0:
            logger.info("Model prepared for k-bit training.")

        # LoRA Config 
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )

        # Applying PEFT 
        model = get_peft_model(model, peft_config)
        
        if rank == 0:
            model.print_trainable_parameters()
            logger.info("Model wrapped with PEFT LoRA.")

        # SFTConfig 
        sft_config = SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_train_batch_size, # Use same for eval
            gradient_accumulation_steps=args.gradient_accumulation_steps, 
            
            learning_rate=args.learning_rate, 
            
            weight_decay=0.0,
            warmup_ratio=0.03,
            max_steps=-1,
            
            
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            eval_strategy="steps",
            eval_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.logging_steps,
            
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            report_to="none", 
            fp16=True, 
            bf16=False,
            
            group_by_length=True, 
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            lr_scheduler_type="cosine",
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            packing=False,
            
            
            ddp_find_unused_parameters=False,
            ddp_backend="nccl",
            ddp_timeout=1800,
        )

        # SFTTrainer 
        
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=final_dataset["train"],
            eval_dataset=final_dataset["validation"],
            processing_class=tokenizer,
        )

        if rank == 0:
            logger.info("\nStarting QLoRA DDP fine-tuning...")
        
        start_time = time.time()
        
        
        trainer.train()

        end_time = time.time()
        training_duration_minutes = (end_time - start_time) / 60
        
        if rank == 0:
            logger.info(f"Training finished in: {training_duration_minutes:.2f} minutes")

        
        
        if rank == 0:
            logger.info("Rank 0 starting final save and plot...")
            
           
            final_save_dir = os.path.join(sft_config.output_dir, "final")
            trainer.save_model(final_save_dir)
            tokenizer.save_pretrained(final_save_dir)
            logger.info(f"Final model and tokenizer saved to: {final_save_dir}")

           
            log_history = trainer.state.log_history
            log_save_path = os.path.join(sft_config.output_dir, "training_logs.json")
            with open(log_save_path, "w") as f:
                json.dump(log_history, f, indent=2)
            logger.info(f"Training logs saved to: {log_save_path}")

            
            train_logs = [entry for entry in log_history if "loss" in entry]
            eval_logs = [entry for entry in log_history if "eval_loss" in entry]

            if train_logs and eval_logs:
                train_steps = [entry["step"] for entry in train_logs]
                train_loss = [entry["loss"]for entry in train_logs]
                train_lr = [entry.get("learning_rate", 0) for entry in train_logs] 

                eval_steps = [entry["step"] for entry in eval_logs]
                eval_loss = [entry["eval_loss"] for entry in eval_logs]
                eval_perplexity = [np.exp(loss) for loss in eval_loss]

                
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
                fig.suptitle('Training & Validation Summary (Rank 0)', fontsize=16, y=1.02)
                plt.subplots_adjust(hspace=0.1)

                
                ax1.plot(train_steps, train_loss, label="Train Loss", marker='o', markersize=2)
                ax1.plot(eval_steps, eval_loss, label="Eval Loss", marker='s', markersize=3)
                ax1.set_ylabel("Loss")
                ax1.set_title("Loss (Train vs. Eval)")
                ax1.legend()
                ax1.grid(True)
                ax1.tick_params(labelbottom=False)

               
                ax2.plot(eval_steps, eval_perplexity, label="Eval Perplexity", color='green', marker='s', markersize=3)
                ax2.set_ylabel("Perplexity")
                ax2.set_title("Validation Perplexity")
                ax2.legend()
                ax2.grid(True)
                ax2.tick_params(labelbottom=False)

                
                ax3.plot(train_steps, train_lr, label="Learning Rate", color='red', marker='o', markersize=2)
                ax3.set_xlabel("Step")
                ax3.set_ylabel("Learning Rate")
                ax3.set_title("Learning Rate Schedule")
                ax3.legend()
                ax3.grid(True)

                
                plot_path = os.path.join(sft_config.output_dir, "training_summary_plots.png")
                plt.savefig(plot_path, bbox_inches='tight')
                logger.info(f"All training plots saved to: {plot_path}")
            else:
                logger.warning("Could not find sufficient logs to create plots.")

            
            # --- 11. Log Hardware Info ---
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            logger.info(f"GPU used (Rank 0): {gpu_name}")
            logger.info(f"Peak VRAM allocated (Rank 0): {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
            logger.info("Script completed successfully on Rank 0.")

    except Exception as e:
        logger.error(f"[Rank {rank}] Error during training: {e}", exc_info=True)
    
    finally:
        
        cleanup_distributed()
        print(f"[Rank {rank}] DDP process group destroyed. Exiting.")

if __name__ == "__main__":
    main()