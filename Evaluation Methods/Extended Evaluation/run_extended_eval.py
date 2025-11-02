# run_extended_eval_corrected_v4.py
import os
import tempfile
import torch
import numpy as np
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from lm_eval import evaluator, tasks

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Path to the BASE MODEL (e.g., Llama-2-7b-hf)
BASE_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
# Path to YOUR FINETUNED PEFT ADAPTER (the 'final' directory from your training run)
# Updated path based on your listing
FT_ADAPTER_PATH = "/home/e/e1415353/my_projects/Assignment7Folder/Assignment7/results_241919/dolly_lora_llama2_7b_tuned_v1/final"
# Output directory for evaluation results
EVAL_OUTPUT_DIR = "./evaluation_results_extended_v4"
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

def make_json_serializable(obj):
    """
    Recursively makes an object JSON serializable.
    Handles numpy types, torch tensors, and other common types.
    """
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        # Convert numpy scalars to Python scalars
        return obj.item()
    elif isinstance(obj, np.ndarray):
        # Convert numpy arrays to Python lists
        return obj.tolist()
    elif isinstance(obj, torch.dtype):
        # Convert torch dtype to string representation
        return str(obj)
    elif isinstance(obj, np.dtype):
        # Convert numpy dtype to string representation
        return str(obj)
    # Add more conditions if other types are encountered
    else:
        # Return the object as is if it's already serializable (str, int, float, bool, None)
        return obj

def main():
    try:
        # --- Load Base Model and Tokenizer ---
        logger.info(f"Loading base model: {BASE_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            logger.info("Adding special pad token '<pad>' to tokenizer.")
            tokenizer.add_special_tokens({"pad_token": "<pad>"}) # Add pad token if missing
        # Store the pad token id *before* potentially resizing the model
        pad_token_id = tokenizer.pad_token_id
        logger.info(f"Pad token ID: {pad_token_id}")

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.bfloat16, # Match training dtype
            device_map="auto", # Use available GPU(s)
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # --- Apply the fine-tuned PEFT adapter ---
        logger.info(f"Applying PEFT adapter from: {FT_ADAPTER_PATH}")
        model_with_adapter = PeftModel.from_pretrained(base_model, FT_ADAPTER_PATH)

        # --- Merge the adapter weights into the base model ---
        logger.info("Merging PEFT adapter weights into base model...")
        merged_model = model_with_adapter.merge_and_unload()
        logger.info("Adapter merged successfully.")

        # --- Ensure model config matches tokenizer ---
        # Set pad token id in model config BEFORE saving
        if merged_model.config.pad_token_id != pad_token_id:
            logger.info(f"Updating model config's pad_token_id from {merged_model.config.pad_token_id} to {pad_token_id}")
            merged_model.config.pad_token_id = pad_token_id

        # Check if the pad_token_id is within the model's vocab size
        vocab_size = merged_model.get_input_embeddings().num_embeddings
        logger.info(f"Model's vocabulary size: {vocab_size}")
        if pad_token_id >= vocab_size:
            logger.warning(f"Pad token ID ({pad_token_id}) is >= model's vocab size ({vocab_size}). Resizing model embeddings.")
            # Resize token embeddings to match tokenizer
            merged_model.resize_token_embeddings(len(tokenizer))
            # Re-fetch the vocab size after resizing
            vocab_size_after_resize = merged_model.get_input_embeddings().num_embeddings
            logger.info(f"Model's vocabulary size after resize: {vocab_size_after_resize}")
            # Ensure pad token id is within the new vocab size
            if pad_token_id >= vocab_size_after_resize:
                 # This should ideally not happen if len(tokenizer) is correct, but double-check
                 logger.error(f"Pad token ID ({pad_token_id}) still >= vocab size ({vocab_size_after_resize}) after resize.")
                 raise ValueError("Pad token ID is invalid even after resizing embeddings.")

        # Set pad token id again after potential resize
        merged_model.config.pad_token_id = pad_token_id
        tokenizer.padding_side = "right" # Ensure padding side is correct

        # --- Prepare Model for Evaluation ---
        merged_model.eval() # Set to evaluation mode
        merged_model.tie_weights() # Tie weights if needed (often done automatically by PEFT)

        # --- Define Tasks for Evaluation ---
        # Example tasks from the bonus list. You can add more or modify as needed.
        # Check available tasks with: print(tasks.list_all_tasks())
        task_list = [
            "mmlu", # Massive Multitask Language Understanding
            "gsm8k", # Grade School Math 8K (requires 'gsm8k' in the list, often needs specific prompting)
            "truthfulqa_mc2", # TruthfulQA (Multiple Choice version 2)
            "arc_easy", # AI2 Reasoning Challenge (Easy)
            "arc_challenge", # AI2 Reasoning Challenge (Challenge)
            # Note: BBH (Big-Bench Hard) is available as "bigbench" but requires specific setup or might be very slow.
            # "bigbench_qa" # Example for a subset, check lm_eval docs for details
        ]

        # --- Create a temporary directory to save the merged model ---
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Saving merged model temporarily to: {temp_dir}")
            # Save merged model and tokenizer to the temporary directory
            merged_model.save_pretrained(temp_dir, safe_serialization=True)
            tokenizer.save_pretrained(temp_dir)

            # --- Run Evaluation using the path to the temporary directory ---
            logger.info(f"Starting evaluation on tasks: {task_list}")
            # Use 'hf-auto' and pass the path to the temporary directory containing the merged model
            results = evaluator.simple_evaluate(
                model="hf-auto", # Use 'hf-auto' model name
                model_args=f"pretrained={temp_dir},trust_remote_code=True", # Pass path to temp dir and trust remote code
                tasks=task_list,
                num_fewshot=0, # Number of few-shot examples (adjust based on task recommendations)
                batch_size=1, # Adjust batch size based on your GPU memory (larger is faster but uses more memory)
                device="cuda", # Specify device
                limit=None, # Set to a small number (e.g., 10) for testing, None for full evaluation
                # gen_kwargs={"max_length": 2048}, # Example generation kwargs if needed for specific tasks
            )

        # --- Make Results JSON Serializable and Save ---
        # The 'results' dict from lm_eval might contain non-serializable objects like numpy dtypes
        serializable_results = make_json_serializable(results)
        results_file = os.path.join(EVAL_OUTPUT_DIR, "extended_eval_results_v4.json")
        with open(results_file, "w") as f:
            import json
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Evaluation results saved to: {results_file}")

        # --- Print Summary ---
        logger.info("\n--- Evaluation Summary ---")
        for task_name, task_results in serializable_results["results"].items():
            logger.info(f"Task: {task_name}")
            for metric_name, metric_value in task_results.items():
                if isinstance(metric_value, float):
                    logger.info(f"  {metric_name}: {metric_value:.4f}")
                else:
                    logger.info(f"  {metric_name}: {metric_value}")
            logger.info("-" * 20)

        logger.info("\nExtended evaluation completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc() # Print the full traceback for debugging
        raise # Re-raise to ensure the job script captures the failure

if __name__ == "__main__":
    main()