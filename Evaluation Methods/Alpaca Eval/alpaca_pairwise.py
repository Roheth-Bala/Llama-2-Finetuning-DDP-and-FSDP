#!/usr/bin/env python3
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
import argparse
import re 

class ModelWrapper:
    
    def __init__(self, model_path: str):
    
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print("Model loaded successfully!")
    
    def generate_answer(self, instruction: str, max_length: int = 512) -> str:
        prompt = f"""### Instruction:
{instruction}

### Response:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        
        response_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        
        response = response.strip()
        
        
        if "### Response:" in response:
            response = response.split("### Response:")[1].strip()

        return response

class PrometheusEvaluator:
    
    def __init__(self, judge_model: str = "prometheus-eval/prometheus-7b-v2.0"):
   
        print(f"Loading judge model: {judge_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(judge_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            judge_model,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print("Judge model loaded successfully!")
    
    def create_prompt(self, instruction: str, response_a: str, response_b: str) -> str:
        # Prompt structure for Prometheus pairwise evaluation
        prompt = f"""###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer, and a score rubric representing the evaluation criteria are given.
1. Write a detailed feedback that assesses the quality of the response based on the score rubric, not evaluating in general.
2. After writing the feedback, write a final score "Overall" by choosing from 1 to 5.
3. Refer to the score rubric for each score.

###Instruction:
{instruction}

###Response A:
{response_a}

###Response B:
{response_b}

###Score Rubric:
1: Response A is significantly worse than Response B
2: Response A is worse than Response B  
3: Response A is roughly equal to Response B
4: Response A is better than Response B
5: Response A is significantly better than Response B

###Feedback:"""
        return prompt
    
    def evaluate_pair(self, instruction: str, response_a: str, response_b: str) -> dict:
        prompt = self.create_prompt(instruction, response_a, response_b)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0, 
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
       
        response_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        feedback = response.strip()
        
        # Extract score
        
        scores = re.findall(r'\b([1-5])\b', feedback)
        score = int(scores[-1]) if scores else 3  # Default to tie
        
        return {
            'score': score,
            'feedback': feedback,
            'preference': 'A' if score >= 4 else 'B' if score <= 2 else 'Tie'
        }

def load_alpaca_eval_data(num_instructions: int = None):
    """Load AlpacaEval data from the downloaded file"""
   
    data_path = "/home/e/e1415353/.cache/huggingface/hub/datasets--tatsu-lab--alpaca_eval/snapshots/2edc6fad8be6b14ea7230aabfd08188da6b8b814/alpaca_eval.json"
    
    print(f"Loading AlpacaEval data from: {data_path}")
    if not os.path.exists(data_path):
        print(f"ERROR: AlpacaEval data not found at {data_path}")
        print("Please run `alpaca_eval` once to download the dataset or update the path.")
        exit(1)
        
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} total instructions")
    
    instructions = []
    baseline_answers = []
    
    
    data_to_use = data[:num_instructions] if num_instructions else data
    for item in data_to_use:
        instructions.append(item["instruction"])
        baseline_answers.append(item["output"])  # These are the text-davinci-003 answers
    
    print(f"✅ Loaded {len(instructions)} instructions for evaluation")
    return instructions, baseline_answers

def main():
    parser = argparse.ArgumentParser(description="Run AlpacaEval evaluation with real data")
    parser.add_argument("--your_model_path", type=str, required=True, 
                        help="Path to YOUR fine-tuned model")
    parser.add_argument("--output_dir", type=str, default="./alpacaeval_results", 
                        help="Output directory")
    parser.add_argument("--num_instructions", type=int, default=None, 
                        help="Number of instructions (default: all 805)")
    parser.add_argument("--judge_model", type=str, default="prometheus-eval/prometheus-7b-v2.0", 
                        help="Judge model")
    
    args = parser.parse_args()
    
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ALPACAEVAL PAIRWISE EVALUATION")
    print("=" * 60)
    print(f"Your Model (Response A): {args.your_model_path}")
    print(f"Baseline Model (Response B): text-davinci-003 (from alpaca_eval.json)")
    print(f"Judge Model: {args.judge_model}")
    if args.num_instructions:
        print(f"Number of instructions: {args.num_instructions}")
    else:
        print(f"Number of instructions: ALL 805")
    print("=" * 60)
    
    # Step 1: Load AlpacaEval data
    instructions, baseline_answers = load_alpaca_eval_data(args.num_instructions)
    
    # Step 2: Generate answers from YOUR model
    print(f"\n1. Generating answers from YOUR model for {len(instructions)} instructions...")
    your_model = ModelWrapper(args.your_model_path)
    your_answers = []
    
    for instruction in tqdm(instructions, desc="Your model generating"):
        answer = your_model.generate_answer(instruction)
        your_answers.append(answer)
    
    # Unload your model to save VRAM for the judge model
    del your_model
    torch.cuda.empty_cache()
    print("Unloaded your model to free VRAM.")

    # Step 3: Initialize evaluator
    print("\n2. Loading judge model...")
    evaluator = PrometheusEvaluator(args.judge_model)
    
    # Step 4: Run pairwise evaluation
    print(f"\n3. Running pairwise evaluation on {len(instructions)} examples...")
    results = []
    
    for i, (instruction, your_ans, baseline_ans) in enumerate(tqdm(
        zip(instructions, your_answers, baseline_answers), 
        desc="Evaluating pairs",
        total=len(instructions)
    )):
        try:
            # Response A is your model, Response B is the baseline
            eval_result = evaluator.evaluate_pair(instruction, your_ans, baseline_ans)
            results.append({
                'instruction': instruction,
                'your_answer (A)': your_ans,
                'baseline_answer (B)': baseline_ans,
                'score (A vs B)': eval_result['score'],
                'preference': eval_result['preference'],
                'feedback': eval_result['feedback']
            })
        except Exception as e:
            print(f"Error evaluating pair {i}: {e}")
            results.append({
                'instruction': instruction,
                'your_answer (A)': your_ans,
                'baseline_answer (B)': baseline_ans,
                'score (A vs B)': 3, # Default to tie on error
                'preference': 'Tie',
                'feedback': f'Error: {str(e)}'
            })
    
    # Step 5: Calculate metrics
    preferences = [r['preference'] for r in results]
    scores = [r['score (A vs B)'] for r in results]
    
    win_rate = sum(1 for p in preferences if p == 'A') / len(preferences) * 100
    loss_rate = sum(1 for p in preferences if p == 'B') / len(preferences) * 100
    tie_rate = sum(1 for p in preferences if p == 'Tie') / len(preferences) * 100
    avg_score = sum(scores) / len(scores)
    
    # Step 6: Save results
    print("\n4. Saving results...")
    
    # Detailed results
    df = pd.DataFrame(results)
    df.to_csv(f"{args.output_dir}/detailed_results.csv", index=False)
    
    
    output_for_leaderboard = []
    for instr, ans in zip(instructions, your_answers):
        output_for_leaderboard.append({
            "instruction": instr,
            "output": ans,
            "generator": args.your_model_path
        })
    with open(f"{args.output_dir}/model_outputs.json", "w") as f:
        json.dump(output_for_leaderboard, f, indent=2)
    

    summary = {
        "model_name": args.your_model_path,
        "win_rate_vs_baseline": win_rate,
        "loss_rate_vs_baseline": loss_rate,
        "tie_rate_vs_baseline": tie_rate,
        "average_score": avg_score,
        "total_comparisons": len(results),
        "baseline_model": "text-davinci-003",
        "judge_model": args.judge_model,
        "dataset": f"AlpacaEval ({len(instructions)} instructions)"
    }
    
    with open(f"{args.output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    
    print("\n" + "=" * 60)
    print("ALPACAEVAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"Your Model ('A'): {args.your_model_path}")
    print(f"Baseline ('B'): text-davinci-003")
    print(f"Your Model Win Rate: {win_rate:.2f}%")
    print(f"Your Model Loss Rate: {loss_rate:.2f}%")
    print(f"Tie Rate: {tie_rate:.2f}%")
    print(f"Average Score (1-5): {avg_score:.2f}")
    print(f"Total Evaluations: {len(results)}")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
