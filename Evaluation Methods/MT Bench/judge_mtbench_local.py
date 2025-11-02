import argparse, json, os, re, math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm


PROMETHEUS_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
A perfect answer would be comprehensive, accurate, directly address the question, provide clear explanations with appropriate depth, and demonstrate expert-level understanding of the topic.

###Score Rubrics:
[Is the response helpful, accurate, and appropriate in addressing the given instruction?]
Score 1: The response is completely unhelpful, inaccurate, or inappropriate.
Score 2: The response attempts to address the instruction but contains significant errors or lacks helpfulness.
Score 3: The response is partially helpful and generally accurate but may lack depth or clarity.
Score 4: The response is helpful, accurate, and appropriate with minor issues.
Score 5: The response is exceptionally helpful, accurate, appropriate, and comprehensive.

###Feedback:"""

def extract_prometheus_score(text: str):
    """
    Extract score from Prometheus output format: "Feedback: ... [RESULT] X"
    Converts 1-5 scale to 1-10 scale (multiply by 2)
    """
    s = text.strip()
    print(f"[DEBUG] Raw output length: {len(s)}")
    print(f"[DEBUG] Output preview: {s[:500]}")
    
    
    m = re.search(r'\[RESULT\]\s*(\d+)', s, re.IGNORECASE)
    if m:
        try:
            score_5 = int(m.group(1))
            if 1 <= score_5 <= 5:
                # Convert to 10-point scale
                score_10 = score_5 * 2.0
                
                feedback_match = re.search(r'Feedback:\s*(.*?)\s*\[RESULT\]', s, re.DOTALL | re.IGNORECASE)
                feedback = feedback_match.group(1).strip() if feedback_match else ""
                print(f"[DEBUG] ✓ Parsed Prometheus score: {score_5}/5 -> {score_10}/10")
                return score_10, feedback[:500]
        except Exception as e:
            print(f"[DEBUG] Prometheus parse error: {e}")
    
    
    m = re.search(r'Score\s*[:=]?\s*(\d+)', s, re.IGNORECASE)
    if m:
        try:
            score = int(m.group(1))
            if 1 <= score <= 5:
                score_10 = score * 2.0
                print(f"[DEBUG] ✓ Parsed fallback score: {score}/5 -> {score_10}/10")
                return score_10, s[:500]
            elif 1 <= score <= 10:
                print(f"[DEBUG] ✓ Parsed fallback score: {score}/10")
                return float(score), s[:500]
        except Exception as e:
            print(f"[DEBUG] Fallback parse error: {e}")
    
    print("[DEBUG] ✗ Could not parse any score!")
    return None, s[-400:]

def build_prometheus_input(tok, instruction, response):
    """Build input for Prometheus model."""
    prompt = PROMETHEUS_PROMPT.format(instruction=instruction, response=response)
    print(f"[DEBUG] Prompt length: {len(prompt)} chars")
    
    # Prometheus models don't use chat templates, just direct prompts so we,
    # increased max_length from 2048 to 8192 to avoid truncating long answers.
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=8192)
    
    
    if inputs['input_ids'].shape[1] == 8192:
        print("[WARN] Input was truncated to 8192 tokens. This might be an issue.")
        
    print(f"[DEBUG] Input tokens: {inputs['input_ids'].shape[1]}")
    return inputs

def load_questions(path):
    """Load MT-Bench questions."""
    qs = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            o = json.loads(ln)
            qid = o["question_id"]
            turns = list(o.get("turns") or [])
            if len(turns) < 2:
                turns += [""] * (2 - len(turns))
            qs[qid] = (turns[0], turns[1])
    print(f"[DEBUG] Loaded {len(qs)} questions")
    return qs

def load_model_answers(path):
    """Load model answers. Supports multiple formats."""
    from collections import defaultdict
    ans = defaultdict(lambda: ["",""])

    def _txt(x):
        return x.get("text") or x.get("response") or x.get("content") or ""

    line_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            line_count += 1
            try:
                o = json.loads(ln)
                qid = o.get("question_id")
                if qid is None:
                    print(f"[DEBUG] Line {i}: No question_id found")
                    continue
                
                # Format A: 'choices' list with 'turns' array inside
                if isinstance(o.get("choices"), list) and len(o["choices"]) > 0:
                    choice = o["choices"][0]  
                    
                    # Checking if turns array exists inside choice
                    if isinstance(choice.get("turns"), list):
                        turns = choice["turns"]
                        if len(turns) >= 1:
                            ans[qid][0] = turns[0]  # Turn 1
                        if len(turns) >= 2:
                            ans[qid][1] = turns[1]  # Turn 2
                    
                    else:
                        for it in o["choices"]:
                            t = it.get("turn") or it.get("turn_id")
                            if t in (1,2):
                                ans[qid][t-1] = _txt(it)
                
                # Format B: direct turn in object
                elif "turn" in o or "turn_id" in o:
                    t = o.get("turn") or o.get("turn_id")
                    if t in (1,2):
                        ans[qid][t-1] = _txt(o)
                
                # Format C: direct text fields
                else:
                    txt = _txt(o)
                    if txt and not ans[qid][0]:
                        ans[qid][0] = txt
                    elif txt:
                        ans[qid][1] = txt
                        
            except Exception as e:
                print(f"[DEBUG] Error parsing line {i}: {e}")
    
    print(f"[DEBUG] Read {line_count} lines, loaded answers for {len(ans)} questions")
    
    # Show sample
    if ans:
        sample_qid = list(ans.keys())[0]
        print(f"[DEBUG] Sample answer (qid={sample_qid}):")
        print(f"  Turn 1: {len(ans[sample_qid][0])} chars - {ans[sample_qid][0][:100]}...")
        print(f"  Turn 2: {len(ans[sample_qid][1])} chars - {ans[sample_qid][1][:100]}...")
    else:
        print("[WARN] No answers loaded! Check file format.")
    
    return ans

def main():
    ap = argparse.ArgumentParser(description="MT-Bench judge with Prometheus")
    ap.add_argument("--judge_model", required=True, help="Prometheus model path")
    ap.add_argument("--questions_file", required=True)
    ap.add_argument("--model_answers_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--first_n", type=int, default=0, help="Evaluate first N questions (0=all)")
    ap.add_argument("--dtype", default="float16", choices=["float32","float16","bfloat16"])
    ap.add_argument("--raw_dump", default=None, help="Save raw model outputs")
    ap.add_argument("--debug_stop_after", type=int, default=0, help="Stop after N questions for debugging")
    args = ap.parse_args()

    print("="*60)
    print("Running MT-Bench Judge with PROMETHEUS")
    print("="*60)
    print(f"Judge:       {args.judge_model}")
    print(f"Questions:   {args.questions_file}")
    print(f"Answers:     {args.model_answers_file}")
    print(f"First N:     {args.first_n if args.first_n > 0 else 'ALL'}")
    print(f"Out:         {args.out_file}")
    print(f"Debug:       Stop after {args.debug_stop_after} questions" if args.debug_stop_after else "")
    print("="*60)

    torch_dtype = {"float32":torch.float32,"float16":torch.float16,"bfloat16":torch.bfloat16}[args.dtype]
    
    print("\n[1/4] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.judge_.model, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    print(f"  Tokenizer loaded. Vocab size: {tok.vocab_size}")

    print("\n[2/4] Loading model...")
    mdl = AutoModelForCausalLM.from_pretrained(
        args.judge_model, 
        torch_dtype=torch_dtype, 
        device_map="auto", 
        trust_remote_code=True
    ).eval()
    print(f"  Model loaded on device: {mdl.device}")

    print("\n[3/4] Loading questions and answers...")
    qs = load_questions(args.questions_file)
    ans = load_model_answers(args.model_answers_file)
    
    qids = sorted(qs.keys())
    if args.first_n and args.first_n > 0:
        qids = qids[:args.first_n]
    print(f"  Evaluating {len(qids)} questions")

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    out_f = open(args.out_file, "w", encoding="utf-8")
    raw_f = open(args.raw_dump, "w", encoding="utf-8") if args.raw_dump else None

    print("\n[4/4] Running judgments...")
    scores = []
    
    for idx, qid in enumerate(tqdm(qids, desc="Judging")):
        print(f"\n{'='*60}")
        print(f"Question {idx+1}/{len(qids)} (qid={qid})")
        print(f"{'='*60}")
        
        q1, q2 = qs[qid]
        a1, a2 = ans.get(qid, ["",""])
        
        if not a1 and not a2:
            print(f"[WARN] No answers found for qid={qid}")
        
        print(f"Q1: {len(q1)} chars, A1: {len(a1)} chars")
        print(f"Q2: {len(q2)} chars, A2: {len(a2)} chars")

        turn_objs = []
        for turn, (q, a) in enumerate(((q1,a1),(q2,a2)), start=1):
            if not q or not a:
                print(f"  Turn {turn}: SKIPPED (empty)")
                continue
            
            print(f"\n--- Turn {turn} ---")
            print(f"Question: {q[:150]}...")
            print(f"Answer: {a[:150]}...")
            
            inputs = build_prometheus_input(tok, q, a)
            inputs = {k:v.to(mdl.device) for k,v in inputs.items()}
            
            print(f"Generating judgment...")
            with torch.no_grad():
                out_ids = mdl.generate(
                    **inputs,
                    max_new_tokens=512,  # Prometheus can be verbose
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            
            # Decode only the generated part
            gen = tok.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            print(f"Generated ({len(gen)} chars):")
            print(gen)
            print("-" * 60)
            
            if raw_f:
                raw_f.write(json.dumps({
                    "question_id": qid, 
                    "turn": turn, 
                    "raw": gen,
                    "question": q[:200],
                    "answer": a[:200]
                }, ensure_ascii=False)+"\n")
                raw_f.flush()

            sc, expl = extract_prometheus_score(gen)
            turn_objs.append((sc, expl))
            
            if sc:
                print(f"✓ Score: {sc}/10")
            else:
                print(f"✗ Failed to extract score")

        valid = [sc for sc,_ in turn_objs if isinstance(sc,(int,float))]
        if valid:
            mean_sc = sum(valid)/len(valid)
            expl = " | ".join((ex or "")[:200] for _, ex in turn_objs)[:500]
            result = {"question_id": qid, "score": mean_sc, "explanation": expl}
            print(f"\n✓ Final score for qid={qid}: {mean_sc:.2f}/10")
        else:
            result = {"question_id": qid, "score": None, "explanation": "Failed to parse"}
            print(f"\n✗ No valid scores for qid={qid}")
        
        out_f.write(json.dumps(result, ensure_ascii=False)+"\n")
        out_f.flush()
        
        if valid:
            scores.append(mean_sc)
        
        
        if args.debug_stop_after > 0 and idx + 1 >= args.debug_stop_after:
            print("\n" + "="*60)
            print(f"DEBUG MODE: Stopping after {args.debug_stop_after} question(s)")
            print("="*60)
            break

    out_f.close()
    if raw_f: 
        raw_f.close()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if scores:
        mean = sum(scores)/len(scores)
        var = sum((s-mean)**2 for s in scores)/(len(scores)-1) if len(scores)>1 else 0.0
        se = math.sqrt(var/len(scores)) if len(scores)>1 else 0.0
        ci = 1.96*se if len(scores)>1 else 0.0
        print(f"Scores parsed: {len(scores)}/{len(qids)} questions")
        print(f"Mean: {mean:.3f}/10")
        if len(scores) > 1:
            print(f"Std Dev: {math.sqrt(var):.3f}")
            print(f"95% CI: [{mean-ci:.3f}, {mean+ci:.3f}]")
    else:
        print("[WARN] No valid scores parsed!")
        print("Check the raw output file to see what Prometheus generated")
    
    print(f"\n[OK] Judgments -> {args.out_file}")
    if args.raw_dump:
        print(f"[OK] Raw generations -> {args.raw_dump}")
    print("="*60)

if __name__ == "__main__":
    main()