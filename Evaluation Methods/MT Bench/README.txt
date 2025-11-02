

Steps to Follow


1.  File Placement:

      Place both files (`run_prometheus_judge.sh` and `judge_mtbench_local.py`) into your FastChat judge directory.
      
        /home/roheth/my_project/assignment7/FastChat/fastchat/llm_judge/

2.  Verify Model Answers

      Before we run, we must make sure the model answers you generated with 'run_mt_bench_gen.sh' are in the correct directory.
      
        `/home/roheth/my_project/assignment7/FastChat/fastchat/llm_judge/data/mt_bench/model_answer/`
      We can see these two files (or files with similar names, which you can then update in the `MODELS_TO_TEST` array in the script):
          Llama2-7B-HF-Base.jsonl
          Llama2-7B-Dolly-QLoRA-Finetuned.jsonl

3.  Make Script Executable:

       Navigate to the directory from Step 1 and run:
        
        chmod +x run_prometheus_judge.sh
        

4.  Submit the Job

    Submit the script to the SLURM scheduler:
       
        sbatch run_prometheus_judge.sh
        

5.  Results

      You can check the live output of your job using:
        
        tail -f logs/mtbench_prometheus_[YOUR_JOB_ID].out
        
       When the job is finished, the final scores (the `.jsonl` files) will be located in:
        `/home/roheth/my_project/assignment7/FastChat/fastchat/llm_judge/data/mt_bench/model_judgment/`
      The log file will contain the printed summaries