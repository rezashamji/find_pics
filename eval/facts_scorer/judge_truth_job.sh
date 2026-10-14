#!/bin/bash
# RESULTS 59: the 9B product judge on the dev truth questions of the given libraries (inside scripts/race_sbatch.sh).
# Usage: bash eval/facts_scorer/judge_truth_job.sh <U1,U2,...> <log name>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
mkdir -p eval/facts_scorer/logs
LOG=eval/facts_scorer/logs/$2.log
bash scripts/prewarm_env.sh vllm $FP_ROOT/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B > $LOG.warm 2>&1
bash scripts/keepwarm.sh vllm 600 &
KW=$!
export VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID" >> $LOG
python eval/facts_scorer/judge_truth.py --users=$1 >> $LOG 2>&1
echo "EXIT $? $(date)" >> $LOG
kill $KW 2>/dev/null
