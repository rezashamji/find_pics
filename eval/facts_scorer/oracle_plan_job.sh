#!/bin/bash
# RESULTS 59 fix B: 27B oracle on the shipped planner's TRAIN questions (inside scripts/race_sbatch.sh; FP8 GPU).
# Usage: bash eval/facts_scorer/oracle_plan_job.sh <U1,U2,...> <log name>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
LOG=eval/facts_scorer/logs/$2.log
bash scripts/prewarm_env.sh vllm $FP_ROOT/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B-FP8 > $LOG.warm 2>&1
bash scripts/keepwarm.sh vllm 600 &
KW=$!
export VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID" >> $LOG
python eval/facts_scorer/oracle_plan.py --users=$1 --questions=eval/facts_scorer/data/plan_questions_train.parquet >> $LOG 2>&1
echo "EXIT $? $(date)" >> $LOG
kill $KW 2>/dev/null
