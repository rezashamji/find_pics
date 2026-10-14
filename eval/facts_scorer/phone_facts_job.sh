#!/bin/bash
# RESULTS 59 step 2: phone-size facts for some libraries (inside scripts/race_sbatch.sh, vLLM env).
# Usage: bash eval/facts_scorer/phone_facts_job.sh <model dir> <tag> <U1,U2,...> <log name>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
mkdir -p eval/facts_scorer/logs
LOG=eval/facts_scorer/logs/$4.log
bash scripts/prewarm_env.sh vllm $1 > $LOG.warm 2>&1
bash scripts/keepwarm.sh vllm 600 &
KW=$!
export VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID" >> $LOG
python eval/facts_scorer/phone_facts.py --model=$1 --tag=$2 --users=$3 >> $LOG 2>&1
echo "EXIT $? $(date)" >> $LOG
kill $KW 2>/dev/null
