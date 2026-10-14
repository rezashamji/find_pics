#!/bin/bash
# RESULTS 61: one student training (inside scripts/race_sbatch.sh; vLLM env has torch + transformers + peft).
# Usage: bash eval/facts_writer/train_job.sh <name> <train_writer.py args...>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
N=$1; shift
LOG=eval/facts_writer/logs/$N.log
bash scripts/prewarm_env.sh vllm > $LOG.warm 2>&1
bash scripts/keepwarm.sh vllm 600 &
KW=$!
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID" >> $LOG
python eval/facts_writer/train_writer.py "$@" >> $LOG 2>&1
echo "EXIT $? $(date)" >> $LOG
kill $KW 2>/dev/null
