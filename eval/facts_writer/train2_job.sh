#!/bin/bash
# RESULTS 61b: one multi-GPU student training (inside scripts/race_sbatch.sh with a multi-GPU gres).
# Usage: bash eval/facts_writer/train2_job.sh <name> <ngpu> <train_writer2.py args...>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
N=$1; G=$2; shift 2
LOG=eval/facts_writer/logs/$N.log
bash scripts/prewarm_env.sh vllm > $LOG.warm 2>&1
bash scripts/keepwarm.sh vllm 600 &
KW=$!
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x$G job $SLURM_JOB_ID" >> $LOG
torchrun --standalone --nproc_per_node=$G eval/facts_writer/train_writer2.py "$@" >> $LOG 2>&1
echo "EXIT $? $(date)" >> $LOG
kill $KW 2>/dev/null
