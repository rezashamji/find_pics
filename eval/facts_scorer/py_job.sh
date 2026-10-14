#!/bin/bash
# RESULTS 59: run one python script on a GPU node (inside scripts/race_sbatch.sh, vLLM env).
# Usage: bash eval/facts_scorer/py_job.sh <log name> <script.py> <args...>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
mkdir -p eval/facts_scorer/logs
N=$1; shift
LOG=eval/facts_scorer/logs/$N.log
bash scripts/prewarm_env.sh vllm > $LOG.warm 2>&1
bash scripts/keepwarm.sh vllm 600 &
KW=$!
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID" >> $LOG
python "$@" >> $LOG 2>&1
echo "EXIT $? $(date)" >> $LOG
kill $KW 2>/dev/null
