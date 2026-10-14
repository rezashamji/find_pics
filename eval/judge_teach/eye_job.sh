#!/bin/bash
# RESULTS 58: one judge on every eye-labelled row (eval/judge_teach/judge_eye.py), inside scripts/race_sbatch.sh.
# Usage: bash eval/judge_teach/eye_job.sh <name> <model dir or HF repo> [tp=1] [extra judge_eye args...]
# Env passed through: FP_JUDGE_THINK, FP_THINK_TOKENS, FP_MAX_SEQS.
NAME=$1; M=$2; TP=${3:-1}; shift 3
[ "$TP" = auto ] && TP=$(nvidia-smi -L | wc -l)
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
mkdir -p eval/judge_teach/logs
LOG=eval/judge_teach/logs/eye_$NAME.log
D=$M; [ -d "$M" ] || D=$FP_ROOT/.cache/huggingface/hub/models--${M//\//--}
echo "== $(date) $(hostname) $NAME $M tp=$TP $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" >> $LOG
bash scripts/prewarm_env.sh vllm $D >> $LOG 2>&1
bash scripts/keepwarm.sh vllm 900 &
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_DEEP_GEMM=0 FP_MAX_SEQS=${FP_MAX_SEQS:-256}
python eval/judge_teach/judge_eye.py $NAME $M --tp=$TP "$@" >> $LOG 2>&1
echo "EXIT $? $(date)" >> $LOG
