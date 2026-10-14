#!/bin/bash
# RESULTS 58: one shard of the training pairs labelled by one judge (inside scripts/race_sbatch.sh, vLLM env).
# Usage: bash eval/judge_teach/label_job.sh <k> <K> <name> <model> [tp=1]
K_=$1; KK=$2; NAME=$3; M=$4; TP=${5:-1}
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
LOG=eval/judge_teach/logs/label_${NAME}_$K_.log
[ -f data/public/judge_teach/labels/$NAME/part$K_.parquet ] && { echo "already done" >> $LOG; exit 0; }
D=$M; [ -d "$M" ] || D=$FP_ROOT/.cache/huggingface/hub/models--${M//\//--}
echo "== $(date) $(hostname) $NAME shard $K_/$KK $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" >> $LOG
bash scripts/prewarm_env.sh vllm $D data/public/judge_teach/img/part$K_.bin >> $LOG 2>&1
bash scripts/keepwarm.sh vllm 900 &
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_DEEP_GEMM=0 FP_MAX_SEQS=${FP_MAX_SEQS:-256}
python eval/judge_teach/label.py judge --shard=$K_/$KK --name=$NAME --model=$M --tp=$TP >> $LOG 2>&1
echo "EXIT $? $(date)" >> $LOG
