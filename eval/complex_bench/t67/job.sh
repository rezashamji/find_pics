#!/bin/bash
# RESULTS 67: rewrite c programs + oracle on the new questions, one shard (inside scripts/race_sbatch.sh).
# Usage: bash eval/complex_bench/t67/job.sh <k> <K>
k=$1; K=$2
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
mkdir -p eval/complex_bench/t67/logs
LOG=eval/complex_bench/t67/logs/regen_$k.log
M=Qwen/Qwen3.5-27B-FP8
D=$FP_ROOT/.cache/huggingface/hub/models--${M//\//--}
echo "== $(date) $(hostname) shard $k/$K $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" >> $LOG
bash scripts/prewarm_env.sh vllm $D >> $LOG 2>&1
bash scripts/keepwarm.sh vllm 900 &
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_DEEP_GEMM=0 FP_MAX_SEQS=96
python eval/complex_bench/t67/regen.py --shard=$k/$K >> $LOG 2>&1
echo "EXIT $? $(date)" >> $LOG
