#!/bin/bash
# RESULTS 63: direct whole-request judge, one shard (inside scripts/race_sbatch.sh).
# Usage: bash eval/complex_bench/t63/job.sh <tag> <HF repo> <k> <K> [extra direct.py args, e.g. --tp=2 --units=...]
TAG=$1; M=$2; k=$3; K=$4; shift 4
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
mkdir -p eval/complex_bench/t63/logs
LOG=eval/complex_bench/t63/logs/${TAG}_$k.log
D=$FP_ROOT/.cache/huggingface/hub/models--${M//\//--}
echo "== $(date) $(hostname) $TAG $M shard $k/$K $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" >> $LOG
bash scripts/prewarm_env.sh vllm $D >> $LOG 2>&1
bash scripts/keepwarm.sh vllm 900 &
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_DEEP_GEMM=0
python eval/complex_bench/t63/direct.py $TAG $M --shard=$k/$K "$@" >> $LOG 2>&1
echo "EXIT $? $(date)" >> $LOG
