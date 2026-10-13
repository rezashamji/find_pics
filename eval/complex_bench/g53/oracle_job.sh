#!/bin/bash
# RESULTS 53: helper oracle job (inside scripts/race_sbatch.sh): answer the new g questions for the given libraries, taken
# from the TAIL of a slow fp_cb53_regen shard in reverse order (oracle.py is incremental per library).
# Usage: bash eval/complex_bench/g53/oracle_job.sh <snapshot dir> <log> <U1,U2,...>
SNAP=$1; LOG=$2; USERS=$3
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
bash scripts/prewarm_env.sh vllm $FP_ROOT/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B-FP8 > $LOG.warm 2>&1
bash scripts/keepwarm.sh vllm 900 &
export VLLM_WORKER_MULTIPROC_METHOD=spawn FP_MAX_SEQS=96 VLLM_USE_DEEP_GEMM=0
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> $LOG
python $SNAP/oracle.py --users=$USERS >> $LOG 2>&1
echo "exit $? $(date)" >> $LOG
