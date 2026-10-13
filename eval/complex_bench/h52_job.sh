#!/bin/bash
# RESULTS 52: one GPU job (inside scripts/race_sbatch.sh, vLLM env active): rewrite the h truth programs of a shard of
# libraries with Qwen3.5-27B-FP8, then the oracle answers the NEW questions only (same model load).
# Usage: bash eval/complex_bench/h52_job.sh <snapshot dir with gen.py/oracle.py/common.py> <log file> <k/K> [regen_h|regen_g]
# (RESULTS 53: regen_g rewrites category g the same way)
SNAP=$1; LOG=$2; SHARD=$3; WHAT=${4:-regen_h}
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
bash scripts/prewarm_env.sh vllm $FP_ROOT/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B-FP8 > $LOG.warm 2>&1
bash scripts/keepwarm.sh vllm 900 &
export VLLM_WORKER_MULTIPROC_METHOD=spawn FP_MAX_SEQS=${FP_MAX_SEQS:-96} VLLM_USE_DEEP_GEMM=0   # DeepGEMM JIT needs nvcc (absent)
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> $LOG
python $SNAP/gen.py $WHAT --shard=$SHARD >> $LOG 2>&1
rc=$?
echo "exit $rc $(date)" >> $LOG
exit $rc
