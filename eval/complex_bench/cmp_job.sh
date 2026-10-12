#!/bin/bash
# One complex-bench job (plan stage or search stage) inside scripts/race_sbatch.sh (vLLM env already active).
# Usage: bash eval/complex_bench/cmp_job.sh <src dir for PYTHONPATH (frozen snapshot)> <log file> <python args...>
#   WARM="dir ..." : extra model folders to read into the page cache first (the 9B judge's is always warmed).
SNAP=$1; LOG=$2; shift 2
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
bash scripts/prewarm_env.sh vllm $FP_ROOT/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B $WARM > $LOG.warm 2>&1
bash scripts/keepwarm.sh vllm 900 &
export PYTHONPATH=$SNAP VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "host $(hostname) start $(date)" > $LOG
python "$@" >> $LOG 2>&1
rc=$?
echo "exit $rc $(date)" >> $LOG
exit $rc
