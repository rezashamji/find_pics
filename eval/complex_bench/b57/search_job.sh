#!/bin/bash
# RESULTS 57: DEV search stage (9B judge, shared answer cache runs/cmp/jcache: same photo + question -> same answer) of
# one plan file, one library shard (--by-user k/K); run_system.py resumes from its part file (requeue-safe).
# Usage (inside scripts/race_sbatch.sh): bash eval/complex_bench/b57/search_job.sh <plans.json> <out dir> <k> <K>
P=$1; OUT=$2; k=$3; K=$4
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
mkdir -p $OUT
LOG=$OUT/search_$k.log
bash scripts/prewarm_env.sh vllm $FP_ROOT/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B > $LOG.warm 2>&1
bash scripts/keepwarm.sh vllm 600 &
export PYTHONPATH=$FP_ROOT/.cache/wt_m54_s1/src VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID" >> $LOG
python eval/complex_bench/run_system.py --split=dev --by-user --shard=$k/$K --plans=$P \
  --cache=eval/complex_bench/runs/cmp/jcache --out=$OUT >> $LOG 2>&1
echo "SEARCH_EXIT $? $(date)" >> $LOG
