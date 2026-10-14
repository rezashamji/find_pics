#!/bin/bash
# RESULTS 59c: the ALL-JUDGE shipped stack (9B on every in-scope photo) on TRAIN libraries with the RESULTS 60 code and
# the shipped planner's TRAIN plans - the reference for the cascade on TRAIN, and it fills the shared 9B answer cache.
# Usage (inside scripts/race_sbatch.sh): bash eval/facts_scorer/alljudge_job.sh <k> <K>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
OUT=eval/facts_scorer/replay/train_alljudge; k=$1; K=$2
mkdir -p $OUT
LOG=$OUT/search_$k.log
bash scripts/prewarm_env.sh vllm $FP_ROOT/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B > $LOG.warm 2>&1
bash scripts/keepwarm.sh vllm 600 &
KW=$!
export PYTHONPATH=$FP_ROOT/.cache/wt_r59b/src VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID" >> $LOG
python eval/complex_bench/run_system.py --split=train --by-user --shard=$k/$K --plans=eval/facts_scorer/replay/train_plans/plans_r60.json \
  --cache=eval/complex_bench/runs/cmp/jcache --out=$OUT >> $LOG 2>&1
echo "SEARCH_EXIT $? $(date)" >> $LOG
kill $KW 2>/dev/null
