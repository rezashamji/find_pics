#!/bin/bash
# RESULTS 59b: one library shard of the cascade replay (inside scripts/race_sbatch.sh; vLLM env; frozen RESULTS-60 code).
# Usage: bash eval/facts_scorer/cascade_job.sh <out dir> <k> <K> <tv dir> <ce dir> <facts parquet>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
OUT=$1; k=$2; K=$3
mkdir -p $OUT
LOG=$OUT/cascade_$k.log
bash scripts/prewarm_env.sh vllm > $LOG.warm 2>&1
bash scripts/keepwarm.sh vllm 600 &
KW=$!
export PYTHONPATH=$FP_ROOT/.cache/wt_r59b/src VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID" >> $LOG
python eval/facts_scorer/cascade.py --split=dev --shard=$k/$K --plans=eval/complex_bench/runs/cmp/r60v2_base/plan/plans.json \
  --cache=eval/complex_bench/runs/cmp/jcache --tv=$4 --ce=$5 --facts=$6 --out=$OUT ${7:+--variants=$7} >> $LOG 2>&1
echo "EXIT $? $(date)" >> $LOG
kill $KW 2>/dev/null
