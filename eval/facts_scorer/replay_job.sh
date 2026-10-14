#!/bin/bash
# RESULTS 59: one library shard of the end-to-end replay (inside scripts/race_sbatch.sh, vLLM env, shipped code).
# Usage: bash eval/facts_scorer/replay_job.sh <out dir> <k> <K> <replay.py args...>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
OUT=$1; k=$2; K=$3; shift 3
mkdir -p $OUT
LOG=$OUT/replay_$k.log
bash scripts/prewarm_env.sh vllm > $LOG.warm 2>&1
bash scripts/keepwarm.sh vllm 600 &
KW=$!
export PYTHONPATH=$FP_ROOT/.cache/wt_m54_s1/src
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID" >> $LOG
python eval/facts_scorer/replay.py --split=dev --shard=$k/$K --plans=eval/complex_bench/runs/cmp/r57_base/plan/plans.json --out=$OUT "$@" >> $LOG 2>&1
echo "EXIT $? $(date)" >> $LOG
kill $KW 2>/dev/null
