#!/bin/bash
# RESULTS 54: plan stage of one system (s0 | s1) on TEST, then on DEV (reference), inside scripts/race_sbatch.sh.
# Dev reuses the same model's earlier raw outputs for identical prompts (.cache/m54/dev_reuse_<sys>.json); test reuses
# nothing. A finished plans.json is never re-made (requeue-safe: each split is planned once).
S=$1
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source eval/complex_bench/m54/systems.sh
M=$(model_of $S); LOG=$OUT/logs/plan_$S.log
bash scripts/prewarm_env.sh vllm $M > $LOG.warm 2>&1
bash scripts/keepwarm.sh vllm 900 &
export PYTHONPATH=$(src_of $S) VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) PYTHONPATH=$PYTHONPATH" >> $LOG
for SPLIT in test dev; do
  D=$OUT/${SPLIT}_${S}_plan
  [ -f $D/plans.json ] && { echo "SKIP $SPLIT (plans.json exists)" >> $LOG; continue; }
  R=""; [ $SPLIT = dev ] && R="--reuse=.cache/m54/dev_reuse_$S.json"
  python eval/complex_bench/plan_system.py --split=$SPLIT --out=$D --model=$M $R >> $LOG 2>&1
  echo "PLAN_EXIT $SPLIT $? $(date)" >> $LOG
done
