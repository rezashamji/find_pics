#!/bin/bash
# RESULTS 54: search stage for one shard of libraries (--by-user k/K) of a split, s1 then s0 in the same job, so the
# second run reads the first one's judge answers (shared jcache: same photo + question -> same answer) and the paired
# difference comes from the systems, not from judge noise. run_system.py resumes from its part file (requeue-safe).
# Usage (inside scripts/race_sbatch.sh): bash eval/complex_bench/m54/search_job.sh <split> <k> <K>
SPLIT=$1; k=$2; K=$3
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source eval/complex_bench/m54/systems.sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn
for S in s1 s0; do
  LOG=$OUT/logs/search_${SPLIT}_${S}_$k.log
  bash scripts/prewarm_env.sh vllm $FP_ROOT/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B > $LOG.warm 2>&1
  bash scripts/keepwarm.sh vllm 600 &
  KW=$!
  echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> $LOG
  PYTHONPATH=$(src_of $S) python eval/complex_bench/run_system.py --split=$SPLIT --by-user --shard=$k/$K \
    --plans=$OUT/${SPLIT}_${S}_plan/plans.json --cache=eval/complex_bench/runs/cmp/jcache --out=$OUT/m54_${SPLIT}_$S >> $LOG 2>&1
  echo "exit $? $(date)" >> $LOG
  kill $KW 2>/dev/null
done
