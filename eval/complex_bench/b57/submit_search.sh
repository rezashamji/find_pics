#!/bin/bash
# RESULTS 57: after a plan stage finished, submit its dev search in K library shards (race_sbatch; 9B judge + shared jcache).
# Usage: bash eval/complex_bench/b57/submit_search.sh <name: base|q4|fp16> <K>
N=$1; K=$2
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
P=eval/complex_bench/runs/cmp/r57_$N/plan/plans.json
[ -f $P ] || { echo "missing $P"; exit 1; }
for k in $(seq 0 $((K-1))); do
  bash scripts/race_sbatch.sh fp_b57_s_${N}_$k 3:00:00 bash eval/complex_bench/b57/search_job.sh $P eval/complex_bench/runs/cmp/r57_${N}_search $k $K
done
