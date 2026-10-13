#!/bin/bash
# RESULTS 51: dev search stage on the r51 plans with the 9B judge + shared jcache, 13 shards by library (as 49/49b).
# Usage: bash .cache/cb51/submit_search.sh <run name> <plans.json> <src snapshot>
R=$1; PL=$2; SNAP=$3
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
for k in $(seq 0 12); do
  WARM="" bash scripts/race_sbatch.sh fp_cb51_${R}_$k 2:00:00 bash eval/complex_bench/cmp_job.sh $SNAP \
    eval/complex_bench/runs/cmp/logs/${R}_$k.log eval/complex_bench/run_system.py --split=dev --by-user --shard=$k/13 \
    --plans=$PL --cache=eval/complex_bench/runs/cmp/jcache --out=eval/complex_bench/runs/cmp/${R}
done
