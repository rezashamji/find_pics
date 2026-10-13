#!/bin/bash
# RESULTS 51: the FINAL code (closest-moment fallback removed after the absent-moment probe) replayed on the r51 plans
# with --judge=cache (no model: every answer must come from the shared jcache, which r51_search filled). CPU work.
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
bash scripts/prewarm_env.sh vllm > .cache/cb51/replay_warm.log 2>&1
for u in $(cat eval/complex_bench/eh_diag/dev_users.txt); do find data/public/raw/disbench/images/images/$u -type f -print0; done | xargs -0 -P 64 -n 40 cat > /dev/null
export PYTHONPATH=$FP_ROOT/.cache/cb51/snap_r51f/src
K=$1
python eval/complex_bench/run_system.py --split=dev --by-user --shard=$K/13 --plans=eval/complex_bench/runs/cmp/r51/plans.json \
  --cache=eval/complex_bench/runs/cmp/jcache --judge=cache --out=eval/complex_bench/runs/cmp/r51f_search \
  > eval/complex_bench/runs/cmp/logs/r51f_search_$K.log 2>&1
echo "exit $?" >> eval/complex_bench/runs/cmp/logs/r51f_search_$K.log
