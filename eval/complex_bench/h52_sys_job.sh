#!/bin/bash
# RESULTS 52: the shipped planner (10-05 adapter, phone 4-bit weights) + the 9B judge (shared jcache) on the dev h
# requests that the NEW truth keeps and no earlier run planned (queries/dev_h52new.json), once with the 49c code
# (git 3b1aa08) and once with the RESULTS 51 final code (.cache/cb51/snap_r51f). Plan raw outputs are reused from the
# runs being extended (identical prompts -> identical outputs). Inside scripts/race_sbatch.sh (vLLM env active).
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
SPLIT=${1:-dev_h52new}; OUT=${2:-eval/complex_bench/runs/cmp/r52_new}; mkdir -p $OUT   # args: queries split, output dir
M=$FP_ROOT/models/planner_4b27ball_q4merged
bash scripts/prewarm_env.sh vllm $M $FP_ROOT/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B > $OUT/warm.log 2>&1
bash scripts/keepwarm.sh vllm 900 &
export VLLM_WORKER_MULTIPROC_METHOD=spawn
for V in 49c 51f; do
  if [ $V = 49c ]; then SRC=$FP_ROOT/.cache/r52_src49c/src; REUSE=.cache/cb49c/plan_A4oldq4/raw.json
  else SRC=$FP_ROOT/.cache/cb51/snap_r51f/src; REUSE=eval/complex_bench/runs/cmp/r51/raw.json; fi
  echo "== $V plan $(date) host $(hostname)" >> $OUT/run.log
  PYTHONPATH=$SRC python eval/complex_bench/plan_system.py --split=$SPLIT --out=$OUT/plan_$V --model=$M --reuse=$REUSE >> $OUT/run.log 2>&1
  echo "PLAN_EXIT $V $?" >> $OUT/run.log
  PYTHONPATH=$SRC python eval/complex_bench/run_system.py --split=$SPLIT --plans=$OUT/plan_$V/plans.json \
    --cache=eval/complex_bench/runs/cmp/jcache --out=$OUT/search_$V >> $OUT/run.log 2>&1
  echo "SEARCH_EXIT $V $?" >> $OUT/run.log
done
