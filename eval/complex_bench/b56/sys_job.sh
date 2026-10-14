#!/bin/bash
# RESULTS 56: the SHIPPED stack S1 of RESULTS 54 (10-05 planner adapter on the phone's 4-bit weights + code 64709ef, the
# sparse worktree .cache/wt_m54_s1) + the 9B judge (shared jcache) on dev b requests the NEW truth keeps that S1 never
# planned (queries/<split>.json). Plan raw outputs reused from the m54 dev S1 run (identical prompts -> identical outputs).
# Usage (inside scripts/race_sbatch.sh, vLLM env active): bash eval/complex_bench/b56/sys_job.sh <split> <out dir>
SPLIT=$1; OUT=$2
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
mkdir -p $OUT
M=$FP_ROOT/models/planner_4b27ball_q4merged; SRC=$FP_ROOT/.cache/wt_m54_s1/src
bash scripts/prewarm_env.sh vllm $M $FP_ROOT/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B > $OUT/warm.log 2>&1
bash scripts/keepwarm.sh vllm 900 &
export VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "== s1 plan $(date) host $(hostname)" >> $OUT/run.log
PYTHONPATH=$SRC python eval/complex_bench/plan_system.py --split=$SPLIT --out=$OUT/plan_s1 --model=$M \
  --reuse=eval/complex_bench/runs/m54/dev_s1_plan/raw.json >> $OUT/run.log 2>&1
echo "PLAN_EXIT $?" >> $OUT/run.log
PYTHONPATH=$SRC python eval/complex_bench/run_system.py --split=$SPLIT --plans=$OUT/plan_s1/plans.json \
  --cache=eval/complex_bench/runs/cmp/jcache --out=$OUT/search_s1 >> $OUT/run.log 2>&1
echo "SEARCH_EXIT $?" >> $OUT/run.log
