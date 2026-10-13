#!/bin/bash
# RESULTS 51: dev plan stage with the shipped planner (10-05 adapter on the phone's 4-bit weights) on the new code
# (frozen snapshot .cache/cb51/snap_r51/src), reusing the 49c run's raw outputs; then the 30-conversation gate.
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
M=$FP_ROOT/models/planner_4b27ball_q4merged
bash scripts/prewarm_env.sh vllm $M > .cache/cb51/plan_warm.log 2>&1
export PYTHONPATH=$FP_ROOT/.cache/cb51/snap_r51/src VLLM_WORKER_MULTIPROC_METHOD=spawn
python eval/complex_bench/plan_system.py --split=dev --out=eval/complex_bench/runs/cmp/r51 --model=$M \
  --reuse=.cache/cb49c/plan_A4oldq4/raw.json > .cache/cb51/plan_r51.log 2>&1
echo "PLAN_EXIT $?" >> .cache/cb51/plan_r51.log
FP_VLM_MODEL=$M FP_PLANNERS_ONLY_CONV=1 FP_PLANNERS_OUT=eval/complex_bench/runs/cmp/gate/r51_q4 \
  python eval/eval_planners.py > eval/complex_bench/runs/cmp/gate/r51_q4.log 2>&1
echo "GATE_EXIT $?" >> eval/complex_bench/runs/cmp/gate/r51_q4.log
