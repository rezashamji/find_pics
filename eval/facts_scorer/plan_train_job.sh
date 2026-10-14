#!/bin/bash
# RESULTS 59 fix B: the SHIPPED planner (10-05 adapter on the phone's 4-bit weights, shipped code) plans every TRAIN
# request, so the scorer can be trained on the question wording the product actually asks (oracle labels added next).
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
OUT=eval/facts_scorer/replay/train_plans
mkdir -p $OUT
M=$FP_ROOT/models/planner_4b27ball_q4merged
bash scripts/prewarm_env.sh vllm $M > $OUT/plan.log.warm 2>&1
bash scripts/keepwarm.sh vllm 900 &
export PYTHONPATH=$FP_ROOT/.cache/wt_m54_s1/src VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "host $(hostname) start $(date) job $SLURM_JOB_ID" >> $OUT/plan.log
python eval/complex_bench/plan_system.py --split=train --out=$OUT/plan --model=$M >> $OUT/plan.log 2>&1
echo "PLAN_EXIT $? $(date)" >> $OUT/plan.log
