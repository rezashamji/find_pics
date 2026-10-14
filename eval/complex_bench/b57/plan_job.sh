#!/bin/bash
# RESULTS 57: DEV plan stage of one planner (merged model dir) with the shipped code (.cache/wt_m54_s1/src = 64709ef =
# HEAD src), then the 30-conversation gate (eval_planners.py, FP_PLANNERS_ONLY_CONV) on the same weights.
# Usage (inside scripts/race_sbatch.sh): bash eval/complex_bench/b57/plan_job.sh <model dir> <out dir> [reuse raw.json]
M=$1; OUT=$2; R=$3
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
mkdir -p $OUT
bash scripts/prewarm_env.sh vllm $M > $OUT/plan.log.warm 2>&1
bash scripts/keepwarm.sh vllm 900 &
export PYTHONPATH=$FP_ROOT/.cache/wt_m54_s1/src VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID model $M" >> $OUT/plan.log
if [ ! -f $OUT/plan/plans.json ]; then
  python eval/complex_bench/plan_system.py --split=dev --out=$OUT/plan --model=$M ${R:+--reuse=$R} >> $OUT/plan.log 2>&1
  echo "PLAN_EXIT $? $(date)" >> $OUT/plan.log
fi
if ! grep -q "conversations:" $OUT/gate.log 2>/dev/null; then
  FP_VLM_MODEL=$M FP_PLANNERS_ONLY_CONV=1 FP_PLANNERS_OUT=$OUT/gate python eval/eval_planners.py > $OUT/gate.log 2>&1
  echo "GATE_EXIT $? $(date)" >> $OUT/gate.log
fi
