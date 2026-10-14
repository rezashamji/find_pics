#!/bin/bash
# RESULTS 60: DEV plan stage of one planner with the code in a frozen snapshot (the real-X exemption), reusing an earlier
# run's raw planner outputs (identical prompts -> identical answers), then the 30-conversation gate on the same weights.
# Usage (inside scripts/race_sbatch.sh): bash eval/complex_bench/r60/plan_job.sh <model dir> <out dir> <reuse raw.json> <src snapshot> [gate]
M=$1; OUT=$2; R=$3; SNAP=$4; G=$5
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
mkdir -p $OUT
bash scripts/prewarm_env.sh vllm $M > $OUT/plan.log.warm 2>&1
bash scripts/keepwarm.sh vllm 900 &
export PYTHONPATH=$SNAP VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID model $M snap $SNAP" >> $OUT/plan.log
if [ ! -f $OUT/plan/plans.json ]; then
  python eval/complex_bench/plan_system.py --split=dev --out=$OUT/plan --model=$M --reuse=$R >> $OUT/plan.log 2>&1
  echo "PLAN_EXIT $? $(date)" >> $OUT/plan.log
fi
if [ -n "$G" ] && ! grep -q "conversations:" $OUT/gate.log 2>/dev/null; then
  FP_VLM_MODEL=$M FP_PLANNERS_ONLY_CONV=1 FP_PLANNERS_OUT=$OUT/gate python eval/eval_planners.py > $OUT/gate.log 2>&1
  echo "GATE_EXIT $? $(date)" >> $OUT/gate.log
fi
