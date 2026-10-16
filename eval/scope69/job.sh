#!/bin/bash
# RESULTS 69: one library shard, configs AJ/DRY/ON x bf16/fp32 at 100% facts (inside scripts/race_sbatch.sh; vLLM env;
# frozen RESULTS-60 code exactly as RESULTS 65 / 68). Usage: bash eval/scope69/job.sh <split> <k> <K> [extra args]
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
SPLIT=$1; k=$2; K=$3; shift 3
M=eval/facts_scorer/models
OUT=${OUTDIR:-eval/scope69/replay/$SPLIT}
REC=${RECDIR:-data/public/scope69/record/$SPLIT}
mkdir -p $OUT $REC
LOG=$OUT/cascade_$k.log
if [ $SPLIT = train ]; then TV="$M/tvw_t2b_final_rs2_xfit_h{fold}"; VEC="$M/vecB_h{fold}"; F=eval/facts_scorer/data/facts_w_t2b_final_rs2_xfitm.parquet
  PL=eval/facts_scorer/replay/train_plans/plans_r60.json
else TV=$M/tvw_t2b_final_rs2_xfit_full; VEC=$M/vecB_full; F=eval/facts_scorer/data/facts_w_p2b_devmix_rs2m.parquet
  PL=eval/complex_bench/runs/cmp/r60v2_base/plan/plans.json; fi
CFG=${CONFIGS:-AJ_bf16,DRY_bf16,ON_bf16,DRY_fp32,ON_fp32}
bash scripts/prewarm_env.sh vllm > $LOG.warm 2>&1
bash scripts/keepwarm.sh vllm 600 &
KW=$!
export PYTHONPATH=$FP_ROOT/.cache/wt_r59b/src VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID" >> $LOG
python eval/scope69/cascade69.py --split=$SPLIT --shard=$k/$K --plans=$PL --cache=eval/complex_bench/runs/cmp/jcache \
  --tv=$TV --vec=$VEC --ce=$M/ce12B_full --facts=$F --orders=eval/partial65/data/orders.parquet --configs=$CFG \
  --out=$OUT --rec=$REC --save_sub=r69_${SPLIT:0:1} "$@" >> $LOG 2>&1
echo "EXIT $? $(date)" >> $LOG
kill $KW 2>/dev/null
