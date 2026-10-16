#!/bin/bash
# RESULTS 65: one library shard, all configs (inside scripts/race_sbatch.sh; vLLM env; frozen RESULTS-60 code as 59d/64).
# Usage: bash eval/partial65/job.sh <split> <k> <K>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
SPLIT=$1; k=$2; K=$3; shift 3
M=eval/facts_scorer/models
OUT=${OUTDIR:-eval/partial65/replay/$SPLIT}
mkdir -p $OUT
LOG=$OUT/cascade_$k.log
if [ $SPLIT = train ]; then TV="$M/tvw_t2b_final_rs2_xfit_h{fold}"; VEC="$M/vecB_h{fold}"; F=eval/facts_scorer/data/facts_w_t2b_final_rs2_xfitm.parquet
  PL=eval/facts_scorer/replay/train_plans/plans_r60.json
else TV=$M/tvw_t2b_final_rs2_xfit_full; VEC=$M/vecB_full; F=eval/facts_scorer/data/facts_w_p2b_devmix_rs2m.parquet
  PL=eval/complex_bench/runs/cmp/r60v2_base/plan/plans.json; fi
CFG=${CONFIGS:-$(python3 -c "
c=['100_N_A','0_N_A','0_N_V']+[f'{v}_{o}_{p}' for v in (10,25,50,75) for o in 'NRE' for p in ('A','V','VS')]
print(','.join(c))")}
bash scripts/prewarm_env.sh vllm > $LOG.warm 2>&1
bash scripts/keepwarm.sh vllm 600 &
KW=$!
export PYTHONPATH=$FP_ROOT/.cache/wt_r59b/src VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID" >> $LOG
python eval/partial65/cascade65.py --split=$SPLIT --shard=$k/$K --plans=$PL --cache=eval/complex_bench/runs/cmp/jcache \
  --tv=$TV --vec=$VEC --ce=$M/ce12B_full --facts=$F --orders=eval/partial65/data/orders.parquet --configs=$CFG \
  --out=$OUT --save_sub=r65_${SPLIT:0:1} "$@" >> $LOG 2>&1
echo "EXIT $? $(date)" >> $LOG
kill $KW 2>/dev/null
