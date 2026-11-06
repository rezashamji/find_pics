#!/bin/bash
# RESULTS 71: gpu_test admits only 2 submitted jobs per user (QOSMaxSubmitJobPerUserLimit), so this loop feeds the shards one
# at a time: whenever fewer than 2 of my gpu_test jobs exist, submit a gpu_test copy (account mzitnik_lab, MIG A100 20 GB,
# 60 GB host memory cap) of the next unfinished shard that has no gpu_test copy yet. Same job name as the kempner_requeue
# copy (eval/nj71/launch.sh), so race_guard keeps exactly one. Ends when every listed shard has CASCADE_DONE.
#   bash eval/nj71/feed_gputest.sh <stage> <split> <k ...>      (background)
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source env.sh
STAGE=$1; SPLIT=$2; shift 2; KS="$*"; [ $SPLIT = train ] && K=24 || K=13
[ $SPLIT = train ] && SRC=t2b_final_rs2_xfit || SRC=p2b_devmix_rs2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [ $STAGE = s1 ]; then
  G=$(for t in 05 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95; do printf ",NJ_fp32_T$t"; done); export CONFIGS=DRY_fp32$G
else
  T=$(awk '{print $1}' eval/nj71/threshold.txt)
  export CONFIGS=NJ_fp32_T${T},NJ_fp32_T${T}_Fk16,NJ_fp32_T${T}_Fk24,NJ_fp32_T${T}_Fk32,NJ_fp32_T${T}_Fk48,NJ_fp32_T${T}_Fk64,NJ_fp32_T${T}_0,NJ_fp32_T${T}_10,NJ_fp32_T${T}_25,NJ_fp32_T${T}_50
fi
export OUTDIR=eval/nj71/replay/$STAGE/$SPLIT RECDIR=data/public/nj71/record/$STAGE/$SPLIT
FV="--fvar=eval/facts_scorer/data/facts_w_r70_${SRC}_{tag}m.parquet"
while true; do
  left=0
  for k in $KS; do
    J=fp_r71${STAGE}_${SPLIT:0:1}_$k
    grep -q CASCADE_DONE $OUTDIR/cascade_$k.log 2>/dev/null && continue
    left=$((left+1))
    squeue -u $USER -h -p gpu_test -o %j | grep -qx $J && continue
    [ $(squeue -u $USER -h -p gpu_test | wc -l) -ge 2 ] && continue
    ACCT=mzitnik_lab SPECS="gpu_test:gpu:1" MEM=60G bash scripts/race_sbatch.sh $J 00:45:00 bash eval/scope69/job.sh $SPLIT $k $K "$FV" > /dev/null 2>&1 \
      && echo "fed $J $(date +%T)"
  done
  [ $left = 0 ] && { echo "FEED_DONE $STAGE $SPLIT $(date)"; break; }
  sleep 30
done
