#!/bin/bash
# RESULTS 70 launcher: the phone configuration (DRY fp32, policy V, 100%) with the writer line capped at K tokens (configs
# DRY_fp32_F<tag>, tag = k16..k64 / r48 r64 rfull) + the uncapped control DRY_fp32, same jobs (models loaded once).
#   bash eval/trunc70/launch.sh <train|dev>   (one shard per library; sbatch only, background)
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source env.sh
SPLIT=$1; [ $SPLIT = train ] && K=24 || K=13
[ $SPLIT = train ] && SRC=t2b_final_rs2_xfit || SRC=p2b_devmix_rs2
export SPECS="kempner_requeue:gpu:nvidia_h100_80gb_hbm3:1 kempner_rtx:gpu:1 kempner_h100:gpu:1 kempner_h200:gpu:1"   # no A100-40GB (lazy 9B)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CONFIGS=${CONFIGS:-DRY_fp32,DRY_fp32_Fk16,DRY_fp32_Fk24,DRY_fp32_Fk32,DRY_fp32_Fk48,DRY_fp32_Fk64,DRY_fp32_Fr48,DRY_fp32_Fr64,DRY_fp32_Frfull}
export OUTDIR=${OUTDIR:-eval/trunc70/replay/$SPLIT} RECDIR=${RECDIR:-data/public/trunc70/record/$SPLIT}
FV="--fvar=eval/facts_scorer/data/facts_w_r70_${SRC}_{tag}m.parquet"
inq() { squeue -u $USER -h -o %j | grep -qx "$1"; }
for k in $(seq 0 $((K-1))); do
  J=fp_r70_${SPLIT:0:1}_$k
  { grep -q CASCADE_DONE $OUTDIR/cascade_$k.log 2>/dev/null || inq $J; } && continue
  MEM=128G bash scripts/race_sbatch.sh $J 08:00:00 bash eval/scope69/job.sh $SPLIT $k $K "$FV" > /dev/null
done
echo "R70_SUBMITTED $SPLIT $(date)"
