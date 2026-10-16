#!/bin/bash
# RESULTS 64: R59d cascade with SHARED facts (eval/dedup64/group.py G(s)); scorers NOT retrained (pre-registered):
# TRAIN = fold scorers tvw_t2b_final_rs2_xfit_h{fold} over the cross-fit 61b 2B + v2 lines; dev = tvw_t2b_final_rs2_xfit_full
# over the product p2b_all + v2 lines (exactly the 61d per-photo baselines c61d_{train_t2b_final_rs2_xfit,dev_p2b_devmix_rs2}).
# Usage: bash eval/dedup64/launch.sh <train|dev> <s tag, e.g. 0.96 or inf> ...   (background, login node: sbatch only)
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source env.sh
SPLIT=$1; shift
M=eval/facts_scorer/models; TP=eval/facts_scorer/replay/train_plans/plans_r60.json
export SPECS="kempner_requeue:gpu:nvidia_h100_80gb_hbm3:1 kempner_rtx:gpu:1 kempner_h100:gpu:1 kempner_h200:gpu:1"   # no A100-40GB (lazy 9B fails engine init there)
inq() { squeue -u $USER -h -o %j | grep -qx "$1"; }
for S in "$@"; do
  F=eval/dedup64/data/facts_shared_s$S.parquet; [ -f $F ] || { echo "missing $F"; exit 1; }
  O=eval/dedup64/replay/${SPLIT}_s$S
  if [ $SPLIT = train ]; then K=12; TV="$M/tvw_t2b_final_rs2_xfit_h{fold}"; ENV="env SPLIT=train PLANS=$TP"; else K=13; TV=$M/tvw_t2b_final_rs2_xfit_full; ENV=""; fi
  for k in $(seq 0 $((K-1))); do
    J=fp_r64_${SPLIT:0:1}_${S}_$k
    { grep -q CASCADE_DONE $O/cascade_$k.log 2>/dev/null || inq $J; } && continue
    MEM=128G bash scripts/race_sbatch.sh $J 04:00:00 $ENV bash eval/facts_scorer/cascade_job.sh \
      $O $k $K "$TV" $M/ce12B_full $F r59d --save_sub=r64_${SPLIT:0:1}_$S > /dev/null
  done
done
echo "R64_SUBMITTED $SPLIT $* $(date)"
