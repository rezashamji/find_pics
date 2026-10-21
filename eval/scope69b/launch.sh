#!/bin/bash
# RESULTS 69b launcher: the RESULTS 65/66 partial-coverage curve (0/10/25/50%, newest-first, policy V) under the adopted
# DRY scope at fp32. Same harness + job as RESULTS 69 (eval/scope69/{cascade69.py,job.sh}); configs DRY_fp32_<cov>.
#   bash eval/scope69b/launch.sh <train|dev>   (one shard per library; sbatch only, background)
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source env.sh
SPLIT=$1; [ $SPLIT = train ] && K=24 || K=13
export SPECS="kempner_requeue:gpu:nvidia_h100_80gb_hbm3:1 kempner_rtx:gpu:1 kempner_h100:gpu:1 kempner_h200:gpu:1"   # no A100-40GB (lazy 9B)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # 10-16 17:55: first launch hung (19/19 jobs idle 18 min in a futex wait with a CLOSE-WAIT socket to a CloudFront :443 host)
export CONFIGS=${CONFIGS:-DRY_fp32_0,DRY_fp32_10,DRY_fp32_25,DRY_fp32_50}
export OUTDIR=${OUTDIR:-eval/scope69b/replay/$SPLIT} RECDIR=${RECDIR:-data/public/scope69b/record/$SPLIT}
inq() { squeue -u $USER -h -o %j | grep -qx "$1"; }
for k in $(seq 0 $((K-1))); do
  J=fp_r69b_${SPLIT:0:1}_$k
  { grep -q CASCADE_DONE $OUTDIR/cascade_$k.log 2>/dev/null || inq $J; } && continue
  MEM=128G bash scripts/race_sbatch.sh $J 08:00:00 bash eval/scope69/job.sh $SPLIT $k $K > /dev/null
done
echo "R69B_SUBMITTED $SPLIT $(date)"
