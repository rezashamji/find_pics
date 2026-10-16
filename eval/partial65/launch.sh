#!/bin/bash
# RESULTS 65 launcher: bash eval/partial65/launch.sh <train|dev>   (one shard per library; background, sbatch only)
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source env.sh
SPLIT=$1; [ $SPLIT = train ] && K=24 || K=13
export SPECS="kempner_requeue:gpu:nvidia_h100_80gb_hbm3:1 kempner_rtx:gpu:1 kempner_h100:gpu:1 kempner_h200:gpu:1"   # no A100-40GB (lazy 9B)
inq() { squeue -u $USER -h -o %j | grep -qx "$1"; }
for k in $(seq 0 $((K-1))); do
  J=fp_r65_${SPLIT:0:1}_$k
  { grep -q CASCADE_DONE eval/partial65/replay/$SPLIT/cascade_$k.log 2>/dev/null || inq $J; } && continue
  MEM=128G bash scripts/race_sbatch.sh $J 08:00:00 bash eval/partial65/job.sh $SPLIT $k $K > /dev/null
done
echo "R65_SUBMITTED $SPLIT $(date)"
