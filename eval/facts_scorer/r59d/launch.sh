#!/bin/bash
# RESULTS 59d: submit the cascade replays (GPU via race_sbatch; CPU partitions unavailable 10-15, see JOURNAL).
# Usage: bash eval/facts_scorer/r59d/launch.sh <4b|27b>
#   4b : TRAIN (fold scorers, 12 shards) r59c,r59d,r59d_a,r59d_b,r59d_c ; dev (13 shards) the same + aj59d,aj0,aj59c,ajb
#   27b: TRAIN + dev r59c,r59d (reference)
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source env.sh
F=$1; M=eval/facts_scorer/models; D=eval/facts_scorer/data
export SPECS="kempner_requeue:gpu:nvidia_h100_80gb_hbm3:1 kempner_requeue:gpu:nvidia_a100-sxm4-40gb:1 kempner_rtx:gpu:1 kempner_h100:gpu:1 kempner:gpu:1"
TP=eval/facts_scorer/replay/train_plans/plans_r60.json
if [ $F = 4b ]; then
  TV_T="$M/towervec12B_q3vl4bq4r_h{fold}"; TV_D=$M/towervec12B_q3vl4bq4r; CE=$M/ce12B_q3vl4bq4r; FX=$D/facts_q3vl4bq4rm.parquet
  VT=r59c,r59d,r59d_a,r59d_b,r59d_c; VD=r59c,r59d,r59d_a,r59d_b,r59d_c,aj59d,aj0,aj59c,ajb
else
  TV_T="$M/towervec12B_h{fold}"; TV_D=$M/towervec12B_full; CE=$M/ce12B_full; FX=$D/facts_27bm.parquet
  VT=r59c,r59d; VD=r59c,r59d
fi
for k in $(seq 0 11); do
  MEM=128G bash scripts/race_sbatch.sh fp_r59d_t${F}_$k 04:00:00 env SPLIT=train PLANS=$TP bash eval/facts_scorer/cascade_job.sh \
    eval/facts_scorer/replay/c59d_train_$F $k 12 "$TV_T" $CE $FX $VT --save_sub=r59d_t$F > /dev/null
done
for k in $(seq 0 12); do
  MEM=128G bash scripts/race_sbatch.sh fp_r59d_d${F}_$k 04:00:00 bash eval/facts_scorer/cascade_job.sh \
    eval/facts_scorer/replay/c59d_dev_$F $k 13 $TV_D $CE $FX $VD --save_sub=r59d_d$F > /dev/null
done
echo "R59D_SUBMITTED $F $(date)"
