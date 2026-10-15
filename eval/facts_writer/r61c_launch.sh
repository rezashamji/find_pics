#!/bin/bash
# RESULTS 61c: re-evaluate the EXISTING short-line writers under the R59d cascade (eval/facts_scorer route.py / cascade.py
# as of 500ade1, variant r59d; r59c re-run alongside as a reproduction check). No retraining: each writer's facts and its
# own millisecond scorers from RESULTS 61 / 61b (TRAIN fold models, full model for dev). Usage: bash r61c_launch.sh
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source env.sh
M=eval/facts_scorer/models; D=eval/facts_scorer/data; TP=eval/facts_scorer/replay/train_plans/plans_r60.json
export SPECS="kempner_requeue:gpu:nvidia_h100_80gb_hbm3:1 kempner_requeue:gpu:nvidia_a100-sxm4-40gb:1 kempner_rtx:gpu:1 kempner_h100:gpu:1 kempner:gpu:1"
for T in short27b t4b_final_xfit t2b_final_xfit s4b_xfit; do
  F=$D/facts_w_${T}m.parquet
  for k in $(seq 0 11); do
    MEM=128G bash scripts/race_sbatch.sh fp_r61c_t_${T}_$k 04:00:00 env SPLIT=train PLANS=$TP bash eval/facts_scorer/cascade_job.sh \
      eval/facts_writer/replay/c61c_train_$T $k 12 "$M/tvw_${T}_h{fold}" $M/ce12B_full $F r59c,r59d --save_sub=r61c_t_$T > /dev/null
  done
  for k in $(seq 0 12); do
    MEM=128G bash scripts/race_sbatch.sh fp_r61c_d_${T}_$k 04:00:00 bash eval/facts_scorer/cascade_job.sh \
      eval/facts_writer/replay/c61c_dev_$T $k 13 $M/tvw_${T}_full $M/ce12B_full $F r59c,r59d --save_sub=r61c_d_$T > /dev/null
  done
done
echo "R61C_SUBMITTED $(date)"
