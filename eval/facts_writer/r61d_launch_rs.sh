#!/bin/bash
# RESULTS 61d (3): end-to-end evaluation of REPEAT-STOPPED facts <tag> (made by r61d_repeat.py from t2b_final_xfit):
# facts parquet -> millisecond scorer (towervec, same recipe as launch_writer.sh) retrained on these facts (full + fold
# models h0..h3) -> the R59d cascade (variant r59d, code as in 61c) on TRAIN (fold scorers) and dev (full scorer).
# Background, login node (only sbatch + waits). Usage: bash eval/facts_writer/r61d_launch_rs.sh <tag>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source env.sh
T=$1; M=eval/facts_scorer/models; D=eval/facts_scorer/data; TP=eval/facts_scorer/replay/train_plans/plans_r60.json
FD=eval/facts_writer/facts/$T
[ -f $D/facts_w_${T}m.parquet ] || python eval/facts_scorer/data.py --facts=$FD --tag=w_${T}m --meta=1 --salvage=0 --pairs=0
F=$D/facts_w_${T}m.parquet
export SPECS="kempner_requeue:gpu:nvidia_h100_80gb_hbm3:1 kempner_requeue:gpu:nvidia_a100-sxm4-40gb:1 kempner_rtx:gpu:1 kempner_h100:gpu:1 kempner:gpu:1"
X="--extra=$D/pairs_trainplan.parquet --facts=$F"
inq() { squeue -u $USER -h -o %j | grep -qx "$1"; }
for v in full h0 h1 h2 h3; do
  H=""; [ $v != full ] && H="--hold=${v#h}"
  { [ -f $M/tvw_${T}_$v/weights.pt ] || inq fp_r61d_tv_${T}_$v; } && continue
  MEM=128G bash scripts/race_sbatch.sh fp_r61d_tv_${T}_$v 05:00:00 bash eval/facts_scorer/train_job.sh tvw_${T}_$v --kind=towervec \
    --base=sentence-transformers/all-MiniLM-L12-v2 --steps=10000 $H --predict=none $X --out=$M/tvw_${T}_$v > /dev/null
done
until [ -f $M/tvw_${T}_full/weights.pt ] && [ -f $M/tvw_${T}_h0/weights.pt ] && [ -f $M/tvw_${T}_h1/weights.pt ] && \
      [ -f $M/tvw_${T}_h2/weights.pt ] && [ -f $M/tvw_${T}_h3/weights.pt ]; do sleep 60; done
for k in $(seq 0 11); do
  { grep -q CASCADE_DONE eval/facts_writer/replay/c61d_train_$T/cascade_$k.log 2>/dev/null || inq fp_r61d_t_${T}_$k; } && continue
  MEM=128G bash scripts/race_sbatch.sh fp_r61d_t_${T}_$k 04:00:00 env SPLIT=train PLANS=$TP bash eval/facts_scorer/cascade_job.sh \
    eval/facts_writer/replay/c61d_train_$T $k 12 "$M/tvw_${T}_h{fold}" $M/ce12B_full $F r59d --save_sub=r61d_t_$T > /dev/null
done
for k in $(seq 0 12); do
  { grep -q CASCADE_DONE eval/facts_writer/replay/c61d_dev_$T/cascade_$k.log 2>/dev/null || inq fp_r61d_d_${T}_$k; } && continue
  MEM=128G bash scripts/race_sbatch.sh fp_r61d_d_${T}_$k 04:00:00 bash eval/facts_scorer/cascade_job.sh \
    eval/facts_writer/replay/c61d_dev_$T $k 13 $M/tvw_${T}_full $M/ce12B_full $F r59d --save_sub=r61d_d_$T > /dev/null
done
echo "R61D_RS_SUBMITTED $T $(date)"
