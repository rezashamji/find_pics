#!/bin/bash
# RESULTS 61d (3): end-to-end evaluation of REPEAT-STOPPED facts <tag> (made by r61d_repeat.py from t2b_final_xfit):
# facts parquet -> millisecond scorer (towervec, same recipe as launch_writer.sh) retrained on these facts (full + fold
# models h0..h3) -> the R59d cascade (variant r59d, code as in 61c) on TRAIN (fold scorers) and dev (full scorer).
# SEED=k (env, default 0): scorer seed; k != 0 names the run <tag>_seed<k> (placebo: scorer-retraining noise on the same facts).
# Background, login node (only sbatch + waits). Usage: [SEED=1] bash eval/facts_writer/r61d_launch_rs.sh <tag>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source env.sh
T=$1; SEED=${SEED:-0}; R=$T; [ $SEED != 0 ] && R=${T}_seed$SEED; M=eval/facts_scorer/models; D=eval/facts_scorer/data; TP=eval/facts_scorer/replay/train_plans/plans_r60.json
FD=eval/facts_writer/facts/$T
[ -f $D/facts_w_${T}m.parquet ] || python eval/facts_scorer/data.py --facts=$FD --tag=w_${T}m --meta=1 --salvage=0 --pairs=0
F=$D/facts_w_${T}m.parquet
export SPECS="kempner_requeue:gpu:nvidia_h100_80gb_hbm3:1 kempner_rtx:gpu:1 kempner_h100:gpu:1 kempner_h200:gpu:1"   # no A100-40GB: the lazily loaded 9B judge fails engine init there (61d: Mamba cache blocks)
X="--extra=$D/pairs_trainplan.parquet --facts=$F"
inq() { squeue -u $USER -h -o %j | grep -qx "$1"; }
for v in full h0 h1 h2 h3; do
  H=""; [ $v != full ] && H="--hold=${v#h}"
  { [ -f $M/tvw_${R}_$v/weights.pt ] || inq fp_r61d_tv_${R}_$v; } && continue
  MEM=128G bash scripts/race_sbatch.sh fp_r61d_tv_${R}_$v 05:00:00 bash eval/facts_scorer/train_job.sh tvw_${R}_$v --kind=towervec \
    --base=sentence-transformers/all-MiniLM-L12-v2 --steps=10000 $H --predict=none $X --seed=$SEED --out=$M/tvw_${R}_$v > /dev/null
done
until [ -f $M/tvw_${R}_full/weights.pt ] && [ -f $M/tvw_${R}_h0/weights.pt ] && [ -f $M/tvw_${R}_h1/weights.pt ] && \
      [ -f $M/tvw_${R}_h2/weights.pt ] && [ -f $M/tvw_${R}_h3/weights.pt ]; do sleep 60; done
for k in $(seq 0 11); do
  { grep -q CASCADE_DONE eval/facts_writer/replay/c61d_train_$R/cascade_$k.log 2>/dev/null || inq fp_r61d_t_${R}_$k; } && continue
  MEM=128G bash scripts/race_sbatch.sh fp_r61d_t_${R}_$k 04:00:00 env SPLIT=train PLANS=$TP bash eval/facts_scorer/cascade_job.sh \
    eval/facts_writer/replay/c61d_train_$R $k 12 "$M/tvw_${R}_h{fold}" $M/ce12B_full $F r59d --save_sub=r61d_t_$R > /dev/null
done
for k in $(seq 0 12); do
  { grep -q CASCADE_DONE eval/facts_writer/replay/c61d_dev_$R/cascade_$k.log 2>/dev/null || inq fp_r61d_d_${R}_$k; } && continue
  MEM=128G bash scripts/race_sbatch.sh fp_r61d_d_${R}_$k 04:00:00 bash eval/facts_scorer/cascade_job.sh \
    eval/facts_writer/replay/c61d_dev_$R $k 13 $M/tvw_${R}_full $M/ce12B_full $F r59d --save_sub=r61d_d_$R > /dev/null
done
echo "R61D_RS_SUBMITTED $R $(date)"
