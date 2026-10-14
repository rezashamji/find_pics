#!/bin/bash
# RESULTS 59 fix B: when all 24 TRAIN libraries have planner-question oracle answers, build the pairs and submit
# the fix-B trainings (facts + place/month metadata, extra pairs). Run in the background from the login node.
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source env.sh
until [ $(ls eval/facts_scorer/data/oracle_plan/*.parquet 2>/dev/null | wc -l) -ge 24 ]; do sleep 30; done
python eval/facts_scorer/build_trainplan.py
export SPECS="kempner_requeue:gpu:nvidia_h100_80gb_hbm3:1 kempner_requeue:gpu:nvidia_a100-sxm4-40gb:1 kempner_rtx:gpu:1 kempner_h100:gpu:1 kempner:gpu:1"
M=eval/facts_scorer/models; X="--facts=eval/facts_scorer/data/facts_27bm.parquet --extra=eval/facts_scorer/data/pairs_trainplan.parquet"
for v in h0 full h1 h2 h3; do
  H=""; P="--predict=dev,eye"; [ $v != full ] && H="--hold=${v#h}" && P="--predict=hold,eye"; [ $v = h0 ] && P="--predict=hold,dev,eye"
  MEM=128G bash scripts/race_sbatch.sh fp_r59_ce12B_$v 05:00:00 bash eval/facts_scorer/train_job.sh ce12B_$v --kind=ce --base=cross-encoder/ms-marco-MiniLM-L12-v2 --epochs=1 $H $P $X --out=$M/ce12B_$v > /dev/null
  MEM=128G bash scripts/race_sbatch.sh fp_r59_tv12B_$v 05:00:00 bash eval/facts_scorer/train_job.sh towervec12B_$v --kind=towervec --base=sentence-transformers/all-MiniLM-L12-v2 --steps=10000 $H $P $X --out=$M/towervec12B_$v > /dev/null
done
echo "FIXB_SUBMITTED $(date)"
