#!/bin/bash
# RESULTS 59: when a facts set's two fix-B scorers are trained, make their equal-weight ensemble and submit the dev
# end-to-end replays (ce, towervec, ensemble; product wording as asked, no FP_R59_QNORM: these models trained on it).
# Usage (login node, background): bash eval/facts_scorer/launch_replays.sh <suffix: full | q3vl4bq4r | ...> <facts parquet>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
S=$1; F=$2; M=eval/facts_scorer/models
until [ -f $M/ce12B_$S/pred_dev.parquet ] && [ -f $M/towervec12B_$S/pred_dev.parquet ]; do sleep 60; done
mkdir -p $M/ensB_$S
echo "{\"kind\": \"ens\", \"members\": [\"$M/ce12B_$S\", \"$M/towervec12B_$S\"], \"weights\": [0.5, 0.5], \"base\": \"\", \"facts\": \"$F\"}" > $M/ensB_$S/config.json
export SPECS="kempner_requeue:gpu:nvidia_h100_80gb_hbm3:1 kempner_requeue:gpu:nvidia_a100-sxm4-40gb:1 kempner_rtx:gpu:1 kempner_h100:gpu:1 kempner:gpu:1 kempner_requeue:gpu:nvidia_a100_3g.20gb:1"
for m in ce12B_$S towervec12B_$S ensB_$S; do for k in 0 1 2 3; do
  MEM=96G bash scripts/race_sbatch.sh fp_r59_rB_${m}_$k 03:00:00 bash eval/facts_scorer/replay_job.sh eval/facts_scorer/replay/$m $k 4 --scorer=$M/$m --facts=$F > /dev/null
done; done
echo "REPLAYS_SUBMITTED $S $(date)"
