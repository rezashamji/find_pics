#!/bin/bash
# RESULTS 61b: when every label shard is done, submit the six 4-GPU student trainings and start the checkpoint
# evaluation waiters (phone form + facts -> towervec -> R59c cascade). Background, login node.
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source env.sh
until [ $(grep -l LABEL_DONE eval/facts_writer/logs/label_*.log 2>/dev/null | wc -l) -ge 24 ]; do sleep 60; done
R=$FP_ROOT
export SPECS="kempner_requeue:gpu:nvidia_h100_80gb_hbm3:4 kempner_rtx:gpu:4 kempner_h100:gpu:4 kempner_h200:gpu:4"
for s in "4b qwen3vl_4b_mlx4sim" "2b qwen3vl_2b_mlx4sim" "08 qwen35_08b_mlx4sim"; do
  set -- $s
  for h in "A 0,1" "B 2,3"; do
    set -- $1 $2 $h
    MEM=240G bash scripts/race_sbatch.sh fp_r61b_t$1_$3 12:00:00 bash eval/facts_writer/train2_job.sh t$1_$3 4 \
      --base=$R/models/$2 --folds=$4 --extra=1 --epochs=2 --save_at=100000,200000,400000 --out=eval/facts_writer/models/t$1_$3 > /dev/null
  done
done
echo "TRAIN_SUBMITTED $(date)"
CK4="seen_100000 seen_200000 seen_400000 final"; CK2="seen_100000 seen_400000 final"; CK08="final"
for sz in 4b 2b 08; do
  case $sz in 4b) CK=$CK4; B=$R/models/qwen3vl_4b_mlx4sim;; 2b) CK=$CK2; B=$R/models/qwen3vl_2b_mlx4sim;; 08) CK=$CK08; B=$R/models/qwen35_08b_mlx4sim;; esac
  for c in $CK; do
    for h in A B; do (bash eval/facts_writer/launch_ckpt.sh t${sz}_$h $c $B > eval/facts_writer/logs/launch_ckpt_t${sz}_${h}_$c.log 2>&1 &); done
    (bash eval/facts_writer/launch_writer.sh t${sz}_${c}_xfit > eval/facts_writer/logs/launch_writer_t${sz}_$c.log 2>&1 &)
  done
done
echo "WAITERS_STARTED $(date)"
