#!/bin/bash
# RESULTS 61b: (re)start every checkpoint-evaluation waiter, detached; idempotent (safe after a session restart).
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
R=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
for sz in 4b 2b; do
  case $sz in 4b) CK="seen_100000 seen_200000 seen_400000 final"; B=$R/models/qwen3vl_4b_mlx4sim;; 2b) CK="seen_100000 seen_400000 final"; B=$R/models/qwen3vl_2b_mlx4sim;; esac
  for c in $CK; do
    for h in A B; do setsid nohup bash eval/facts_writer/launch_ckpt.sh t${sz}_$h $c $B > eval/facts_writer/logs/launch_ckpt_t${sz}_${h}_$c.log 2>&1 < /dev/null & done
    setsid nohup bash eval/facts_writer/launch_writer.sh t${sz}_${c}_xfit > eval/facts_writer/logs/launch_writer_t${sz}_$c.log 2>&1 < /dev/null &
  done
done
echo RELAUNCHED
