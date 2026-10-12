#!/bin/bash
# Submit one M48-scale job to gpu_test (-A mzitnik_lab; 20 GB GPUs, CPU-light / small-model work), racing the same
# NAME submitted by scripts/race_sbatch.sh (same lock). Usage: bash eval/learned_head/gtest.sh NAME TIME "command"
ROOT=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
NAME=$1; T=$2; CMD=$3
sbatch --parsable -p gpu_test -A mzitnik_lab --gres=gpu:1 -c 8 --mem=60G -t $T -J $NAME -o $ROOT/slurm/logs/${NAME}_%j.out \
  --wrap "cd $ROOT; bash $ROOT/scripts/race_guard.sh $NAME || exit 0; source $ROOT/env.sh; export HOME=\$FP_ROOT/.cache/home; deactivate 2>/dev/null; source $ROOT/envs/vllm/bin/activate; export PYTHONPATH=\$FP_ROOT/src; $CMD; rc=\$?; for j in \$(squeue -h -u \$USER -n $NAME -o %i); do [ \"\$j\" != \"\$SLURM_JOB_ID\" ] && scancel \$j; done; exit \$rc"
