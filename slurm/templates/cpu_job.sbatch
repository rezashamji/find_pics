#!/bin/bash
# CPU-only helper job (no model): SCRIPT="..." via --export, eval'd from the repo root.
# Usage: sbatch --export=ALL,SCRIPT="python scripts/replay_pairing.py keys" slurm/templates/cpu_job.sbatch
#SBATCH -J fp_cpu
#SBATCH -p gpu_test
#SBATCH -A mzitnik_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/slurm/logs/%x_%j.out
source /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/env.sh
export HOME=$FP_ROOT/.cache/home; mkdir -p $HOME
cd $FP_ROOT
export PYTHONPATH=$FP_ROOT/src
eval "$SCRIPT"
