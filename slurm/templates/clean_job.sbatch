#!/bin/bash
# Usage: SCRIPT="scripts/x.py args" sbatch --export=ALL slurm/templates/vlm_job.sbatch
#SBATCH -J fp_vlm
#SBATCH -p kempner
#SBATCH -A kempner_mzitnik_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH --exclude=holygpu8a19102
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/slurm/logs/%x_%j.out
source /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/env.sh
export HOME=$FP_ROOT/.cache/home; mkdir -p $HOME  # stray writes to ~ land inside find_pics
deactivate 2>/dev/null; source $FP_ROOT/envs/clean/bin/activate
cd $FP_ROOT; export PYTHONPATH=$FP_ROOT/src
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
eval "$SCRIPT"
