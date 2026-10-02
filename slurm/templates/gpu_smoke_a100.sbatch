#!/bin/bash
#SBATCH -J fp_smoke_a100
#SBATCH -p kempner
#SBATCH -A kempner_mzitnik_lab
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH --exclude=holygpu8a19102
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/slurm/logs/%x_%j.out
source /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/env.sh
export HOME=$FP_ROOT/.cache/home; mkdir -p $HOME  # stray writes to ~ land inside find_pics
cd $FP_ROOT; export PYTHONPATH=$FP_ROOT/src
nvidia-smi
curl -s -o /dev/null -w "internet from compute node: %{http_code}\n" https://huggingface.co
python scripts/smoke.py
