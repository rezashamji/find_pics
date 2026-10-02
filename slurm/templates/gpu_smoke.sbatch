#!/bin/bash
#SBATCH -J fp_smoke
#SBATCH -p kempner_h100
#SBATCH -A kempner_mzitnik_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH --exclude=holygpu8a19102
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/slurm/logs/%x_%j.out
source /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/env.sh
cd $FP_ROOT; export PYTHONPATH=$FP_ROOT/src
nvidia-smi
curl -s -o /dev/null -w "internet from compute node: %{http_code}\n" https://huggingface.co
python scripts/smoke.py
