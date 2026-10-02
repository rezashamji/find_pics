#!/bin/bash
# Usage: sbatch --array=0-$((K-1))%16 --export=ALL,IDX=<index_dir>,K=<n_shards> slurm/templates/index_array.sbatch
#SBATCH -J fp_index
#SBATCH -p kempner_h100
#SBATCH -A kempner_mzitnik_lab
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/slurm/logs/%x_%A_%a.out
source /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/env.sh
cd $FP_ROOT
export PYTHONPATH=$FP_ROOT/src
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m findpics.index "$IDX" --shard $SLURM_ARRAY_TASK_ID --n-shards $K --workers 12
