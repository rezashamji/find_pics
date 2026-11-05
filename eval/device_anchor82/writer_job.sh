#!/bin/bash
# M82: one MLX writer shard on a GPU node (inside scripts/race_sbatch.sh), both writers in sequence (each loaded once).
# MLX CUDA backend in the Debian container tools/mlxpy/py312.sif + venv tools/mlxpy/venv_cuda (as RESULTS 61d parity).
# Usage: bash eval/device_anchor82/writer_job.sh <k> <K> <out dir> <manifest[,manifest]> [models spec]
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
R=$PWD
k=$1; K=$2; OUT=$3; MAN=$4
BASE=$(ls -d $R/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-2B-Instruct-4bit/snapshots/*/ | head -1)
TRAINED=$R/eval/facts_writer/models/p2b_all_final_mlx
MODELS=${5:-trained=$TRAINED,base=$BASE}
mkdir -p $OUT
LOG=$OUT/writer_$k.log
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID" >> $LOG
# page cache first (Lustre page faults are slow): container image, the venv's libraries and code, both checkpoints
t0=$(date +%s)
cat tools/mlxpy/py312.sif > /dev/null
find tools/mlxpy/venv_cuda \( -name "*.so*" -o -name "*.py" -o -name "*.pyc" -o -name "*.json" \) -type f -print0 | xargs -0 -P 32 -n 50 cat > /dev/null
find -L $BASE $TRAINED -type f -print0 | xargs -0 -P 4 -n 4 cat > /dev/null
echo "prewarm $(( $(date +%s)-t0 ))s" >> $LOG
CU=/n/sw/helmod-rocky8/apps/Core/cuda/12.9.1-fasrc01/cuda
singularity exec --nv --cleanenv -B $R:$R -B $CU:$CU \
  --env HF_HOME=$R/.cache/huggingface,HF_HUB_OFFLINE=1,CUDA_HOME=$CU,CUDA_PATH=$CU,PYTHONPATH=$R/src tools/mlxpy/py312.sif \
  tools/mlxpy/venv_cuda/bin/python eval/device_anchor82/writer_mlx.py --models=$MODELS --manifest=$MAN --shard=$k/$K \
  --out=$OUT >> $LOG 2>&1
echo "EXIT $? $(date)" >> $LOG
