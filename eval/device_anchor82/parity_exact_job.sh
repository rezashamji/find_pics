#!/bin/bash
# M82 (B') parity-exact job: build the exact PNGs, check the processor, trained writer K = 48 (MLX container, one GPU).
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
R=$PWD; LOG=data/public/device_anchor82/logs/parity_exact.log
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID" >> $LOG
cat tools/mlxpy/py312.sif > /dev/null
find tools/mlxpy/venv_cuda \( -name "*.so*" -o -name "*.py" -o -name "*.pyc" -o -name "*.json" \) -type f -print0 | xargs -0 -P 32 -n 50 cat > /dev/null
find -L eval/facts_writer/models/p2b_all_final_mlx -type f -print0 | xargs -0 -P 4 -n 4 cat > /dev/null
CU=/n/sw/helmod-rocky8/apps/Core/cuda/12.9.1-fasrc01/cuda
# MLX on CUDA segfaults now and then (exit 139); the script resumes from parity_exact/rows.jsonl, so retry
for try in 1 2 3 4 5 6; do
  singularity exec --nv --cleanenv -B $R:$R -B $CU:$CU \
    --env HF_HOME=$R/.cache/huggingface,HF_HUB_OFFLINE=1,CUDA_HOME=$CU,CUDA_PATH=$CU tools/mlxpy/py312.sif \
    tools/mlxpy/venv_cuda/bin/python eval/device_anchor82/parity_exact.py >> $LOG 2>&1
  rc=$?; echo "EXIT $rc try $try $(date)" >> $LOG
  [ $rc = 0 ] && break
done
