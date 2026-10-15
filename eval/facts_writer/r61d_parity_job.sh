#!/bin/bash
# RESULTS 61d (2): real-MLX (CPU) parity run on a Slurm node. Usage: bash eval/facts_writer/r61d_parity_job.sh <mlx dir> <photos json> <tag>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
R=$PWD; O=.cache/r61d/parity
for C in 100000 20; do
  singularity exec --cleanenv -B $R:$R --env HF_HOME=$R/.cache/huggingface,OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} tools/mlxpy/py312.sif \
    tools/mlxpy/venv/bin/python eval/facts_writer/r61d_parity_mlx.py $1 $2 $O/mlx_$3_ctx$C.jsonl --ctx=$C > $O/mlx_$3_ctx$C.log 2>&1
  echo "EXIT $? ctx $C $(date)" >> $O/mlx_$3_ctx$C.log
done
