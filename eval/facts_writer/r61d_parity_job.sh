#!/bin/bash
# RESULTS 61d (2): real-MLX parity run on a GPU node (MLX CUDA backend, mlx[cuda12] 0.32.3 + mlx-vlm 0.7.4 in the Debian
# container tools/mlxpy/py312.sif, venv tools/mlxpy/venv_cuda; MLX's CPU backend took 1,815 s for one photo).
# Usage: bash eval/facts_writer/r61d_parity_job.sh <mlx dir> <photos json> <tag> [q4 dir for the HF-processor pixel dump]
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
R=$PWD; O=.cache/r61d/parity
if [ -n "$4" ] && [ ! -f .cache/r61d/pix_hf.npz ]; then
  envs/vllm/bin/python .cache/r61d/pix_dump.py hf $4 .cache/r61d/pix_hf.npz > .cache/r61d/pix_hf.log 2>&1
fi
for C in 100000 20; do
  CU=/n/sw/helmod-rocky8/apps/Core/cuda/12.9.1-fasrc01/cuda
  singularity exec --nv --cleanenv -B $R:$R -B $CU:$CU --env HF_HOME=$R/.cache/huggingface,CUDA_HOME=$CU,CUDA_PATH=$CU tools/mlxpy/py312.sif \
    tools/mlxpy/venv_cuda/bin/python eval/facts_writer/r61d_parity_mlx.py $1 $2 $O/mlx_$3_ctx$C.jsonl --ctx=$C > $O/mlx_$3_ctx$C.log 2>&1
  echo "EXIT $? ctx $C $(date)" >> $O/mlx_$3_ctx$C.log
done
