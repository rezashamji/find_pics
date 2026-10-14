#!/bin/bash
# RESULTS 58: the phone judge's two simulations on the eye set, inside scripts/race_sbatch.sh (one GPU):
#   phone_real = models/q3vl4b_real (RESULTS 38: the REAL mlx-community 4-bit language tensors de-quantized, vision bf16)
#   phone_v8   = models/q3vl4b_v8   (same language tensors + the vision tower at 8-bit, MLX affine group 64 = the tower
#                the app ships since M36/M38; built here with scripts/sim_mlx_quant.py --from-mlx --vision-bits 8)
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
mkdir -p eval/judge_teach/logs
if [ ! -f models/q3vl4b_v8/sim_quant.json ]; then
  python scripts/sim_mlx_quant.py Qwen/Qwen3-VL-4B-Instruct mlx-community/Qwen3-VL-4B-Instruct-4bit models/q3vl4b_v8 \
    --from-mlx --vision-bits 8 > eval/judge_teach/logs/sim_v8.log 2>&1
fi
bash eval/judge_teach/eye_job.sh phone_real $FP_ROOT/models/q3vl4b_real 1
bash eval/judge_teach/eye_job.sh phone_v8 $FP_ROOT/models/q3vl4b_v8 1
