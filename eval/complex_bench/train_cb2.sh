#!/bin/bash
# B2 (v2 target rules, eval/complex_bench/sft_from_truth.py --rules=v2 + moment-question fix): same recipe.
# Track B (10-12): retrain the 4B planner LoRA (RESULTS 31 recipe: r16/alpha32, lr 1e-4, 2 epochs, accum 8) on the old
# 27B distillation rows + the complex-bench TRAIN rows (x3), then put the adapter on the phone's 4-bit weights.
# Resumable (eval/train_planner_lora.py checkpoints every 50 steps). fla kernels from an overlay (.cache/fla_overlay).
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
L=eval/complex_bench/runs/cmp/train_cb2.log
bash scripts/prewarm_env.sh vllm $FP_ROOT/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B > $L.warm 2>&1
bash scripts/keepwarm.sh vllm 900 &
export PYTHONPATH=$FP_ROOT/.cache/fla_overlay:$FP_ROOT/src
echo "host $(hostname) start $(date) job $SLURM_JOB_ID" >> $L
if [ ! -f models/planner_4bcb2_merged/config.json ]; then
  FP_DISTILL_FILE=data/public/distill/planner_cbmix2.jsonl FP_DISTILL_TAG=cb2 python eval/train_planner_lora.py --epochs 2 >> $L 2>&1 || exit 1
fi
python scripts/merge_lora_into.py models/qwen35_4b_mlx4sim models/planner_4bcb2_lora models/planner_4bcb2_q4merged >> $L 2>&1
echo "TRAIN_CB2_DONE $(date)" >> $L
