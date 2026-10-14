#!/bin/bash
# RESULTS 57 (track B3): retrain the 4B planner LoRA, RESULTS 31 recipe (r16/alpha32, lr 1e-4, 2 epochs, accum 8), on
# data/public/distill/planner_cbmix3.jsonl (b57/make_mix.py: the 4,426 old 27B distillation train rows + the B3 rows x3,
# as B2), then merge it into the phone's 4-bit weights. Resumable (train_planner_lora.py checkpoints every 50 steps).
# fla kernels from the overlay (.cache/fla_overlay; 4 s/step on an RTX PRO 6000 in RESULTS 49).
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
L=eval/complex_bench/runs/cmp/train_cb3.log
bash scripts/prewarm_env.sh vllm $FP_ROOT/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B > $L.warm 2>&1
bash scripts/keepwarm.sh vllm 900 &
export PYTHONPATH=$FP_ROOT/.cache/fla_overlay:$FP_ROOT/src
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID" >> $L
if [ ! -f models/planner_4bcb3_merged/config.json ]; then
  FP_DISTILL_FILE=data/public/distill/planner_cbmix3.jsonl FP_DISTILL_TAG=cb3 python eval/train_planner_lora.py --epochs 2 >> $L 2>&1 || exit 1
fi
[ -f models/planner_4bcb3_q4merged/config.json ] || python scripts/merge_lora_into.py models/qwen35_4b_mlx4sim models/planner_4bcb3_lora models/planner_4bcb3_q4merged >> $L 2>&1
echo "TRAIN_CB3_DONE $(date)" >> $L
