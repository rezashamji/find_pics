#!/bin/bash
# RESULTS 48 GPU steps (one gpu_test job): train the no-|img-txt| variant + a JDonly re-run that keeps every seed,
# rescore beach with the RESULTS 47 question (+ int8-simulated text tower).
# Run inside a job with the vllm env active (eval/learned_head/gtest.sh NAME TIME "bash eval/learned_head/ship_gpu.sh").
set -e
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
bash scripts/prewarm_env.sh vllm .cache/huggingface/hub/models--timm--PE-Core-B-16
bash scripts/keepwarm.sh vllm 1200 &
KW=$!
python eval/learned_head/train_scale.py JDonly-na JDonly-r
python eval/learned_head/head_ship.py rescore JDonly JDonly-r JDonly-na
# exports + Swift fixtures: eval/learned_head/ship_export.sh (separate job)
kill $KW 2>/dev/null || true
echo SHIP_GPU_DONE
