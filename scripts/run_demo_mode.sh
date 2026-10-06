#!/bin/bash
# Reza's heavier-vs-fit demo with a judge mode (prob | hard | rating) and model, scored against his labels.
# hard/rating simulate a judge without probabilities (Apple Foundation Models). Private outputs only.
# Usage (vLLM env, GPU): FP_JUDGE_MODE=rating FP_VLM_MODEL=... bash scripts/run_demo_mode.sh <tag>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
OUT=data/private/sample_runs/mode_$1; rm -rf $OUT
echo "photos and videos of me looking heavier vs photos and videos of me looking fit" | python -m findpics.cli chat \
  --index data/private/index_sample --out $OUT --me "Reza" --today 2026-10-04 > $OUT.log 2>&1 || echo FAILED
chmod -R go-rwx data/private/sample_runs
python scripts/score_demo.py $OUT
