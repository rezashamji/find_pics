#!/bin/bash
# "selfies" on Reza's AirDropped sample (front-camera scope + selfie question + default look). Private outputs only.
# Usage (vLLM env, GPU): bash scripts/run_selfies_sample.sh
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
OUT=data/private/sample_runs/selfies_cam; rm -rf $OUT
echo "selfies" | python -m findpics.cli chat --index data/private/index_sample --out $OUT --me "Reza" --today 2026-10-06 > $OUT.log 2>&1 || echo FAILED
chmod -R go-rwx data/private/sample_runs
grep -E "^Album|Selfies|front|BACK" $OUT.log | head -8
