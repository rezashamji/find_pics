#!/bin/bash
# RESULTS 48: export both heads to the phone format + Swift golden fixtures (after ship_gpu.sh's training / rescoring).
set -e
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
bash scripts/prewarm_env.sh vllm > /dev/null
bash scripts/keepwarm.sh vllm 600 &
KW=$!
rm -f eval/learned_head/scale/ship_export.txt
for R in JDonly JDonly-na; do
  python eval/learned_head/head_ship.py export $R 0 eval/learned_head/scale/head_${R}_s0.fphead
  HEAD_NAME=learned_head.fphead python eval/learned_head/head_ship.py fixture eval/learned_head/scale/head_${R}_s0.fphead eval/learned_head/scale/fixture_${R}.json
done
kill $KW 2>/dev/null || true
echo SHIP_EXPORT_DONE
