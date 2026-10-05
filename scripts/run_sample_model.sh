#!/bin/bash
# Same searches as run_sample.sh on Reza's sample, with a different judge/planner model (phone candidates).
# Usage (vLLM env, GPU): FP_VLM_MODEL=Qwen/Qwen3.5-4B bash scripts/run_sample_model.sh <tag>
# Outputs: data/private/sample_runs/<tag>_<name>/ (private, never committed).
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
TAG=$1; IDX=data/private/index_sample; OUT=data/private/sample_runs
run() {
  local name="${TAG}_$1"; shift
  rm -rf "$OUT/$name"
  printf '%s\n' "$@" | python -m findpics.cli chat --index $IDX --out "$OUT/$name" --me "Reza" --today 2026-10-04 \
    > "$OUT/$name.log" 2>&1 || echo "FAILED $name"
  grep -E "^\[round|^Album '|judge_question" "$OUT/$name.log" | tail -6
}
echo "model: $FP_VLM_MODEL"
run demo "photos and videos of me looking heavier vs photos and videos of me looking fit"
run food "food photos"
run night "photos at night"
run screens "screenshots"
run outdoors "photos of me outdoors"
chmod -R go-rwx $OUT
echo MODEL_RUN_DONE
