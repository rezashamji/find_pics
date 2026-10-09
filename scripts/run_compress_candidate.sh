#!/bin/bash
# One weight-compression candidate for the phone judge (Qwen3-VL-4B, MLX 4-bit), end to end on one GPU (10-09):
#   1. fake-quantize (scripts/sim_mlx_quant.py <flags>)            -> models/q3vl4b_<tag>   (skipped if done)
#   2. eye-label judge test, same code + labels as RESULTS 32-33  -> .cache/snap_jd/eval/judge_distill_test/q3vl4b_<tag>.json
#   3. Reza's heavier-vs-fit demo, shipped planner, prob mode      -> data/private/sample_runs/mode_q3vl4b_<tag>.score
# Usage (inside race_sbatch.sh, vLLM env): bash scripts/run_compress_candidate.sh <tag> [sim flags...]
#   tag "base_rerun" with no flags re-tests the existing baseline checkpoint (run-to-run noise).
set -e
ROOT=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
cd $ROOT
TAG=$1; shift
M=$ROOT/models/q3vl4b_$TAG
[ "$TAG" = base_rerun ] && M=$ROOT/models/qwen3vl_4b_mlx4sim
mkdir -p eval/compress
if [ ! -f $M/sim_quant.json ] && [ "$TAG" != base_rerun ]; then
  python scripts/sim_mlx_quant.py Qwen/Qwen3-VL-4B-Instruct mlx-community/Qwen3-VL-4B-Instruct-4bit $M "$@" \
    > eval/compress/sim_$TAG.log 2>&1
fi
J=.cache/snap_jd/eval/judge_distill_test/q3vl4b_$TAG.json
if [ ! -f $J ]; then
  (cd .cache/snap_jd && PYTHONPATH=$ROOT/.cache/snap_jd/src python eval/eval_judge_distill.py gen q3vl4b_$TAG $M) \
    > eval/compress/judge_$TAG.log 2>&1
fi
export PYTHONPATH=$ROOT/.cache/snap_2b/src VLLM_WORKER_MULTIPROC_METHOD=spawn FP_MAX_SEQS=128 FP_JUDGE_MODE=prob \
  FP_VLM_MODEL=$M FP_PLANNER_MODEL=$ROOT/models/planner_4b27ball_q4merged
bash scripts/run_demo_mode.sh q3vl4b_$TAG > data/private/sample_runs/mode_q3vl4b_$TAG.score 2>&1
chmod go-rwx data/private/sample_runs/mode_q3vl4b_$TAG.score
echo DONE $TAG
