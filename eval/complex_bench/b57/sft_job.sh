#!/bin/bash
# RESULTS 57 (track B3): planner SFT targets from the corrected TRAIN truths (queries/train.json as built for RESULTS 56:
# h2 / g2 / b2 programs, 643 requests, 24 libraries; never dev/test), target rules v2 (RESULTS 49 B2), teacher
# Qwen3.5-27B-FP8 (FP8: H100 / H200 / RTX PRO 6000 only). Product code for grounding + checks = the shipped code
# (.cache/wt_m54_s1/src = 64709ef src = HEAD src). Usage (inside scripts/race_sbatch.sh): bash .../sft_job.sh <out dir> [extra args]
OUT=$1; shift
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
mkdir -p $OUT
bash scripts/prewarm_env.sh vllm $FP_ROOT/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B-FP8 > $OUT/sft.log.warm 2>&1
bash scripts/keepwarm.sh vllm 900 &
export PYTHONPATH=$FP_ROOT/.cache/wt_m54_s1/src VLLM_WORKER_MULTIPROC_METHOD=spawn FP_MAX_SEQS=64
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID" >> $OUT/sft.log
python eval/complex_bench/sft_from_truth.py --queries=eval/complex_bench/queries/train.json --out=$OUT --rules=v2 "$@" >> $OUT/sft.log 2>&1
echo "SFT_EXIT $? $(date)" >> $OUT/sft.log
