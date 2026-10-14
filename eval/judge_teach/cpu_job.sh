#!/bin/bash
# RESULTS 58: run a CPU step on the 'test' partition (account mzitnik_lab) with the vLLM env.
# Usage: sbatch -p test -A mzitnik_lab -c 8 --mem=64G -t 2:00:00 -J <name> -o <log> eval/judge_teach/cpu_job.sh "<command>"
ROOT=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
cd $ROOT
source $ROOT/env.sh
export HOME=$FP_ROOT/.cache/home PYTHONPATH=$FP_ROOT/src
deactivate 2>/dev/null; source $ROOT/envs/vllm/bin/activate
echo "== $(date) $(hostname) $1"
eval "$1"
echo "EXIT $? $(date)"
