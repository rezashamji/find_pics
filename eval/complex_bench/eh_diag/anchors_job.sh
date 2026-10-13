#!/bin/bash
# RESULTS 51 step 1: anchor-stage diagnostics on a compute node (vLLM env, cached 9B judge only; src snapshot = HEAD + WIP)
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
bash scripts/prewarm_env.sh vllm > .cache/cb51/anchors_warm.log 2>&1
for u in $(cat eval/complex_bench/eh_diag/dev_users.txt); do find data/public/raw/disbench/images/images/$u -type f -print0; done | xargs -0 -P 64 -n 40 cat > /dev/null
export PYTHONPATH=$FP_ROOT/.cache/cb51/snap/src
python eval/complex_bench/eh_diag/anchors.py all > eval/complex_bench/eh_diag/anchors.log 2>&1
echo "exit $?" >> eval/complex_bench/eh_diag/anchors.log
