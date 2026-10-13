#!/bin/bash
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
bash scripts/prewarm_env.sh vllm $FP_ROOT/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B > .cache/cb51/probe_warm.log 2>&1
bash scripts/keepwarm.sh vllm 900 &
for u in $(cat eval/complex_bench/eh_diag/dev_users.txt); do find data/public/raw/disbench/images/images/$u -type f -print0; done | xargs -0 -P 64 -n 40 cat > /dev/null
export PYTHONPATH=$FP_ROOT/.cache/cb51/snap/src VLLM_WORKER_MULTIPROC_METHOD=spawn
python eval/complex_bench/eh_diag/probe_absent.py > eval/complex_bench/eh_diag/probe_absent.log 2>&1
echo "exit $?" >> eval/complex_bench/eh_diag/probe_absent.log
