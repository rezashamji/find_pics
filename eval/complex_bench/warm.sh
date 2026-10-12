#!/bin/bash
# Warm this node's page cache with the vLLM env before Python starts (10-12: on loaded compute nodes, mmap page
# faults on Lustre read ~350 KB/s, so importing torch/vLLM took > 1 h; parallel sequential reads take ~1-2 min).
SP=$FP_ROOT/envs/vllm/lib/python3.12/site-packages
t0=$(date +%s)
find $SP -name "*.so*" -size +256k -print0 | xargs -0 -P 16 -n 4 cat > /dev/null
find $SP $FP_ROOT/.cache/uv-python -name "*.py" -o -name "*.pyc" -o -name "*.so*" -size -256k | xargs -P 32 -n 200 cat > /dev/null 2>&1
echo "warm.sh: page cache warmed in $(( $(date +%s) - t0 )) s"
