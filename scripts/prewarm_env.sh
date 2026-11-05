#!/bin/bash
# Work around the 10-12 Lustre slowdown on some nodes: every small-file read / page fault costs ~0.3-0.4 s (memory-
# mapped shared libraries page-fault at ~22 KB/s; `import torch` = thousands of .py/.pyc reads), while large sequential
# reads are fast and parallel reads scale. So: read every shared library, the Python binary, and every .py/.pyc of the
# env + stdlib once, in parallel (cat > /dev/null), which puts them in the node's page cache; later imports hit RAM.
# Extra args: more files/dirs read the same way (e.g. model weights, which vLLM / safetensors also mmap).
#   bash scripts/prewarm_env.sh [env_name=vllm] [dir_or_file ...]
ROOT=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
E=${1:-vllm}; shift
S=$ROOT/envs/$E/lib/python3.12/site-packages
U=$(dirname $(dirname $(readlink -f $ROOT/envs/$E/bin/python)))
t0=$(date +%s)
cat $(readlink -f $ROOT/envs/$E/bin/python) > /dev/null
find $S $U/lib -name "*.so*" -type f -print0 | xargs -0 -P 8 -n 20 cat > /dev/null
echo "  libs $(( $(date +%s)-t0 ))s"
find $S $U/lib/python3.12 $ROOT/src \( -name "*.py" -o -name "*.pyc" -o -name "*.json" -o -name "*.pth" \) -type f -print0 \
  | xargs -0 -P 64 -n 50 cat > /dev/null
echo "  py $(( $(date +%s)-t0 ))s"
for d in "$@"; do find -L "$d" -type f \( -name "*.safetensors" -o -name "*.bin" -o -name "*.json" -o -name "*.pt" -o -name "*.txt" -o -name "*.model" \) -print0 | xargs -0 -P 4 -n 4 cat > /dev/null; done
echo "prewarm $E $* $(( $(date +%s)-t0 ))s on $(hostname)"
