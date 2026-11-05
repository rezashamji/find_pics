#!/bin/bash
# M82 job wrapper: fill the node's page cache first (scripts/prewarm_env.sh: Lustre page faults on mmapped libraries and
# safetensors run at ~150 KB/s on slow days; a parallel sequential read takes seconds), then run the command.
# Usage: bash eval/device_anchor82/run.sh <env> <dir,dir,...|-> <command...>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
E=$1; P=$2; shift 2
export VLLM_WORKER_MULTIPROC_METHOD=spawn     # vLLM engine core: CUDA cannot be re-initialised in a forked child
[ "$P" = - ] && P=""
bash scripts/prewarm_env.sh $E ${P//,/ } 2>&1 | tail -1
exec "$@"
