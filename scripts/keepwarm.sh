#!/bin/bash
# 10-12 Lustre: pages read by prewarm_env.sh get dropped again after a while (client lock LRU), and a Python/vLLM
# startup that then page-faults on them crawls at ~100 KB/s. Re-read the env every 15 s for SECONDS (default 900) while
# the job starts up (already-cached reads cost ~seconds). Run in the background before the python command:
#   bash scripts/keepwarm.sh vllm 900 & python ...
E=${1:-vllm}; T=${2:-900}
end=$(( $(date +%s) + T ))
while [ $(date +%s) -lt $end ]; do
  bash /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/scripts/prewarm_env.sh $E > /dev/null 2>&1
  sleep 15
done
