#!/bin/bash
# RESULTS 55: dump.py for one category + user list (CPU). Usage: bash eval/complex_bench/b55/dump_one.sh <cat> <users,comma>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
mkdir -p eval/complex_bench/b55/logs
LOG=eval/complex_bench/b55/logs/dump_$1_one.log
bash scripts/prewarm_env.sh vllm $FP_ROOT/.cache/huggingface/hub/models--timm--PE-Core-L-14-336 > $LOG.warm 2>&1
python eval/complex_bench/b55/dump.py cat=$1 users=$2 >> $LOG 2>&1
echo "exit $? $(date)" >> $LOG
