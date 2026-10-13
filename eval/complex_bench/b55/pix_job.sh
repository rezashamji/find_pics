#!/bin/bash
# RESULTS 55: pixel stats for one shard of a split's libraries (CPU). Usage: bash eval/complex_bench/b55/pix_job.sh <split> <k> <K>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
mkdir -p eval/complex_bench/b55/logs
LOG=eval/complex_bench/b55/logs/pix_$1_$2.log
bash scripts/prewarm_env.sh vllm > $LOG.warm 2>&1
python eval/complex_bench/b55/pix.py split=$1 k=$2 K=$3 >> $LOG 2>&1
echo "exit $? $(date)" >> $LOG
