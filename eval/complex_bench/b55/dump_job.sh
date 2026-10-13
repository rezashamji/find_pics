#!/bin/bash
# RESULTS 55: per-photo evidence dump (dump.py) for one shard of the dev users of a category. CPU work (cache-only judge).
# Usage (inside scripts/race_sbatch.sh): bash eval/complex_bench/b55/dump_job.sh <cat> <k> <K>
CAT=$1; k=$2; K=$3
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
mkdir -p eval/complex_bench/b55/logs
LOG=eval/complex_bench/b55/logs/dump_${CAT}_$k.log
bash scripts/prewarm_env.sh vllm $FP_ROOT/.cache/huggingface/hub/models--timm--PE-Core-L-14-336 > $LOG.warm 2>&1
USERS=$(python -c "
import json
Q=[q for q in json.load(open('eval/complex_bench/queries/dev.json'))['queries'] if q['cat']=='$CAT']
U=sorted({q['user'] for q in Q})[$k::$K]; print(','.join(U))")
for u in ${USERS//,/ }; do find data/public/raw/disbench/images/images/$u -type f -print0; done | xargs -0 -P 64 -n 40 cat > /dev/null
echo "users $USERS $(date)" >> $LOG
python eval/complex_bench/b55/dump.py cat=$CAT users=$USERS >> $LOG 2>&1
echo "exit $? $(date)" >> $LOG
