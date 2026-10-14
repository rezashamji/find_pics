#!/bin/bash
# RESULTS 61: after student <name> (s4b_A ...) finishes training, submit its phone-form + facts job. Background, login node.
# A students write TRAIN folds {2,3} + all dev libraries; B students write TRAIN folds {0,1}.
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source env.sh
N=$1; B=$2
until grep -q "TRAIN_DONE" eval/facts_writer/logs/$N.log 2>/dev/null; do sleep 60; done
U=$(python - <<PY
import json
sp=json.load(open('eval/complex_bench/splits.json')); tr=sorted(sp['train'])
f={u:k for k in range(4) for u in tr[k::4]}
half='$N'.split('_')[-1]
us=[u for u in tr if f[u] in ((2,3) if half=='A' else (0,1))] + (sp['dev'] if half=='A' else [])
print(','.join(us))
PY
)
export SPECS="kempner_requeue:gpu:nvidia_h100_80gb_hbm3:1 kempner_requeue:gpu:nvidia_a100-sxm4-40gb:1 kempner_rtx:gpu:1 kempner_h100:gpu:1 kempner:gpu:1"
MEM=160G bash scripts/race_sbatch.sh fp_r61_w_$N 05:00:00 bash eval/facts_writer/student_job.sh $N $B $U > /dev/null
echo "STUDENT_JOB_SUBMITTED $N $(date)"
