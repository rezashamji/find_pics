#!/bin/bash
# RESULTS 61b: for student <N> (t4b_A ...) and checkpoint <C>: wait for the adapter, submit its phone-form + facts job.
# A students write TRAIN folds {2,3} + dev; B students write TRAIN folds {0,1}. Background, login node.
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source env.sh
N=$1; C=$2; B=$3
until [ -f eval/facts_writer/models/$N/$C/adapter_config.json ] && [ -f eval/facts_writer/models/$N/$C/adapter_model.safetensors ]; do sleep 60; done
sleep 30
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
MEM=160G bash scripts/race_sbatch.sh fp_r61b_w_${N}_$C 05:00:00 bash eval/facts_writer/student2_job.sh $N $C $B $U > /dev/null
echo "CKPT_JOB_SUBMITTED $N $C $(date)"
