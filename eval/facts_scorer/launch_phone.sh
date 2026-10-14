#!/bin/bash
# RESULTS 59 step 2: when a phone facts set is complete (70,580 TRAIN+DEV photos), build its facts text (+ place/month
# metadata, salvage of truncated JSON) and train the fix-B scorer configuration on it (same as ce12B/towervec12B_full).
# Usage (login node, background): bash eval/facts_scorer/launch_phone.sh <tag>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source env.sh
T=$1
until [ $(cat eval/facts_scorer/phone_facts/$T/*.jsonl 2>/dev/null | wc -l) -ge 70580 ]; do sleep 60; done
python eval/facts_scorer/data.py --facts=eval/facts_scorer/phone_facts/$T --tag=${T}m --meta=1 --salvage=1 --pairs=0
until [ -f eval/facts_scorer/data/pairs_trainplan.parquet ]; do sleep 60; done
export SPECS="kempner_requeue:gpu:nvidia_h100_80gb_hbm3:1 kempner_requeue:gpu:nvidia_a100-sxm4-40gb:1 kempner_rtx:gpu:1 kempner_h100:gpu:1 kempner:gpu:1"
M=eval/facts_scorer/models; X="--facts=eval/facts_scorer/data/facts_${T}m.parquet --extra=eval/facts_scorer/data/pairs_trainplan.parquet"
MEM=128G bash scripts/race_sbatch.sh fp_r59_ce12B_$T 05:00:00 bash eval/facts_scorer/train_job.sh ce12B_$T --kind=ce --base=cross-encoder/ms-marco-MiniLM-L12-v2 --epochs=1 --predict=dev $X --out=$M/ce12B_$T > /dev/null
MEM=128G bash scripts/race_sbatch.sh fp_r59_tv12B_$T 05:00:00 bash eval/facts_scorer/train_job.sh towervec12B_$T --kind=towervec --base=sentence-transformers/all-MiniLM-L12-v2 --steps=10000 --predict=dev $X --out=$M/towervec12B_$T > /dev/null
echo "PHONE_SUBMITTED $T $(date)"
