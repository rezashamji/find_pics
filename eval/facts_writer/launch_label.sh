#!/bin/bash
# RESULTS 61b: when all fetch + dHash shards exist, dedupe and submit the 27B labelling (24 shards). Background.
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source env.sh
until [ $(ls data/public/extra_photos/shards/*.parquet 2>/dev/null | wc -l) -ge 16 ] && [ $(ls data/public/extra_photos/disbench_dhash/*.parquet 2>/dev/null | wc -l) -ge 4 ]; do sleep 30; done
python eval/facts_writer/extra_data.py dedupe
export SPECS="kempner_requeue:gpu:nvidia_h100_80gb_hbm3:1 kempner_rtx:gpu:1 kempner_h100:gpu:1 kempner_h200:gpu:1"
for k in $(seq 0 23); do MEM=128G bash scripts/race_sbatch.sh fp_r61b_label_$k 06:00:00 bash eval/facts_writer/label_job.sh $k 24 > /dev/null; done
echo "LABEL_SUBMITTED $(date)"
