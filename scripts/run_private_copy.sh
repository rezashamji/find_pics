#!/bin/bash
# Reza's library from Apple's "copy of your data" zips -> index -> PLUMBING REPORT ONLY (no searches: holdout, 10-03).
# Put the downloaded zips in data/private/apple_copy_zips/ (chmod 700 folder). Run from the repo root, in the background:
#   bash scripts/run_private_copy.sh          (Claude: run_in_background; each step waits for its Slurm jobs)
# Steps: 1 ingest zips (photos -> 1600 px JPEG + EXIF, videos -> 720p; skips Recently Deleted / deleted / shared albums;
#        never deletes anything)  2 scan  3 index on 16 GPU shards (race_sbatch, main env)  4 plumbing report.
set -e
ROOT=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics; cd $ROOT
Z=${Z:-data/private/apple_copy_zips}; C=${C:-data/private/apple_copy}; IDX=${IDX:-data/private/index}; K=${K:-16}
REPORT=${REPORT:-data/private/audits/plumbing_report.json}   # (dry runs on public data override all of these)
[ -d "$Z" ] && ls $Z/*.zip >/dev/null 2>&1 || { echo "no zips in $Z"; exit 1; }
chmod 700 data/private
wait_jobs() { while squeue -h -u $USER -o %j | grep -q "$1"; do sleep 60; done; }
MAIN="deactivate 2>/dev/null; source \$FP_ROOT/envs/fp/bin/activate;"

echo "[1/4] ingest $(ls $Z/*.zip | wc -l) zips"; date
for k in $(seq 0 $((K-1))); do      # zips split across K jobs (video re-encoding is the slow part)
  scripts/race_sbatch.sh fp_priv_ingest_s$k 12:00:00 "$MAIN python -m findpics.cli apple-copy $Z $C --shard $k --n-shards $K" >/dev/null
done
wait_jobs '^fp_priv_ingest_s'
echo "[2/4] scan"; date
scripts/race_sbatch.sh fp_priv_scan 04:00:00 "$MAIN python -m findpics.cli scan $C/library $IDX" >/dev/null
wait_jobs '^fp_priv_scan$'
echo "[3/4] index on $K shards"; date
for k in $(seq 0 $((K-1))); do
  scripts/race_sbatch.sh fp_priv_idx_s$k 06:00:00 "$MAIN python -m findpics.index $IDX --shard $k --n-shards $K --workers 6" >/dev/null
done
wait_jobs '^fp_priv_idx_s'
echo "[4/4] plumbing report (data/private/audits/plumbing_report.json)"; date
source env.sh; PYTHONPATH=src python scripts/plumbing_report.py $IDX --sizes 3000 --out $REPORT
echo PRIVATE_PLUMBING_DONE; date
