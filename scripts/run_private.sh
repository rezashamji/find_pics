#!/bin/bash
# One command for Reza's library once the osxphotos export has landed in data/private/apple_export/.
# Everything (index, albums, review page, audits) stays under data/private/ (chmod 700, gitignored).
# Usage: bash scripts/run_private.sh "<Apple People name for Reza>" ["<request>"]
set -e
source /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/env.sh
cd $FP_ROOT; export PYTHONPATH=$FP_ROOT/src
ME="$1"
REQ="${2:-Find every photo and video of $ME where he looks heavier or out of shape, and the best photos and videos of him from the past 6 months where he looks fit and athletic. Make two albums: $ME heavier, and $ME fit.}"
EXP=${EXP:-data/private/apple_export}; IDX=${IDX:-data/private/index}; OUT=${OUT:-data/private/albums_$(date +%Y%m%d_%H%M)}
META=${META:-$EXP/library_metadata.json}; [ -f "$META" ] || META=""
python -m findpics.cli scan "$EXP" "$IDX" ${META:+--metadata $META}
rm -rf "$IDX/shards"   # fresh index for a fresh scan (shards are keyed by row order)
N=$(python -c "import pandas as pd; print(len(pd.read_parquet('$IDX/items.parquet')))")
K=$(( (N + 3999) / 4000 )); [ $K -gt 16 ] && K=16; [ $K -lt 1 ] && K=1
echo "items=$N shards=$K"
J1=$(IDX=$PWD/$IDX K=$K sbatch --parsable --export=ALL -p kempner -c 8 --mem=64G -t 02:00:00 --array=0-$((K-1)) slurm/templates/index_array.sbatch)
echo "index job $J1"
REQ_ESC=$(printf '%q' "$REQ")
J2=$(SCRIPT="-m findpics.cli ask $REQ_ESC --index $IDX --out $OUT --me $(printf '%q' "$ME")" sbatch --parsable --export=ALL --dependency=afterok:$J1 -J fp_ask -t 02:00:00 slurm/templates/vlm_job.sbatch)
echo "ask job $J2 (after $J1) -> $OUT"
