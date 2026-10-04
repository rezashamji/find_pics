#!/bin/bash
# Reza's AirDropped sample (data/private/sample, 620 files, his OK 10-04) end to end, as ONE GPU job so it survives any
# window closing: dedupe -> scan -> index -> face sheet -> provisional "me" (most frequent face; Reza confirms) ->
# demo + everyday searches. Everything stays under data/private/ (chmod 700, never committed). Never deletes a photo:
# duplicates are skipped by linking only the unique files into a separate folder.
# Usage (from the repo root): scripts/race_sbatch.sh fp_sample 04:00:00 "bash scripts/run_sample.sh"
set -e
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
P=data/private; LIB=$P/sample_lib; IDX=$P/index_sample; OUT=$P/sample_runs; mkdir -p $OUT $P/audits
MAIN="deactivate 2>/dev/null || true; source $PWD/envs/fp/bin/activate"
VLLM="deactivate 2>/dev/null || true; source $PWD/envs/vllm/bin/activate"
export PYTHONPATH=$PWD/src

echo "[1/5] dedupe (links to unique files)"; date
eval "$MAIN"
python - <<'EOF'
import hashlib, os
from pathlib import Path
src, dst = Path("data/private/sample"), Path("data/private/sample_lib")
dst.mkdir(exist_ok=True); seen = {}; dup = 0
for f in sorted(src.iterdir(), key=lambda p: (" " in p.stem, p.name)):    # originals before "name 2.JPG" copies
    if not f.is_file() or f.name.startswith("."):
        continue
    h = hashlib.md5(f.read_bytes()).hexdigest()
    if h in seen:
        dup += 1; continue
    seen[h] = f.name
    link = dst / f.name.replace(" ", "_")
    if not link.exists():
        os.symlink(os.path.abspath(f), link)
print(f"unique {len(seen)}, duplicates skipped {dup}")
EOF
chmod -R go-rwx $P

echo "[2/5] scan + index"; date
python -m findpics.cli scan $LIB $IDX
rm -rf $IDX/shards
python -m findpics.index $IDX --shard 0 --n-shards 1 --workers 8
python scripts/plumbing_report.py $IDX --sizes 1000 --out $P/audits/sample_plumbing.json

echo "[3/5] face sheet + provisional me (largest group; Reza confirms)"; date
python -m findpics.cli people $IDX --top 12
# keep Reza's own pick (groups 1+2, 10-04) across re-indexing: named_people.json stores his face fingerprints
grep -q '"Reza"' $IDX/named_people.json 2>/dev/null || python -m findpics.cli name $IDX "Reza" 1

echo "[4/5] searches"; date
eval "$VLLM"
run() {   # one conversation per request (a new --out folder), models load once per request
  local name="$1"; shift
  rm -rf "$OUT/$name"
  printf '%s\n' "$@" | python -m findpics.cli chat --index $IDX --out "$OUT/$name" --me "Reza" --today 2026-10-04 \
    > "$OUT/$name.log" 2>&1 || echo "FAILED $name"
  grep -E "^\[round|^Album '|I don't know|Could not" "$OUT/$name.log" | tail -8
}
run demo "photos and videos of me looking heavier vs photos and videos of me looking fit"
run food "food photos"
run night "photos at night"
run vids "videos of me"
run screens "screenshots"
run outdoors "photos of me outdoors"
run people "photos of me with other people"
echo "[5/5] done"; date; chmod -R go-rwx $P
echo SAMPLE_DONE
