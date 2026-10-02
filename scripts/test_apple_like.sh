#!/bin/bash
# End-to-end test of the Apple export path on a synthetic, public-data export (HEIC, previews, HEVC .mov, metadata).
# Job A (main env, GPU): build export -> scan -> index.   Job B (vLLM env, after A): two asks.
set -e
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
A=$(sbatch --parsable -p kempner -A kempner_mzitnik_lab --gres=gpu:1 -c 8 --mem=64G -t 01:00:00 --exclude=holygpu8a19102 \
  -J fp_applelike -o slurm/logs/%x_%j.out --wrap 'cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics; source /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/env.sh; export HOME=$FP_ROOT/.cache/home; export PYTHONPATH=$FP_ROOT/src;
  rm -rf data/public/apple_like_export data/public/index_apple_like;
  python scripts/build_apple_like.py &&
  python -m findpics.cli scan data/public/apple_like_export data/public/index_apple_like --metadata data/public/apple_like_export/library_metadata.json &&
  python -m findpics.index data/public/index_apple_like --shard 0 --n-shards 1 --workers 6')
Q1="Find every photo and video of Kevin Bacon."
Q2="Find every photo and video of Kevin Bacon where he looks heavier or out of shape, and the best photos and videos of him from 2010 to 2015 where he looks fit. Make two albums: Bacon heavier, and Bacon fit."
B=$(sbatch --parsable --dependency=afterok:$A -p kempner -A kempner_mzitnik_lab --gres=gpu:1 -c 8 --mem=64G -t 01:00:00 \
  --exclude=holygpu8a19102 -J fp_applelike_ask -o slurm/logs/%x_%j.out --wrap "cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics; source /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/env.sh; export HOME=\$FP_ROOT/.cache/home;
  deactivate 2>/dev/null; source /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/envs/vllm/bin/activate; export PYTHONPATH=\$FP_ROOT/src;
  python -m findpics.cli ask data/public/index_apple_like '$Q1' --out data/public/albums_apple_like_q1 --me 'Kevin Bacon' --today 2026-10-02 &&
  python -m findpics.cli ask data/public/index_apple_like '$Q2' --out data/public/albums_apple_like_q2 --me 'Kevin Bacon' --today 2026-10-02")
echo "A=$A B=$B"
