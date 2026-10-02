#!/bin/bash
# Public test data -> data/public/raw (inside find_pics). Logs to slurm/logs/download_public.log
source /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/env.sh
cd $FP_ROOT/data/public/raw
HF=/n/home01/rshamji/.local/bin/hf
# IMDB full scene images (identity + photo year): first 24 shards (~5GB)
$HF download systemk-ai/imdb-wiki --repo-type dataset --include "imdb/train-0000[0-9]-*" "imdb/train-0001[0-9]-*" "imdb/train-0002[0-3]-*" --local-dir imdb_wiki || echo IMDB_FAIL
# CelebA aligned faces with identity + attributes (Chubby, Double_Chin)
$HF download flwrlabs/celeba --repo-type dataset --include "img_align+identity+attr/test-*" LICENSE README.md --local-dir celeba || echo CELEBA_FAIL
# Pexels videos (concept retrieval in video)
$HF download minh132/pexels-videos --repo-type dataset --local-dir pexels_videos || echo PEXELS_FAIL
echo DOWNLOAD_DONE
