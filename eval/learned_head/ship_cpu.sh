#!/bin/bash
# RESULTS 48 CPU replay step: bash eval/learned_head/ship_cpu.sh SETTING TAG SOURCE...  (env: SR_PER_SEARCH=SOURCE)
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
bash scripts/prewarm_env.sh fp > /dev/null 2>&1
bash scripts/keepwarm.sh fp 300 &
KW=$!
ST=$1; TAG=$2; shift 2
SR_TAG=$TAG python eval/learned_head/ship_replay.py $ST "$@"
kill $KW 2>/dev/null || true
