#!/bin/bash
# RESULTS 61b: 27B labelling of one shard of the extra photos (inside scripts/race_sbatch.sh; FP8 GPU).
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
k=$1; K=$2
LOG=eval/facts_writer/logs/label_$k.log
bash scripts/prewarm_env.sh vllm $FP_ROOT/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B-FP8 > $LOG.warm 2>&1
bash scripts/keepwarm.sh vllm 600 &
KW=$!
export VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID" >> $LOG
python eval/facts_writer/label_extra.py --shard=$k/$K >> $LOG 2>&1
echo "EXIT $? $(date)" >> $LOG
kill $KW 2>/dev/null
