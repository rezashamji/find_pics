#!/bin/bash
# RESULTS 61: phone form of one student (LoRA merged, language re-quantized to MLX 4-bit) + its facts for the libraries
# it never trained on (inside scripts/race_sbatch.sh). Usage: bash eval/facts_writer/student_job.sh <name> <base dir> <users>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
N=$1; B=$2; U=$3
LOG=eval/facts_writer/logs/${N}_write.log
bash scripts/prewarm_env.sh vllm > $LOG.warm 2>&1
bash scripts/keepwarm.sh vllm 600 &
KW=$!
export VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID" >> $LOG
Q=eval/facts_writer/models/${N}_q4
[ -f $Q/config.json ] || python eval/facts_writer/merge_q4.py $B eval/facts_writer/models/$N/best $Q >> $LOG 2>&1
TAG=${N%_*}_xfit
python eval/facts_writer/write_facts.py --model=$Q --tag=$TAG --users=$U >> $LOG 2>&1
echo "EXIT $? $(date)" >> $LOG
kill $KW 2>/dev/null
