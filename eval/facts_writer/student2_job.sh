#!/bin/bash
# RESULTS 61b: phone form of one student CHECKPOINT + its cross-fit facts (inside scripts/race_sbatch.sh).
# Usage: bash eval/facts_writer/student2_job.sh <student dir name, e.g. t4b_A> <checkpoint, e.g. seen_100000> <base dir> <users>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
N=$1; C=$2; B=$3; U=$4
LOG=eval/facts_writer/logs/${N}_${C}_write.log
bash scripts/prewarm_env.sh vllm > $LOG.warm 2>&1
bash scripts/keepwarm.sh vllm 600 &
KW=$!
export VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID" >> $LOG
Q=eval/facts_writer/models/${N}_${C}_q4
[ -f $Q/config.json ] || python eval/facts_writer/merge_q4.py $B eval/facts_writer/models/$N/$C $Q >> $LOG 2>&1
TAG=${N%_*}_${C}_xfit
python eval/facts_writer/write_facts.py --model=$Q --tag=$TAG --users=$U >> $LOG 2>&1
echo "EXIT $? $(date)" >> $LOG
kill $KW 2>/dev/null
