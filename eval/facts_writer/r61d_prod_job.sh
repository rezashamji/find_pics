#!/bin/bash
# RESULTS 61d (1)+(2): the PRODUCT 2B writer (p2b_all: all TRAIN + extra, 61b recipe) -> vLLM phone form (merge_q4.py),
# its dev facts (write_facts.py, as every writer), and the MLX checkpoint the app ships (to_mlx.py, checked against the
# phone form). Inside scripts/race_sbatch.sh, 1 GPU, vLLM env. Usage: bash eval/facts_writer/r61d_prod_job.sh <dev users csv>
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
LOG=eval/facts_writer/logs/p2b_all_write.log
bash scripts/prewarm_env.sh vllm > $LOG.warm 2>&1
bash scripts/keepwarm.sh vllm 600 &
KW=$!
export VLLM_WORKER_MULTIPROC_METHOD=spawn
B=$FP_ROOT/models/qwen3vl_2b_mlx4sim; A=eval/facts_writer/models/p2b_all/final; Q=eval/facts_writer/models/p2b_all_final_q4
REF=$(ls -d $FP_ROOT/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-2B-Instruct-4bit/snapshots/*)
echo "host $(hostname) start $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader) job $SLURM_JOB_ID" >> $LOG
[ -f $Q/config.json ] || python eval/facts_writer/merge_q4.py $B $A $Q >> $LOG 2>&1
python eval/facts_writer/write_facts.py --model=$Q --tag=p2b_all_dev --users=$1 >> $LOG 2>&1
[ -f eval/facts_writer/models/p2b_all_final_mlx/model.safetensors ] || \
  python eval/facts_writer/to_mlx.py $B $A eval/facts_writer/models/p2b_all_final_mlx --ref=$REF --check=$Q >> $LOG 2>&1
echo "EXIT $? $(date)" >> $LOG
kill $KW 2>/dev/null
