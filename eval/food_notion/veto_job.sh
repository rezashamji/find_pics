#!/bin/bash
# GPU job: eval/eval_food_notion.py gen (shop-front veto wordings, phone-weight judge) inside scripts/race_sbatch.sh
# (vLLM env already active). Lustre workaround: prewarm env + model, keep warm during startup.
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
LOG=eval/food_notion/veto_gen.log
bash scripts/prewarm_env.sh vllm $FP_ROOT/models/q3vl4b_real > $LOG.warm 2>&1
find data/public/index_disbench -type f -print0 | xargs -0 -P 8 -n 4 cat > /dev/null
bash scripts/keepwarm.sh vllm 900 &
echo "host $(hostname) start $(date)" > $LOG
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> $LOG
python -u eval/eval_food_notion.py gen >> $LOG 2>&1
rc=$?
echo "exit $rc $(date)" >> $LOG
exit $rc
