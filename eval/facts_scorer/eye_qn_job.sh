#!/bin/bash
# RESULTS 59: eye-set predictions of every fold model with the question normalisation (FP_R59_QNORM=1).
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
LOG=eval/facts_scorer/logs/eye_qn.log
bash scripts/prewarm_env.sh vllm > $LOG.warm 2>&1
export FP_R59_QNORM=1
for m in ce12 towervec12 vec tower12; do for k in 0 1 2 3; do
  python eval/facts_scorer/predict.py --model=eval/facts_scorer/models/${m}_h$k --set=eye --name=_qn >> $LOG 2>&1
  echo "$m h$k $?" >> $LOG
done; done
python eval/facts_scorer/predict.py --model=eval/facts_scorer/models/tower12_h0 --set=eye >> $LOG 2>&1
echo "EXIT $(date)" >> $LOG
