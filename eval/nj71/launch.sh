#!/bin/bash
# RESULTS 71 launcher (judge-free answers, NJ): the R69 harness (eval/scope69/{cascade69.py,job.sh}) with configs
#   DRY_fp32            = the RESULTS 69 / 70 phone configuration (control: returned sets must equal R69 / R70 DRY_fp32)
#   NJ_fp32_T<xx>       = NJ, global facts-P threshold xx/100 for judge-kind leaves (cascade69.py header)
#   NJ_fp32_T<xx>_F<k>  = NJ at writer cap K (RESULTS 70 facts parquets);  NJ_fp32_T<xx>_<cov> = NJ at coverage cov %
#   bash eval/nj71/launch.sh <stage> <train|dev>   stage s1 = control + the T grid; s2 = NJ at the TRAIN t (in-job control: must equal s1) + K caps + coverage
# One shard per library; GPU only through scripts/race_sbatch.sh -> kempner_requeue RTX PRO 6000 (the R70 hardware;
# no judge is loaded by NJ, the control reads the judge cache). Background only.
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source env.sh
STAGE=$1; SPLIT=$2; [ $SPLIT = train ] && K=24 || K=13
[ $SPLIT = train ] && SRC=t2b_final_rs2_xfit || SRC=p2b_devmix_rs2
export SPECS=${SPECS:-"kempner_requeue:gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1 kempner_requeue:gpu:nvidia_a100-sxm4-40gb:1"}   # 11-06: A100 sibling added (RTX queue slow); 45-min limit (jobs take ~5 min) so backfill can start them
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [ $STAGE = s1 ]; then
  G=$(for t in 05 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95; do printf ",NJ_fp32_T$t"; done)
  export CONFIGS=${CONFIGS:-DRY_fp32$G}
else
  T=$(cat eval/nj71/threshold.txt | awk '{print $1}')
  [ -n "$T" ] || { echo "no eval/nj71/threshold.txt"; exit 1; }
  export CONFIGS=${CONFIGS:-NJ_fp32_T${T},NJ_fp32_T${T}_Fk16,NJ_fp32_T${T}_Fk24,NJ_fp32_T${T}_Fk32,NJ_fp32_T${T}_Fk48,NJ_fp32_T${T}_Fk64,NJ_fp32_T${T}_0,NJ_fp32_T${T}_10,NJ_fp32_T${T}_25,NJ_fp32_T${T}_50}
fi
export OUTDIR=eval/nj71/replay/$STAGE/$SPLIT RECDIR=data/public/nj71/record/$STAGE/$SPLIT
FV="--fvar=eval/facts_scorer/data/facts_w_r70_${SRC}_{tag}m.parquet"
inq() { squeue -u $USER -h -o %j | grep -qx "$1"; }
for k in $(seq 0 $((K-1))); do
  J=fp_r71${STAGE}_${SPLIT:0:1}_$k
  { grep -q CASCADE_DONE $OUTDIR/cascade_$k.log 2>/dev/null || inq $J; } && continue
  MEM=96G bash scripts/race_sbatch.sh $J 00:45:00 bash eval/scope69/job.sh $SPLIT $k $K "$FV" > /dev/null
  # 11-06 (coordinator): requeue stuck on Priority -> also a copy on gpu_test (account mzitnik_lab, billing 0, MIG A100 20 GB,
  # 2 running jobs per user; NJ loads no judge); same job name, so race_guard keeps exactly one copy
  ACCT=mzitnik_lab SPECS="gpu_test:gpu:1" MEM=60G bash scripts/race_sbatch.sh $J 00:45:00 bash eval/scope69/job.sh $SPLIT $k $K "$FV" > /dev/null
done
echo "R71_SUBMITTED $STAGE $SPLIT $(date)"
