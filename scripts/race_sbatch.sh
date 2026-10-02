#!/bin/bash
# Submit the SAME job to every GPU partition we can use; the first copy to start cancels its siblings (by job name).
# Why: kempner partitions share the lab's 96-GPU cap (MaxGRESPerAccount) and multi-partition -p is refused for them;
# kempner_requeue has no account QoS cap (preemptible: jobs get requeued, so jobs must be restartable).
# GPU types are named so the 9B judge never lands on a 20 GB MIG slice.
# Usage: scripts/race_sbatch.sh <unique_job_name> <time> <command...>     (vLLM env; run from the repo root)
set -e
NAME=$1; T=$2; shift 2; CMD="$*"
ROOT=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
PRE="cd $ROOT; source $ROOT/env.sh; export HOME=\$FP_ROOT/.cache/home; deactivate 2>/dev/null; source $ROOT/envs/vllm/bin/activate; export PYTHONPATH=\$FP_ROOT/src;
for j in \$(squeue -h -u \$USER -n $NAME -o %i); do [ \"\$j\" != \"\$SLURM_JOB_ID\" ] && scancel \$j; done;"
for spec in "kempner_requeue gpu:nvidia_h100_80gb_hbm3:1" "kempner_requeue gpu:nvidia_a100-sxm4-40gb:1" \
            "kempner_rtx gpu:1" "kempner_h100 gpu:1" "kempner_h200 gpu:1" "kempner gpu:1"; do
  set -- $spec
  extra=""; [ "$1" = kempner_requeue ] && extra="--requeue"
  sbatch --parsable -p $1 -A kempner_mzitnik_lab --gres=$2 -c 8 --mem=96G -t $T --exclude=holygpu8a19102 $extra \
    -J $NAME -o $ROOT/slurm/logs/${NAME}_%j.out --wrap "$PRE $CMD" | tr '\n' ' '
done
echo
