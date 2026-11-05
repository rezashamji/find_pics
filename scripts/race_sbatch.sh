#!/bin/bash
# Submit the SAME job to every GPU partition we can use; the first copy to start cancels its siblings (by job name).
# Why: kempner partitions share the lab's 96-GPU cap (MaxGRESPerAccount) and multi-partition -p is refused for them;
# kempner_requeue has no account QoS cap (preemptible: jobs get requeued, so jobs must be restartable).
# Which copy runs is decided by scripts/race_guard.sh with an atomic lock (mkdir), not by comparing squeue states
# (that raced on 10-03: two copies cancelled each other). Runs inside the job on the compute node: no login session needed.
# GPU types are named so the 9B judge never lands on a 20 GB MIG slice.
# Usage: scripts/race_sbatch.sh <unique_job_name> <time> <command...>     (vLLM env; run from the repo root)
set -e
NAME=$1; T=$2; shift 2; CMD="$*"
ROOT=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
PRE="cd $ROOT; bash $ROOT/scripts/race_guard.sh $NAME || exit 0; source $ROOT/env.sh; export HOME=\$FP_ROOT/.cache/home; deactivate 2>/dev/null; source $ROOT/envs/vllm/bin/activate; export PYTHONPATH=\$FP_ROOT/src;"
# a copy on the preemptible partition keeps the regular copies queued as backup (it can be preempted); a regular copy
# cancels everything else; whichever finishes cancels any copies still queued (POST).
POST="; rc=\$?; for j in \$(squeue -h -u \$USER -n $NAME -o %i); do [ \"\$j\" != \"\$SLURM_JOB_ID\" ] && scancel \$j; done; exit \$rc"
A100=kempner_requeue:gpu:nvidia_a100-sxm4-40gb:1; RTX=kempner_requeue:gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
SPECS=${SPECS:-"$A100 $RTX"}
# Reza 11-05: find_pics must not slow his research. kempner, kempner_rtx, kempner_h100 and kempner_h200 share ONE QoS
# (kempner_base: 16 GPUs per user, 96 per lab account), so any find_pics GPU there takes a slot from his jobs. Only
# kempner_requeue has its own limits (64 GPUs per user, no lab cap; preemptible, cheapest billing 418 vs A100 836.5 /
# RTX 675.6 / H100-H200 2648.8 per GPU). So every copy goes to kempner_requeue, keeping the caller's GPU type
# (kempner -> requeue A100-40GB, kempner_rtx -> requeue RTX PRO 6000), and H100/H200 are dropped - unless FP_ALLOW_BASE=1.
if [ "${FP_ALLOW_BASE:-0}" != 1 ]; then
  keep=""
  for s in $SPECS; do
    case "$s" in
      *h100*|*h200*) echo "race_sbatch: dropping $s (no H100/H200, Reza 11-05)" >&2; continue;;
      kempner_rtx:*) s=$RTX;;
      kempner:*)     s=$A100;;
      kempner_requeue:*) ;;
      *) ;;
    esac
    case " $keep " in *" $s "*) ;; *) keep="$keep $s";; esac
  done
  SPECS=$keep
  if [ -z "${SPECS// /}" ]; then SPECS="$A100 $RTX"; fi
fi
# SPECS="partition:gres ..." overrides the list, e.g. H100/H200 only for FP8 (A100 has no FP8 matmul)
for spec in $SPECS; do spec="${spec%%:*} ${spec#*:}"
  set -- $spec
  extra=""; [ "$1" = kempner_requeue ] && extra="--requeue"
  sbatch --parsable -p $1 -A ${ACCT:-kempner_mzitnik_lab} --gres=$2 -c 8 --mem=${MEM:-96G} -t $T --exclude=holygpu8a19102 $extra \
    -J $NAME -o $ROOT/slurm/logs/${NAME}_%j.out --wrap "$PRE $CMD $POST" | tr '\n' ' '
done
echo
