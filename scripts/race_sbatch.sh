#!/bin/bash
# Submit the SAME job to every GPU partition we can use; the first copy to start cancels its siblings (by job name).
# Why: kempner partitions share the lab's 96-GPU cap (MaxGRESPerAccount) and multi-partition -p is refused for them;
# kempner_requeue has no account QoS cap (preemptible: jobs get requeued, so jobs must be restartable).
# Two copies starting in the same pass: the LOWER job id wins (the higher one sees it running and exits), so they
# can never cancel each other. The cancel runs inside the job on the compute node: no login session needed.
# GPU types are named so the 9B judge never lands on a 20 GB MIG slice.
# Usage: scripts/race_sbatch.sh <unique_job_name> <time> <command...>     (vLLM env; run from the repo root)
set -e
NAME=$1; T=$2; shift 2; CMD="$*"
ROOT=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
PRE="cd $ROOT; source $ROOT/env.sh; export HOME=\$FP_ROOT/.cache/home; deactivate 2>/dev/null; source $ROOT/envs/vllm/bin/activate; export PYTHONPATH=\$FP_ROOT/src;
for x in \$(squeue -h -u \$USER -n $NAME -o %i:%t:%P); do j=\${x%%:*}; r=\${x#*:}; st=\${r%%:*}; part=\${r#*:}; [ \"\$j\" = \"\$SLURM_JOB_ID\" ] && continue;
  if [ \"\$SLURM_JOB_PARTITION\" = kempner_requeue ] && [ \"\$part\" != kempner_requeue ]; then [ \"\$st\" = R ] && { echo \"regular sibling \$j running: exiting\"; exit 0; }; continue; fi;
  if [ \"\$st\" = R ] && [ \"\$j\" -lt \"\$SLURM_JOB_ID\" ] && [ \"\$part\" != kempner_requeue ]; then echo \"sibling \$j already running: exiting\"; exit 0; fi;
  scancel \$j; done;"
# a copy on the preemptible partition keeps the regular copies queued as backup (it can be preempted); a regular copy
# cancels everything else; whichever finishes cancels any copies still queued (POST).
POST="; for j in \$(squeue -h -u \$USER -n $NAME -o %i); do [ \"\$j\" != \"\$SLURM_JOB_ID\" ] && scancel \$j; done"
SPECS=${SPECS:-"kempner_requeue:gpu:nvidia_h100_80gb_hbm3:1 kempner_requeue:gpu:nvidia_a100-sxm4-40gb:1 kempner_rtx:gpu:1 kempner_h100:gpu:1 kempner_h200:gpu:1 kempner:gpu:1"}
# SPECS="partition:gres ..." overrides the list, e.g. H100/H200 only for FP8 (A100 has no FP8 matmul)
for spec in $SPECS; do spec="${spec%%:*} ${spec#*:}"
  set -- $spec
  extra=""; [ "$1" = kempner_requeue ] && extra="--requeue"
  sbatch --parsable -p $1 -A kempner_mzitnik_lab --gres=$2 -c 8 --mem=96G -t $T --exclude=holygpu8a19102 $extra \
    -J $NAME -o $ROOT/slurm/logs/${NAME}_%j.out --wrap "$PRE $CMD $POST" | tr '\n' ' '
done
echo
