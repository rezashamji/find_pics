#!/bin/bash
# Run a command as an extra step inside one of my RUNNING GPU allocations (srun --overlap), with the vLLM env set up.
# Used on 10-12 when the queue was full: the old lh_img embed jobs (paused with SIGSTOP, so their allocation stays)
# lend their idle A100s to the pack-based steps. Usage: bash eval/learned_head/inalloc.sh JOBID LOG "command"
ROOT=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
J=$1; LOG=$2; CMD=$3
nohup srun --jobid=$J --overlap -n1 -c8 --mem=80G bash -c "cd $ROOT; source $ROOT/env.sh; export HOME=\$FP_ROOT/.cache/home; deactivate 2>/dev/null; source $ROOT/envs/vllm/bin/activate; export PYTHONPATH=\$FP_ROOT/src; export SLURM_CPUS_PER_TASK=8; echo start \$(hostname) \$(date); $CMD; echo exit \$? \$(date)" > $LOG 2>&1 &
