#!/bin/bash
# Called at the start of every race_sbatch copy: decides, without races, whether THIS copy runs.
# Exit 0 = run; exit 1 = another copy owns the job, so this copy must stop.
# Why a lock: comparing squeue states is not atomic (10-03 03:00: two copies started 4 s apart, each saw the other as
# pending/younger and each cancelled the other). mkdir on the shared filesystem is atomic: exactly one copy creates it.
# Preference: a regular-partition copy takes the lock over from a preemptible (kempner_requeue) owner.
NAME=$1
LOCK=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/.cache/race_locks/$NAME
mkdir -p "$(dirname "$LOCK")"
me=$SLURM_JOB_ID; mypart=$SLURM_JOB_PARTITION
alive() { squeue -h -j "$1" -o %i 2>/dev/null | grep -q .; }
if mkdir "$LOCK" 2>/dev/null; then
  echo "$me $mypart" > "$LOCK/owner"
else
  read -r owner opart < "$LOCK/owner" 2>/dev/null
  if [ "$owner" = "$me" ]; then :                                  # requeued owner coming back
  elif [ -z "$owner" ] || ! alive "$owner"; then echo "$me $mypart" > "$LOCK/owner"     # stale lock: take over
  elif [ "$opart" = kempner_requeue ] && [ "$mypart" != kempner_requeue ]; then
    echo "$me $mypart" > "$LOCK/owner"; scancel "$owner"           # regular copy replaces a preemptible owner
  else
    echo "race_guard: $owner ($opart) owns $NAME; copy $me exits"; exit 1
  fi
fi
# owner: cancel every other copy. A preemptible owner keeps the regular copies queued as backup.
for x in $(squeue -h -u "$USER" -n "$NAME" -o %i:%P); do
  j=${x%%:*}; p=${x#*:}
  [ "$j" = "$me" ] && continue
  [ "$mypart" = kempner_requeue ] && [ "$p" != kempner_requeue ] && continue
  scancel "$j"
done
exit 0
