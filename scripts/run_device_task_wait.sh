#!/usr/bin/env bash
# run_device_task.sh, but WAIT FOR THE APP TO FINISH instead of sleeping a guessed number of seconds.
#
# Why (mac 11-06 16:55, after losing three runs to this): the app needs ~85-170 s to load the index before it runs
# anything, and run_device_task.sh launches with --terminate-existing. A fixed sleep that is too short does two bad
# things at once - it kills the app mid-search, and because memcheck.txt is buffered, the lines it had already
# written are lost with it. The result looks exactly like a crash in whatever you just built. It is not; it is the
# stopwatch. So: poll the app's own "ONEQUERY done" marker, with an explicit timeout.
# Usage: bash scripts/run_device_task_wait.sh '<#command or query>' [timeout_seconds]
set -uo pipefail
cd "$(dirname "$0")/.."
TASK="${1:?usage: run_device_task_wait.sh '<#command or query>' [timeout]}"
LIMIT="${2:-360}"
LOG="data/private/device_logs/memcheck_wait.txt"
marks () { bash scripts/pull_device_log.sh wait >/dev/null 2>&1 && grep -c "ONEQUERY done\|WPTOKENS\|WPEXACT\|FACTSPARITY\|FACTSONLY n=" "$LOG" || echo -1; }
BEFORE=$(marks)
[ "$BEFORE" -ge 0 ] || { echo "cannot read the device log"; exit 1; }
bash scripts/run_device_task.sh "$TASK" || exit 1
START=$(date +%s)
while :; do
  sleep 20
  NOW=$(marks)
  if [ "$NOW" -gt "$BEFORE" ]; then
    echo "finished after $(( $(date +%s) - START )) s"
    exit 0
  fi
  if [ $(( $(date +%s) - START )) -ge "$LIMIT" ]; then
    echo "TIMED OUT after ${LIMIT} s - the app never logged a completion marker. Last lines:"
    tail -6 "$LOG" | cut -c1-200
    exit 2
  fi
done
