#!/usr/bin/env bash
# Run one developer task on the connected iPhone WITHOUT Reza typing and WITHOUT a sticky launch argument.
#
# Why a file and not `devicectl ... -- -factsOnly`: devicectl persists launch arguments per bundle id, and iOS
# replays them on relaunches nobody asked for. That already killed two timing runs here (JOURNAL 10-11 21:16,
# 21:48) and a replayed "-factsOnly 300" would spend 300 judge calls unasked. Documents/next_query.txt is consumed
# (deleted) by the app on read, so it can only ever run once.
#
# Usage: bash scripts/run_device_task.sh '#factsparity'
#        bash scripts/run_device_task.sh '#factsonly 300'
#        bash scripts/run_device_task.sh 'photos of a dog'     # an ordinary search, timed
set -euo pipefail
cd "$(dirname "$0")/.."
TASK="${1:?usage: run_device_task.sh '<#command or query>'}"
BUNDLE="com.rezashamji.findpics"
UDID="$(xcrun devicectl list devices 2>/dev/null | grep -i iphone | grep -iv simulated \
        | grep -oE '[0-9A-F]{8}-[0-9A-F]{16}' | head -1)"
[ -n "$UDID" ] || { echo "no physical iPhone in devicectl list devices"; exit 1; }
TMP="$(mktemp -t next_query)"
printf '%s' "$TASK" > "$TMP"
xcrun devicectl device copy to --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BUNDLE" --user mobile \
  --source "$TMP" --destination Documents/next_query.txt >/dev/null
rm -f "$TMP"
echo "queued: $TASK"
# FAIL LOUDLY IF THE LAUNCH DID NOT HAPPEN. This used to pipe into grep, so the script's exit status was grep's and
# a refused launch looked like success - and the caller then waited out its whole timeout for a run that never
# started. iOS refuses to launch on a LOCKED device ("could not be unlocked"), which is the common case here and
# cost 20 minutes of wall clock on 11-06 before anyone looked at the message.
OUT="$(xcrun devicectl device process launch --device "$UDID" --terminate-existing "$BUNDLE" 2>&1)"
echo "$OUT" | grep -E "Launched|FailureReason" | head -2
if ! echo "$OUT" | grep -q "Launched application"; then
  echo "LAUNCH REFUSED. If it says 'could not be unlocked', unlock the phone and run this again."
  exit 1
fi
