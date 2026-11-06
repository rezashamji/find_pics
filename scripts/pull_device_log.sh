#!/usr/bin/env bash
# Copy the app's own log (Documents/memcheck.txt) off the connected iPhone into data/private/device_logs/.
# The log is DERIVED FROM REZA'S LIBRARY (counts, ids, album names), so it lands in data/private/ - chmod 700,
# git-ignored - and is never committed. Usage: bash scripts/pull_device_log.sh [tag]
set -euo pipefail
cd "$(dirname "$0")/.."
BUNDLE="com.rezashamji.findpics"
UDID="$(xcrun devicectl list devices 2>/dev/null | grep -i iphone | grep -iv simulated \
        | grep -oE '[0-9A-F]{8}-[0-9A-F]{16}' | head -1)"
[ -n "$UDID" ] || { echo "no physical iPhone in devicectl list devices"; exit 1; }
OUT="data/private/device_logs"
mkdir -p "$OUT"
TAG="${1:-$(date +%H%M%S)}"
xcrun devicectl device copy from --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BUNDLE" --user mobile \
  --source Documents/memcheck.txt --destination "$OUT/memcheck_$TAG.txt" >/dev/null
echo "$OUT/memcheck_$TAG.txt  ($(wc -l < "$OUT/memcheck_$TAG.txt") lines)"
