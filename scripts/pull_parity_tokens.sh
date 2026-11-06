#!/usr/bin/env bash
# Copy the phone's own generated writer token ids (Documents/parity_tokens.json, written by #writerparitytokens)
# into eval/device_anchor82/ for the cluster's divergence localiser (M103(2)).
# These are ids for the CLUSTER'S PUBLIC parity thumbnails, not for Reza's photos, so unlike every other device
# artifact this one is committable.
set -euo pipefail
cd "$(dirname "$0")/.."
BUNDLE="com.rezashamji.findpics"
UDID="$(xcrun devicectl list devices 2>/dev/null | grep -i iphone | grep -iv simulated \
        | grep -oE '[0-9A-F]{8}-[0-9A-F]{16}' | head -1)"
[ -n "$UDID" ] || { echo "no physical iPhone in devicectl list devices"; exit 1; }
OUT="eval/device_anchor82/phone_parity_tokens.json"
xcrun devicectl device copy from --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BUNDLE" --user mobile \
  --source Documents/parity_tokens.json --destination "$OUT" >/dev/null
python3 -c "
import json,sys
d=json.load(open('$OUT'))
print('$OUT', len(d), 'photos, token counts', sorted({len(v) for v in d.values()}))
"
