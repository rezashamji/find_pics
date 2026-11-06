#!/usr/bin/env bash
# Push the parity-exact PNGs (M101) to the phone's Documents/parity_exact/, then run #writerparityexact.
#
# The PNGs are rebuilt on the Mac by eval/device_anchor82/parity_exact_recipe.py and verified pixel-for-pixel
# against the cluster's hashes. They are NOT bundled in the app: 4.9 MB of eval images have no business in a
# shipping bundle, and Documents can be replaced without a rebuild.
# They are public Wikimedia Commons photos, but they live under data/private/ because data/ is git-ignored and
# nothing derived from an eval needs to be in the repo twice.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="data/private/parity_exact"
[ -d "$SRC" ] || { echo "ERROR: $SRC missing - run: data/venv_parity/bin/python eval/device_anchor82/parity_exact_recipe.py --out=$SRC"; exit 1; }
BUNDLE="com.rezashamji.findpics"
UDID="$(xcrun devicectl list devices 2>/dev/null | grep -i iphone | grep -iv simulated \
        | grep -oE '[0-9A-F]{8}-[0-9A-F]{16}' | head -1)"
[ -n "$UDID" ] || { echo "no physical iPhone"; exit 1; }
echo "== pushing $(ls "$SRC" | wc -l | tr -d ' ') files =="
xcrun devicectl device copy to --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BUNDLE" --user mobile \
  --source "$SRC" --destination Documents/parity_exact >/dev/null
echo "pushed to Documents/parity_exact"
bash scripts/run_device_task.sh '#writerparityexact'
