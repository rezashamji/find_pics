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
# A manifest of THIS Mac's file hashes. The phone cannot check the cluster's png_sha256 - PNG file bytes differ
# between zlib builds while the pixels (verified 20/20 by the recipe) are identical - so what it checks is that
# the file it reads is the file that was pushed.
python3 - "$SRC" <<'PYEOF'
import hashlib, json, pathlib, sys
d = pathlib.Path(sys.argv[1])
m = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(d.glob("*.png"))}
(d / "manifest.json").write_text(json.dumps(m))
print(f"manifest: {len(m)} files")
PYEOF
echo "== pushing $(ls "$SRC" | wc -l | tr -d ' ') files =="
xcrun devicectl device copy to --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BUNDLE" --user mobile \
  --source "$SRC" --destination Documents/parity_exact >/dev/null
echo "pushed to Documents/parity_exact"
bash scripts/run_device_task.sh '#writerparityexact'
