#!/usr/bin/env bash
# Build the iPhone app (RELEASE) WITH the increased-memory-limit entitlement and install it on the connected iPhone.
#
# Why (MAC 10-09 21:40, after three days of 0xe8008015 refusals): installd checks the signature's entitlements against
# the EMBEDDED provisioning profile, and Xcode only puts an entitlement in the profile when the target asks for it. A
# .swiftpm (AppleProductTypes) cannot declare one in Package.swift, and re-signing the .app afterwards (this script's old
# approach) is refused. What works: enable "Increased Memory Limit" on the App ID (developer.apple.com, paid team) AND
# pass an entitlements file to xcodebuild, so automatic signing requests a profile that carries it.
# Always RELEASE: Debug is ~85x slower on the Swift pixel loops (MAC 10-07 14:46), which ruins every timing.
# Usage (Mac): bash scripts/build_device_entitled.sh [device-UDID]
set -euo pipefail
cd "$(dirname "$0")/../ios/FindPicsApp.swiftpm"
ENT=".entitlements/findpics.entitlements"
[ -f "$ENT" ] || { echo "ERROR: $ENT missing"; exit 1; }

echo "== building Release for device with $ENT =="
xcodebuild -scheme "find pics" -configuration Release -destination 'generic/platform=iOS' \
  -allowProvisioningUpdates CODE_SIGN_ENTITLEMENTS="$ENT" build | tail -3

APP=$(find "$HOME/Library/Developer/Xcode/DerivedData/FindPicsApp.swiftpm-"*/Build/Products/Release-iphoneos \
        -maxdepth 1 -name "find pics.app" 2>/dev/null | head -1)
[ -n "$APP" ] || { echo "ERROR: built .app not found"; exit 1; }
echo "== built: $APP =="

echo "== verify: signature and embedded profile both carry increased-memory-limit =="
codesign -d --entitlements :- "$APP" 2>/dev/null | grep -q "increased-memory-limit" || { echo "ERROR: not in signature"; exit 1; }
security cms -D -i "$APP/embedded.mobileprovision" 2>/dev/null | grep -q "increased-memory-limit" \
  || { echo "ERROR: not in the embedded profile (is the capability enabled on the App ID?)"; exit 1; }
echo "OK"

UDID="${1:-$(xcrun devicectl list devices 2>/dev/null | awk '/iPhone/ && /connected/ {print $3; exit}')}"
[ -n "$UDID" ] || { echo "No connected iPhone found; install later with: xcrun devicectl device install app --device <UDID> \"$APP\""; exit 0; }
echo "== installing on $UDID =="
xcrun devicectl device install app --device "$UDID" "$APP"
echo "Installed. Launch normally (no -runQuery / -selfCheck / -localSizes), or indexing will not run."
