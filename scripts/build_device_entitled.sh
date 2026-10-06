#!/usr/bin/env bash
# Build the iPhone app for a physical device and sign it with the increased-memory-limit entitlement.
#
# Why this script exists:
#   The 4-bit 4B judge needs ~3.1 GB of weights plus working memory. iOS caps a third-party app's memory
#   well below total RAM by default (an iPhone 18 Pro, 12 GB, still only hands the app ~3.3 GB -> ~2.4 GB
#   free after the embedder + face engine load), so the app's own guard refuses to load the model and shows
#   "This iPhone lets an app use X GB ... needs about 3.6 GB."
#   The fix is the entitlement com.apple.developer.kernel.increased-memory-limit, which raises that cap
#   (8 GB+ phones -> ~6 GB). It is an UNRESTRICTED entitlement: it works with a free Personal Team and needs
#   no App ID capability. But Xcode's AUTOMATIC signing tries to register it in the provisioning profile and
#   fails ("not found and could not be included in profile"), and AppleProductTypes has no capability for it,
#   so it cannot go in Package.swift. The kernel honours the entitlement from the code signature directly, so
#   we build normally and then re-sign the .app, merging the entitlement into Xcode's generated entitlements.
#
# This needs NO Xcode GUI. Output: a device .app whose signature includes the entitlement (verify printed).
set -euo pipefail
cd "$(dirname "$0")/../ios/FindPicsApp.swiftpm"

SCHEME="find pics"
IDENTITY="${CODESIGN_IDENTITY:-Apple Development: rezamshamji@gmail.com (ZN8M63RSR5)}"

echo "== building for device =="
xcodebuild -scheme "$SCHEME" -destination 'generic/platform=iOS' -allowProvisioningUpdates build \
  | tail -3

APP=$(find "$HOME/Library/Developer/Xcode/DerivedData/FindPicsApp.swiftpm-"*/Build/Products/Debug-iphoneos \
        -maxdepth 1 -name "find pics.app" 2>/dev/null | head -1)
[ -n "$APP" ] || { echo "ERROR: built .app not found"; exit 1; }
echo "== built: $APP =="

echo "== merging entitlement and re-signing =="
ENT=$(mktemp /tmp/fp_ent.XXXXXX)  # BSD mktemp: the X's must be at the end of the template
codesign -d --entitlements :- "$APP" 2>/dev/null > "$ENT"
/usr/libexec/PlistBuddy -c "Add :com.apple.developer.kernel.increased-memory-limit bool true" "$ENT" 2>/dev/null \
  || /usr/libexec/PlistBuddy -c "Set :com.apple.developer.kernel.increased-memory-limit true" "$ENT"
codesign --force --sign "$IDENTITY" --entitlements "$ENT" --timestamp=none --generate-entitlement-der "$APP"
rm -f "$ENT"

echo "== verify (should list increased-memory-limit) =="
codesign -d --entitlements :- "$APP" 2>/dev/null | grep -o "com.apple.developer.kernel.increased-memory-limit" \
  || { echo "ERROR: entitlement not present after re-sign"; exit 1; }
echo
echo "OK. Entitled device build at:"
echo "  $APP"
echo "To put it on a connected, unlocked, trusted iPhone (asks for your OK):"
echo "  xcrun devicectl device install app --device <UDID> \"$APP\""
