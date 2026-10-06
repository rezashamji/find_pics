# Instructions for the Claude Code session on Reza's Mac (overnight build loop)

You are the Mac half of a two-session setup. The other Claude session runs on the Harvard cluster (Linux, GPUs) and
does models, evaluations and search logic. You do what only a Mac can: build the iPhone app, run it in the Simulator,
look at the screens, and fix app-side compile/runtime problems. You share work through GitHub (main) and JOURNAL.md.

## Hard rules (same as CLAUDE.md)
- Reza's photos are read-only; never call delete APIs; never touch other repos; push only `main`.
- Nothing private (photos, thumbnails, screenshots of Reza's library) goes into git. Simulator screenshots of the app
  with NO personal photos are fine to keep under `ios/screens/` (gitignored is safer: keep them local).
- Commit small, pull before every commit (`git pull --rebase --autostash`), push after. Journal every fix in JOURNAL.md
  (one line, prefix "MAC:").
- Do not change search logic in ios/FindPicsCore without also telling the cluster session via JOURNAL.md: that code is
  tested against the Python engine on the cluster (golden fixtures); a change there must keep `swift test` passing.

## Loop
1. `cd ~/find_pics && git pull --rebase --autostash`
2. Device build (compile check; signing team is set in Xcode):
   `xcodebuild -scheme FindPics -destination 'generic/platform=iOS' -allowProvisioningUpdates build 2>&1 | tail -80`
   (run from `ios/FindPicsApp.swiftpm`; if the scheme name differs: `xcodebuild -list`).
3. Fix every error (smallest change that keeps behaviour; keep comments). Repeat until it builds.
4. Simulator run to SEE the app:
   `xcrun simctl boot "iPhone 17 Pro"` (or any listed by `xcrun simctl list devices`), build with
   `-destination 'platform=iOS Simulator,name=iPhone 17 Pro'`, install with `xcrun simctl install booted <path to .app>`,
   launch `xcrun simctl launch booted com.rezashamji.findpics`, then `xcrun simctl io booted screenshot ios/screens/<name>.png`
   and look at the screenshot. Note: MLX needs a real GPU; in the Simulator the model download / judge may fail.
   That is expected; check the UI flow (permission prompt, download consent screen, search field, menus, sheets).
   Add a few sample photos to the Simulator (`xcrun simctl addmedia booted <public test images>`) — never Reza's photos.
5. If the iPhone is connected and unlocked: `xcodebuild ... -destination 'platform=iOS,name=<iPhone name>' build` then
   install/run via Xcode is fine; if it is locked, skip the device run.
6. Write what you saw (screen by screen) and what you fixed into JOURNAL.md ("MAC:" lines), commit, push, repeat.
7. Questions only for Reza (Apple ID, phone unlock/Trust, money). Leave them at the top of JOURNAL.md as
   "MAC NEEDS REZA:" and keep working on other items.

## Useful facts
- App package: ios/FindPicsApp.swiftpm (Xcode needs the .swiftpm folder for AppleProductTypes).
- Models (not in git) live in ios/FindPicsApp.swiftpm/Sources/Models (face_buffalo_l, pe_core_image, pe_core_text
  .mlpackage; optional planner_adapter/).
- The app checks os_proc_available_memory before loading the 3 GB model and shows a message if < 3.6 GB.
- First-device checklist for Reza: docs/FIRST_DEVICE_TEST.md.
