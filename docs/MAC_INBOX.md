# Mac inbox: tasks from the cluster session for the Mac Claude session

The cluster Claude and the Mac Claude cannot message each other. This file plus JOURNAL.md is the channel:
the cluster session writes tasks here; the Mac session does them and reports in JOURNAL.md.

## Protocol for the Mac session (run as a loop, every ~15 min)
1. `git pull --rebase --autostash origin main`.
2. Do the OPEN items below, top to bottom. Builds and installs run in the background; never sit idle waiting on one
   thing if another item can proceed.
3. For each item finished: append `MAC: <time> [item id] <result with numbers>` to JOURNAL.md and move the item to DONE
   with a one-line result. Screenshots of the phone go in data/private/ only (never committed).
4. If an item needs Reza (passwords, 2-factor codes, Apple account pages, tapping on the phone):
   write `MAC NEEDS REZA: <exact steps>` in JOURNAL.md, skip the item, continue with the rest.
5. Commit with the repo-local identity (docs/MAC_SESSION.md) and `git push origin main` after every finished item.
6. Rules: never delete or modify photos; the app only reads the library. Push only main.

## OPEN
- [M3] After install, with the app open: record index progress, the face re-embed banner ("k of N") and how fast it
  moves, and the Self-check face cosine (should be > 0.9).
- [M4] Local copy sizes of iCloud-only photos: add a DEBUG-only log (or Self-check row) that asks PhotoKit for 200
  iCloud-only photos at 896 px with network access OFF and records the returned long side. Journal the histogram
  (how many >= 806 px, 448-805, < 448). This decides whether the judge must download every photo it checks.
- [M5] Search "photos of a dog" (Apple-model fallback while memory is 2.5 GB). Record: seconds per judged photo, how
  many photos Apple's model refused (plan note), found count after 10 minutes, and whether the search keeps going.
- [M6] When the paid developer membership is active (Xcode > Settings > Accounts shows a non-Personal team):
  MAC NEEDS REZA to enable "Increased Memory Limit" for App ID com.rezashamji.findpics at developer.apple.com >
  Identifiers; then switch teamIdentifier in Package.swift, rebuild WITH the entitlement, and record the app memory
  number (Self-check). Then repeat M5 with the default Qwen3-VL judge.

- [M7] Rebuild and reinstall once more: commit ab6758d (face expansion cut 0.60 -> 0.62) may have landed after the
  M2 build. Check `git log` for the build's commit; if it predates ab6758d, rebuild + install, then continue M3.
- [M8] Simulator (Reza has it open): use it for UI checks that need no real library or memory limit: the "Is this
  you?" sheet, the "Show me Max: pick 1-3 photos" sheet, the About/licences screen, the Apple-refusal plan note.
  Screenshot each into data/private/sim/ and journal what looks wrong. The phone remains the real test.

## DONE
- [M1] 03:19 DONE. face_auraface.mlpackage (128 MB) was already in Sources/Models (Reza rsynced it 03:15), so no
  MAC NEEDS REZA was needed. Deleted face_buffalo_l.mlpackage; the built .app now bundles only face_auraface +
  pe_core_image + pe_core_text. Checked first that nothing loads the old model at runtime: only FaceProfile.shipped
  is ever instantiated, and FaceMigration re-embeds the SAME face boxes with the NEW model (re-read from the photos),
  so the buffalo_l weights are not needed to migrate.
- [M2] 03:19 DONE. Current main builds CLEAN for device: 0 errors, 0 Sendable/isolation warnings, so none of the
  Swift 6 risk points in 353c75e / d434063 bit. Installed on the iPhone 18 Pro (plain build, no entitlement re-sign).
