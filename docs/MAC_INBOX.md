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
- [M1] Face model swap (commit 353c75e). If `ios/FindPicsApp.swiftpm/Sources/Models/face_auraface.mlpackage` exists
  (Reza copies it with rsync), delete `Sources/Models/face_buffalo_l.mlpackage`. If it does not exist yet:
  MAC NEEDS REZA (the rsync line in docs/BUILD_ON_MAC.md step 3).
- [M2] Build and install the current main on Reza's iPhone (plain build + `xcrun devicectl device install app`; the
  entitlement re-sign is refused on the free team, so skip it). Fix Swift 6 compile errors; the risk points are listed
  in the JOURNAL "MAC: please build" notes of commits 353c75e (AuraFace) and d434063 (Apple refusals, local copies).
  Push every fix.
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

## DONE
