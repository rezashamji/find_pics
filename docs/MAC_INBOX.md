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
- [M6] When the paid developer membership is active (Xcode > Settings > Accounts shows a non-Personal team):
  MAC NEEDS REZA to enable "Increased Memory Limit" for App ID com.rezashamji.findpics at developer.apple.com >
  Identifiers; then switch teamIdentifier in Package.swift, rebuild WITH the entitlement, and record the app memory
  number (Self-check). Then repeat M5 with the default Qwen3-VL judge.

- [M8] Simulator (Reza has it open): use it for UI checks that need no real library or memory limit: the "Is this
  you?" sheet, the "Show me Max: pick 1-3 photos" sheet, the About/licences screen, the Apple-refusal plan note.
  Screenshot each into data/private/sim/ and journal what looks wrong. The phone remains the real test.

## DONE
- [M5] 04:40 DONE. Found a real bug first: d434063's refusal catch used the wrong error type (iOS 27 throws
  LanguageModelError, not GenerationError.guardrailViolation), so the search still died at photo 75; fixed, and it
  now runs on. Numbers: 450 of 16,965 checked in ~9 min, 21 found, 4 refused, 11 undownloadable; 0.81 photos/s
  overall but slowing to 0.39/s (iCloud downloads). Precision looks poor (DOGEcoin article, "STICK DOG" book cover
  among the top results) - flagged, not claimed, since only thumbnails were seen.
- [M7] 04:12 DONE (verified, no rebuild needed). ab6758d is an ancestor of HEAD and FaceProfile now reads
  expand: 0.62 (it was 0.60 at the M2 build). The app installed at 04:11 is from a tree that includes it.
- [M9] 03:57 DONE, and (b) is a NO. (a) 17,365 of 187,120 items have an image vector; 169,923 waitingForICloud, so
  ~91% of the library is not indexed and the dog search's 16,964 scope was just the local items. (b) largest local
  rendition for 200 strided iCloud-only photos: 192 are < 224 px (median 120 = the grid thumbnail), 0 in 224-447,
  1 in 448-805, 7 >= 806. Indexing from local copies would rescue 4%, not ~170k. Only downloading can index them.
- [M4] 03:33 DONE. 896 px, network off, 200 iCloud-only photos strided across the whole library: 187/200 return
  NOTHING, 12 >= 806 px, 1 in 448-805, 0 upscaled. The judge must download ~94% of the photos it checks.
  My first run (newest-200) said the opposite (200/200 at 896) and was a sampling artifact; both are journalled.
- [M3] 03:48 DONE except the image half. Face cosine 0.9814 (bar > 0.9), texts 0.9999. Found and fixed a crash:
  Run self-check force-unwrapped bundle URLs for scene.png / stripes.png, which BUILD_ON_MAC.md never rsyncs, so the
  app died on the first tap. Those two images still need an rsync (MAC NEEDS REZA) for the image cosines.
  Earlier partial result: Index/startup timeline, the face re-embed banner and its rate (60 -> 1680 of 11230 in
  ~3 min; ~10.9/s steady, ~17-20 min total) and the index status line are in JOURNAL.md. The low-memory Apple
  fallback works on device. App memory is now 1.6 GB (was 2.6 GB before the face model loaded). NOT DONE: the
  Self-check face cosine needs a tap on the phone -> MAC NEEDS REZA line in JOURNAL.md.
- [M1] 03:19 DONE. face_auraface.mlpackage (128 MB) was already in Sources/Models (Reza rsynced it 03:15), so no
  MAC NEEDS REZA was needed. Deleted face_buffalo_l.mlpackage; the built .app now bundles only face_auraface +
  pe_core_image + pe_core_text. Checked first that nothing loads the old model at runtime: only FaceProfile.shipped
  is ever instantiated, and FaceMigration re-embeds the SAME face boxes with the NEW model (re-read from the photos),
  so the buffalo_l weights are not needed to migrate.
- [M2] 03:19 DONE. Current main builds CLEAN for device: 0 errors, 0 Sendable/isolation warnings, so none of the
  Swift 6 risk points in 353c75e / d434063 bit. Installed on the iPhone 18 Pro (plain build, no entitlement re-sign).
