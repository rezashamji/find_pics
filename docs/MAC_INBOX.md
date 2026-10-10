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
- [M28] Album completeness note in plain words (FindPicsCore.completenessNote, Search.swift): "Checked the 450 most
  likely of 16,965 photos. Up to about 326 more could be among the other 16,515 (95% sure); "Look at everything"
  checks them. In tests, the AI judge kept about 9 in 10 of the real matches it looked at." (replaces "At least 4% of
  matches found (95% confidence, relative to the AI judge)..."). Note: the bound WAS already wired (your 21:40 said it
  was not; Search.swift onRound). Ride along with the next build; screenshot one album note.
- [M27] NOW, IN THIS ORDER (cluster 10-09 20:40). (0) Relaunch find pics NORMALLY (no -runQuery / -selfCheck /
  -localSizes) and confirm the store mtime moves; from now on every loop tick checks store mtime, not residency.
  (1) Install M24 + M26 together (Release; M23 is over), journal the "Checking more of each video" two-point rate.
  (2) M25 step (1): measure Qwen3-VL-4B's REAL peak footprint on the phone (a developer runner that loads the judge
  ignoring the 3.6 GiB guard and judges one 896 px public photo; then relaunch normally). This decides whether a
  vision-quantized judge fits the 3.2 GiB with NO entitlement at all.
  (3) Entitlement option (a): Reza HAS ticked "Increased Memory Limit" on the App ID (confirmed 10-09 20:45; steps in
  JOURNAL): delete the cached profiles (~/Library/Developer/Xcode/UserData/Provisioning Profiles/ entries for
  com.rezashamji.findpics), rebuild with -allowProvisioningUpdates, dump the embedded profile
  (`security cms -D -i "find pics.app/embedded.mobileprovision"`) and check whether its Entitlements dict now lists
  com.apple.developer.kernel.increased-memory-limit. If yes: scripts/build_device_entitled.sh should now install.
  If no: option (b), a normal Xcode project target, is the remaining route; estimate it before starting.
- [M26] PAIRING SPLIT FIX: REBUILD THE APP (cluster 10-09; eval/RESULTS.md 39). AFTER M23's READOUT (do not install
  anything while the overnight charger test runs; can ride along with the next build, e.g. M24/M25). What changed:
  FindPicsCore/Pairing.swift twoGroups (used by splitPair -> the app's "X vs Y of the same person" albums) now tries two
  EM starts and rejects a fit that collapsed onto one event's tied photos; it now takes the event labels splitPair
  already gets. On Reza's demo caches the 5-bit-vision run went from 137 H + 1 F / 119 F + 18 H (117 unclear) to
  265 + 5 / 114 + 0 (7 unclear); 30 of 33 saved runs are unchanged. No App.swift change needed (same splitPair
  signature). Steps: (1) git pull; (2) `swift test` in ios/FindPicsCore (expect 89/89, PairingTests 4/4);
  (3) Release build + install with the other pending items; (4) on the phone, one "me looking heavier vs me looking
  fit" search: report the two album counts and the "not clearly either" count from the report text.
- [M25] SMALLER PHONE JUDGE: QUANTIZE THE VISION TOWER (cluster 10-09; eval/RESULTS.md 38). AFTER M23's READOUT (do not
  install anything while the overnight charger test runs). Why: Qwen3-VL-4B-Instruct-4bit ships its vision tower in bf16
  (0.83 GB of its 3.09 GB = 2.88 GiB). Measured on the real phone weights: vision 8-bit saves 0.36 GiB and is clean
  (eye labels 231/45 vs 232/47 at 0.7, of 261 right / 141 wrong; Reza's demo 114 vs 115 fit); 6-bit saves 0.455 GiB at
  -6 fit photos of 124; 5-bit BROKE the demo (rejected). UPDATE 10-09 (RESULTS 39): that break was the pairing
  split, not the judge; after the fix (M26) the 5-bit demo is 265 H + 5 F / 114 F + 0 H, so 5-bit (0.50 GiB) is an
  option again; the bits choice still follows the measured peak (step 1). Steps:
  (1) FIRST measure the real Qwen3-VL-4B footprint on the phone with the CURRENT checkpoint: phys_footprint (or
      os_proc_available_memory before/after) at load and the PEAK during one judge call on a 896 px photo. The 3.6 GiB
      guard in App.swift is an unmeasured carry-over from Qwen3.5-4B (weights alone are 2.88 GiB + KV ~0.1 GiB +
      transients). Report the true gap to the 3.2 GiB available.
  (2) Pick bits: gap <= 0.36 GiB (minus ~0.1 margin) -> --bits 8; otherwise --bits 6. `pip install mlx` (Mac),
      `huggingface-cli download mlx-community/Qwen3-VL-4B-Instruct-4bit --local-dir .cache/q3vl4b`,
      `python scripts/quantize_vision_mlx.py .cache/q3vl4b .cache/q3vl4b_visN --bits N --group 64`
      (expect "quantized 104 vision layers"; 8-bit saves 0.385 GB, 6-bit 0.488 GB). mlx_vlm.convert CANNOT do this (it
      always skips vision). Smoke test: `python -m mlx_vlm.generate --model .cache/q3vl4b_visN --image <public photo>
      --prompt "Describe."`.
  (3) Host it where the app downloads from (HF repo under Reza's account, or Background Assets; Reza decides the
      account). mlx-swift-lm 3.32.x loads it as is (Load.swift quantizes every Linear with "<path>.scales", per-layer
      bits from config.json "quantization").
  (4) App.swift: replace the fixed 3.6 GiB guard with a per-model value = measured peak + ~0.15 GiB margin.
  (5) On the phone: judge loads under the 3.2 GiB budget? Peak memory, s per judge call vs the bf16-vision checkpoint,
      and P(yes) on 5 public photos within ~0.05 of the old one.
- [M24] TWO-SWEEP FRAMES PASS (cluster 10-09; FindPicsCore/LazyVideo.swift, Index.swift sampleVideoFrames /
  frameUnits / framesPlan, VideoFrames.sampleEach, SearchView banner). NOT BEFORE M23's morning readout: installing
  kills the overnight test and changes the banner text M23 reads.
  WHAT CHANGED: the frames pass no longer samples every 4 s video by video. Sweep 1 gives EVERY cover-frame-only
  video (11,434 on 10-08 23:56) its MIDDLE and END frames (2 decoded + embedded; the cover vector stays as t = 0;
  when the cover had faces, t = 0 is also decoded for faces only), newest first, 5 in flight, medium-quality movie.
  Only when no cover-only video is left to try does sweep 2 sample every 4 s, reusing sweep 1's frames that fall on
  its times. Banner: "Checking more of each video: k of N" (sweep 1), then "Looking closer inside videos: k of N"
  (sweep 2); both whole job. Album "only checked by their cover frame" note disappears once sweep 1 is done.
  Store: one new flag bit in each record (older builds read it as "fully sampled"; no migration).
  (a) BUILD Release + INSTALL. Swift 6 risk points if it does not compile clean: Index.swift `FramesPlan` /
      `SampledFrames` (Sendable; FrameUnit payloads), `g.addTask` capturing `plan` / `task` / `stage0`,
      `(try? store?.full(id))?.frames` optional flattening, frameUnits' nested `faces(_:)` func and the two closures
      passed to VideoFrames.sampleEach (`times:` + trailing `each`, both non-escaping, mutate locals).
  (b) SWEEP 1 RATE: foreground the app on Wi-Fi + charger, read the banner "Checking more of each video: k of N"
      at two points >= 5 min apart (your own two-point rate, like M20), journal videos/min and the time to finish
      the ~11,434. Prediction (scaled from the 2-s pass's ~46 videos/min at ~9.1 frames = ~420 frames/min, assuming
      cost scales with frames, which is optimistic because each video also has a fixed AVAsset / download cost):
      ~2-3 frames per video -> ~140-210 videos/min -> ~55-80 min for 11,434. Slower than ~100/min means the fixed
      per-video cost dominates; say so with the number.
  (c) SWEEP 2 RATE: once the line switches to "Looking closer inside videos", same two-point read. Prediction ~4
      frames decoded per video -> ~100/min, ~2 h for the same 11,434 (rough: their length mix is unknown).
- [M23] RUNNING RIGHT NOW - DO NOT TOUCH THE PHONE. Step (1) is DONE: the cover pass finished at ~23:55 on 10-08
  and the whole library is searchable (75 of 187,159 left, all unreadable or undownloadable). Step (2), the clean
  M19(b) charger test, STARTED at 23:57 on 10-08 and runs until Reza wakes.
  SETUP AS IT STANDS (verified by screenshot + process list at 23:57): find pics is BACKGROUNDED on the Home
  screen, NOT swiped away (process resident), phone on the charger, on Wi-Fi, untouched.
  BASELINE COUNTERS, read at 23:56, one minute before backgrounding:
     Improving faces:        10,567 of 80,185
     Looking inside videos:   8,907 of 20,341
     Not searchable yet:     17 could not be downloaded + 58 could not be read  (= 75)
  RULES FOR EVERY LOOP TICK UNTIL REZA IS AWAKE (this is the whole experiment; breaking one of these wastes the
  night, which has already happened twice - 10-08 10:24 and 17:06):
    * DO NOT run `devicectl device process launch` on com.rezashamji.findpics. Launching FOREGROUNDS it.
    * DO NOT expect a screenshot to show the counters: they only render in the foreground, so a screenshot will
      show the Home screen. A screenshot is a harmless check that the phone is awake and charging; it is not the
      measurement.
    * The ONLY safe overnight signal is `devicectl device info processes` (it foregrounds nothing): it says
      whether find pics is still alive, and if it dies, whether iOS brings it back. Logged every 15 min to
      .cache/m19_residency.log.
  IN THE MORNING, in this order: (a) read .cache/m19_residency.log for whether the app stayed alive all night;
  (b) ONLY THEN foreground find pics and read the two counters; (c) subtract the baseline above. Any increase
  means iOS ran the BGProcessingTask on the charger. No increase, after a full undisturbed night on power, is
  the answer Reza has been asking for since 07:25, and the banner wording has to change.
- [M19] VERIFY THE CHARGER PATH (your 07:25 point (2): we promise it without evidence). With the new banner wording
  ("continues while the phone charges with the app closed (iOS decides when)"), test it: (a) quick: in Xcode/lldb
  attached to the Release app, trigger the task by hand (docs/MAC_SESSION.md has the
  _simulateLaunchForTaskWithIdentifier command for com.rezashamji.findpics.index) and journal whether the counter
  advances and the task completes or expires cleanly; check `plutil -p` shows BGTaskSchedulerPermittedIdentifiers +
  UIBackgroundModes processing in the built Info.plist. (b) real (CORRECTED per your 08:08: force-quit suppresses
  BGTasks): when the queue still has work, leave find pics BACKGROUNDED (Home screen, NOT swiped away), phone on the
  charger + Wi-Fi + untouched for >= 1 h, then reopen and journal the counter before/after. If (b)
  shows no progress, say so plainly: the banner wording must then change again.
- [M13] FACE UPGRADE PASS (cluster 10-07; FindPicsCore/FaceUpgrade.swift + Index.swift upgradeFaces). Faces found on the
  448 px index read are not reliable identities (PHONE_PARITY: 40-65% of faces drop below the 40 px gate, small-face
  same-face cosine p5 0.19-0.33). Each entry now records `faceSide`; entries with faces from a read < 1280 are re-read
  at FindPicsCore.faceReadSide = 1280 (resizeMode .exact, downloads allowed, 5 requests in flight, newest first,
  saved every 200), only on unconstrained Wi-Fi: foreground in chunks of 600 photos (one indexing job each, app in
  front only) and to the end in the charger BGProcessingTask. Until a photo is upgraded, person albums and "with X"
  filters leave it OUT and the album note says "N photos not checked for faces yet (improving overnight)".
  (a) BUILD AND INSTALL (Release, per your M12 finding). Swift 6 risk points to read if it does not compile clean:
      PhotoIndex.upgradeFaces (withTaskGroup inside the actor, addTask calling the static readFaces), the
      UpgradedFaces Sendable struct (needs DetectedFace implicitly Sendable), AppModel.enqueueFaceUpgradeChunk
      (MainActor job closure, UIApplication.shared.applicationState), Moments.withPeople's new 3-tuple,
      PeopleStore.refreshAfterUpgrade (actor, awaits PhotoIndex.photoFaces; FaceSource gained `side`).
      Saved people are re-derived from the upgraded faces after each chunk that re-read one of their reference photos;
      journal if anyone gets "Is this you?" again with the note "find pics re-read your photos at full size ...".
  (b) JOURNAL THE UPGRADE: the status line reads "Improving faces: k of N photos with faces read at full size" (k =
      photos whose faces are checked, N = photos with faces). Log k and N with times while the app is in front on
      Wi-Fi (>= 10 min), then after a charger night; photos/s = delta k / delta t; battery %/hour if visible (Settings >
      Battery). Also run one people search ("photos of me") and journal its note ("N photos not checked ...") and the
      album count, before and after the night.
  (c) PHOTOKIT'S REAL RENDITION QUALITY (decides whether 448 image vectors are ~0.95 or ~0.99 of full size). For 50
      photos whose ORIGINALS ARE LOCAL (a network-off full-size read succeeds), compute the image vector twice through
      the fixed Embedder (Pillow resize + fp16): from the index read (PhotoLibrary.read side 448, purpose
      .indexForeground = resizeMode .fast; log its long side, expect 448-486) and from a full-size read (.exact at
      the original's size, network off). Journal mean / p5 / min cosine and the long sides seen. Simulation
      (PHONE_PARITY) predicts ~0.95 if PhotoKit's rendition is JPEG q80 4:2:0 and ~0.99 if q95 4:4:4. A DEBUG-only
      runner like runLocal448 is fine (inside #if DEBUG, Release must still build).
- [M6] When the paid developer membership is active (Xcode > Settings > Accounts shows a non-Personal team):
  MAC NEEDS REZA to enable "Increased Memory Limit" for App ID com.rezashamji.findpics at developer.apple.com >
  Identifiers; then switch teamIdentifier in Package.swift, rebuild WITH the entitlement, and record the app memory
  number (Self-check). Then repeat M5 with the default Qwen3-VL judge.

## DONE
- [M14] CLOSED. The index half was answered 10-07 14:46 (Release per-stage medians sum to 31 ms/photo; the ~4 s was
  a Debug -Onone artefact). The open half - per-stage timing of the ~120 s/photo download behaviour - is now moot:
  M17 replaced that path and the download pass runs ~100x faster, so instrumenting it would measure a regime that
  no longer exists. If the NEW download path's internals are wanted, that is a fresh question worth its own item.
- [M15] DONE (verified in use since 07:11). Release builds clean, FindPicsCore swift test 64/64 then 67/67 with the
  new suites, the one-time index.json conversion completed in under ~90 s including app start (never caught on a
  screenshot), and "add + save store: 0 ms" across 74,887 saves shows the append-only journal working. The
  unplanned win: app memory available went 1.6 GB -> 3.2 GB because vectors moved from parsed JSON in anonymous
  memory to memory-mapped Float16 rows, which cut the gap to the 3.6 GB judge from ~2 GB to ~0.4 GB and changes
  the M6 calculus.
- [M21] 09:18 DONE, verified across a relaunch: headline "152,313 of 187,141 photos and videos searchable" rose
  from 151,850 rather than dropping, and the pass line reads "34,828 left" instead of a per-launch count. The
  08:47 defect is fixed at the root.
- [M18] 07:22 DONE: the bar plus "about 1 h left (~614 per minute)" appears on the download pass and a glance now
  tells slow from stuck. Caveat journalled: on the INDEX pass that ran just before, only the bar and count showed,
  with no rate/time-left line, across two screenshots well past the ">= 5 items and >= 60 s" threshold.
- [M17] 07:22 DONE, and the win is large. (a) -queueSizes confirmed the prediction: 39,005 = 2,508 photos + 36,497
  videos (93.6% video), 275.9 h of video, 332,171 frames. (b) Built clean, no Swift 6 risk points bit. Measured:
  download pass 0.095 -> 10.2 items/s (~100x), ETA ~4.8 days -> ~1 h at the time. The videos are mostly not
  downloading at all - .mediumQualityFormat returns a local rendition, the same shape of fix as 448 px .fast in
  M16. Deviation flagged: installed RELEASE for the real run, not DEBUG, because 332,171 frames in a Debug build
  would be ~85x slower on the Swift pixel paths.
- [M16] 17:33 DONE, fix confirmed. The iCloud-only set now indexes from local renditions at 16.5 photos/s
  (4,168 -> 9,585 of 169,546 in 328 s), against 0.008/s before the fix. Whole library ~2.9 h, not ~236 days.
- [M14] 14:46 DONE. Release per-stage medians sum to 31 ms/photo (Core ML 8 ms is the largest; EXIF 3 ms; save 0 ms).
  The ~4 s was a Debug -Onone artefact across the Swift pixel loops, not any one stage.
- [M12] 14:46 DONE. Release index rate 21.2 photos/s (1,588 -> 6,400 of 16,384 in 227 s), 85x the Debug 0.25/s.
  Whole 187k library ~2.5 h. The earlier 8.7-day extrapolation is withdrawn (Debug build).
- [M11] 12:48 DONE. fp16 image tower rsynced by Reza and verified in the built .app; image self-check 0.9345 ->
  0.9999 (scene) and 0.9765 -> 0.9998 (stripes), which per M11's diagnostic table means BOTH the fp16 tower and the
  Pillow-exact resize are live. Text and face unchanged. Cost: app memory 2.03 GB -> 1.13 GB. swift test 42/42 after
  qualifying LibraryItem in three test files (it did not compile on macOS before).
- [M10] 12:26 DONE. iCloud sends DERIVATIVES for small asks, not originals: at 448 the original came down 0/50, at
  896 1/50, at 1280 9/50 (indexing currently asks at 1280). Cost: 448 ~free, 896 0.6 s/photo sequential and 0.2 s
  five-at-a-time (~3x, so latency-bound). Budget for 169,707: ~9.4 h at 896 x5, far less at 448. Caveat: the 448 row
  is suspiciously instant and may mean a ~480 px rendition is already local, which M9's network-off probe missed -
  re-run that probe at 448 with resizeMode .fast, network off, to settle it.
- [M8] 04:46 PARTLY DONE. Simulator build + install + -demoUI screenshot: results UI renders correctly in light mode,
  no layout breakage (data/private/sim/demoui.png). The first "me heavier" tile looks like the app icon - worth a
  tap to see whether it is a failed-thumbnail placeholder. The other three checks need taps and the simulator-control
  tool is not granted access yet -> MAC NEEDS REZA ("Let Claude use it" in the simulator panel).
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
- [M22-trust] CLOSED 10-08 17:32 (Reza tapped through it; the Release build launched). Renumbered: the cluster
  independently opened a different [M22] the same evening. The Release build of M20 (cover-first
  videos) is installed but iOS will not launch it: the phone shows "Unable to Verify App - An Internet connection
  is required to verify trust of the developer 'Apple Development: rezamshamji@gmail.com (ZN8M63RSR5)'". Free
  Personal Team re-verification; a Mac cannot tap it away. Reza: tap Cancel, make sure the phone is on Wi-Fi, then
  tap the find pics icon. If it refuses again: Settings > General > VPN & Device Management > Apple Development:
  rezamshamji@gmail.com > Trust, then open find pics. Then the Mac session measures the cover pass.
