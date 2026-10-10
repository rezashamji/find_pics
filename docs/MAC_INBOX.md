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

## MORNING BRIEF for Reza (written 10-10 07:35, while he slept)

WHAT WORKS NOW, all verified on the device by screenshot, not inferred:
  * Your whole library is searchable: 187,217 of 187,231. Only 59 items are not, and they are items the phone
    cannot read or download at all.
  * THE FACE UPGRADE IS FINISHED: 79,576 of 79,615 photos re-read at full size (the 39 missing are unreadable).
    This is what makes "photos of me" trustworthy, and last night it was dying every 7 minutes. It is done.
  * New photos you took overnight were picked up automatically (the banner showed "Adding new photos ... 14 left").
  * The increased-memory entitlement works, and BOTH models are on the phone: the planner (Qwen3.5-4B) and, as of
    05:40, the PHOTO JUDGE (Qwen3-VL-4B, 2.88 GB) which I side-loaded over the cable in under two minutes after
    the phone's own download failed 16 times on HuggingFace's CDN.
  * Still running: "Checking more of each video", 16,623 of 39,576. Every one of those videos is ALREADY
    searchable by its cover frame; this pass only improves them (191/303 -> 219/303 recall, RESULTS 37).

THE ONE THING I COULD NOT VERIFY, and it is the thing that matters: whether the photo judge ANSWERS well. It is
installed and it loads, but four attempts to drive a search from the Mac produced no error AND no visible
results, and I stopped rather than burn more app restarts on a test that cannot distinguish "slow" from "broken".

WHAT TO DO FIRST, ~5 minutes: F6 below. Type the ten fixed queries into find pics and into Apple Photos, and we
compare. That single test decides whether this is a product. It has a stopping rule written in, deliberately.

FIXED LAST NIGHT (all pushed, all verified on the device): F1 friendly error screen + Try again; F2 the
skip-set bug that ended both improvement passes as if complete; F3 chunks dying when the screen turned off;
F5 developer launch arguments silently stopping all indexing; plus the faces/frames ordering, and
FindPicsCore.passStep with 5 tests so these cannot regress silently.

## FIX LIST (things found and NOT yet done - written down because holding them "in context" meant they were
## deferred every night; Reza 10-10 05:00: "all the things that you wanted to fix, go fix and do it now")
- [F1] DONE 10-10 05:00. Raw NSError on the failure screen -> plain words + a "Try again" button
  (AppModel.friendlyFailure). Reza stared at 40 lines of NSURLErrorDomain twice and the app stayed dead until the
  Mac relaunched it.
- [F2] DONE 10-10 04:44. Both improvement passes ended as if COMPLETE whenever everything left had failed once this
  launch (PhotoIndex.upgradeSkipped / framesSkipped are excluded from `left`, and only retryFailed clears them, which
  the foreground chain never passed). Now they re-check the real counts and retry with retryFailed after 60 s.
- [F3] DONE 10-09 08:36. A chunk that COULD NOT RUN (screen off, or off Wi-Fi) returned the same 0 as "nothing left"
  and killed the chain for good.
- [F4] PARTLY DONE 10-10 05:12, and the first attempt was WRONG. I capped the foreground frames pass at 5,000
  videos to save 43 hours. Reza rejected it: "that decision to cap the 43-hour thing is just a bad decision because
  it caps one of our goals, which is accuracy ... why not do that on one parallel process and do the other stuff on
  another". Correct reasoning: 43 hours is a reason to run it unattended in the background while OTHER work happens
  in parallel, not a reason to discard 191/303 -> 219/303. Cap reverted. What was kept, because it drops nothing:
  both passes share ONE serial chain and were halving each other, so they now run in order of nearest finish line
  (App.enqueueImprovementPasses) - faces ~400/min with ~1 h left goes first, then frames gets the whole machine.
  STILL OPEN: an explicit "check all my videos now" control, so the user chooses rather than the app deciding.
- [F5] DONE 10-10 05:20 (built, not yet installed: faces was ~1 h from finishing and installing costs ~5 min of
  start-up). AppModel.keepIndexingAfterDeveloperRun() - every developer launch argument now starts the normal
  indexing chain as well, so -runQuery / -selfCheck can never again park the whole product silently.
- [F6] OPEN, THE BIGGEST ONE, AND NOW FULLY PREPPED so it costs Reza ~5 minutes. Head-to-head vs Apple Photos
  search on his own library. PLAN.md has listed it as an open risk for days; it has never been measured, and if we
  do not beat what is already on his phone for free then nothing else here matters.
  WHY IT NEEDS HIM AT ALL: I can drive find pics from the Mac (-runQuery) but devicectl cannot inject touches, so
  I cannot type into Apple Photos. Only the Apple half needs his fingers.
  THE 10 QUERIES (fixed in advance so neither side can be tuned to the result; 5 that Apple Photos is known to
  handle and 5 that need real understanding):
     1. photos of a dog                       6. me looking heavier vs me looking fit
     2. food photos                           7. photos where I look tired
     3. selfies                               8. the whiteboard with the diagram on it
     4. beach photos                          9. photos of my passport or ID
     5. sunset photos                        10. the night we got dumplings
  PROTOCOL, per query: Reza types it into Apple Photos, screenshots the top 12 results; I run the same text through
  find pics and screenshot its top 12. Then I label BOTH sets by eye at full size, blind to which app produced
  them where possible, and report right/wrong per app with the denominator (CLAUDE.md: every claim carries its
  denominator). No judging from thumbnails - open the ones that are not obvious.
  WHAT WOULD MAKE US STOP: if Apple Photos matches or beats find pics on queries 1-5 AND find pics does not clearly
  win 6-10, there is no product here and we should say so rather than keep building.
  BLOCKED ON: F7 (the photo judge is not on the phone yet), so find pics would currently answer with Apple's own
  built-in model - i.e. we would be comparing Apple to Apple. F7 must land first or the test is meaningless.
- [F7] OPEN. The photo judge (Qwen3-VL-4B, ~3 GB) is still NOT downloaded; only the planner (Qwen3.5-4B) is. The
  first real search has therefore never run on this phone.

## OPEN
- [M34] (cluster 10-10 ~06:15) Ride-alongs for the next build: (a) M32's retry is now TESTED on the cluster
  (FindPicsCore 94/94 incl. DownloadRetryTests); (b) start() removes stale partial downloads from the app's tmp/
  (CFNetworkDownload_*.tmp older than 1 h; FindPicsCore.isStaleDownloadTemp) - your 05:45 orphaned 2.69 GB file
  should disappear on the first launch of the new build: list the container before/after and journal it.
  For real users the in-app download is still the path (side-loading is a dev workaround): M32 step (3) decides
  whether we also need resume data or an Xet-free endpoint.
- [M33] (cluster 10-10 ~05:00) TWO NOTES ON YOUR 04:20 ENTRY. (1) The download retry is ALREADY WRITTEN and pushed
  (833dba8 = M32 above): do not write a second one; test and build M32. Resume data: check M32 step (3) first; only
  if retries restart from zero is NSURLSessionDownloadTaskResumeData worth wiring. (2) "Starting..." now takes ~5 min
  (2 min on 10-09 morning; the binary store opened in 0.31 s for 187k synthetic entries on Linux). Something grows.
  Time each start() stage once, Release, with -timeIndex or Date() pairs: PhotoIndex.load (store open + journal
  replay; journal size = log-*.log bytes), people/groups load, the planner model load (2.83 GB from flash),
  allAssets() + the library diff, anything else before stage leaves .start. Journal the table. If it is journal
  replay, the store's compaction rule (log > snapshot) may never fire under the face upgrade's replace-heavy load.
- [M32] MODEL DOWNLOAD RETRIES (cluster 10-10 ~04:20; your 02:40 shipping blocker). Judge.load now retries transient
  network errors with backoff (FindPicsCore.downloadRetryDelay: -1001/-1003/-1004/-1005/-1009/-1018/-1020, 2 s x1.5,
  capped 60 s, up to 40 attempts; urlErrorCode() digs NSURLErrorDomain out of wrapped errors). NOT TESTED on the
  cluster: the lab's 120 TB group quota is full and swift test cannot write its build products. So: (1) `swift test`
  in ios/FindPicsCore FIRST (new DownloadRetryTests; expect all pass), (2) Release build with the other pending items,
  (3) check whether the HF hub library resumes a partial file across our retries (watch the blob size across a
  forced Wi-Fi drop) - if every retry restarts the 2.8 GB from zero, say so: then we need resume data or Xet off.
  ALSO STILL OPEN, M30: the 2.83 GB download you saw was mlx-community/Qwen3.5-4B-4bit = the PLANNER. The photo
  judge (Qwen3-VL-4B) downloads on the first search only if the Model menu says qwen3vl. Check it before any search.
- [M31] FAST-MODE STOP RULE (cluster 10-09 ~23:59; eval/RESULTS.md 41). Code only (FindPicsCore Streaming.swift +
  Search.swift); NO separate build: it rides along with the next build you make anyway (M29/M30 or later). Round 1 now
  keeps going while >= 3 of the last 100 head photos are yes (was < 2 of the last 50), up to 2,000 (was 1,500), and fast
  mode runs round 2 by itself when its random check of the rest found >= 3 matches. After that build is installed:
  run one "photos with a car" search in fast mode and journal: judged count when the first answer appears, whether the
  album kept growing afterwards (round 2) and the judged count at "done", and the album note text (screenshot to
  data/private/ only). Expected on a ~17k-photo local scope: first answer after <= 2,150 judged; round 2 only if many
  cars hide in the rest.
- [M29] FIND PICS vs APPLE PHOTOS SEARCH, the real baseline (cluster 10-09 ~23:00; eval/RESULTS.md 40). Same 7,886
  public photos (the 4 DISBench libraries of RESULTS 34) and the same 6 searches, scored on the cluster with the same
  1,950 blind eye labels. The Mac's job: run Photos' search on them and send back the filenames. Reza's own library is
  NEVER touched: a separate, new library holds only these public photos. Nothing here deletes or edits anything.
  (0) First do M30 (the judge must be Qwen3-VL-4B: Model menu = "qwen3vl", NOT "qwen" = Qwen3.5-4B; UserDefaults can
      keep an old "qwen" across installs). Journal which one was active. The cluster's find pics numbers are for qwen3vl.
  (1) FETCH THE PHOTOS (no Reza, no 2FA): `git pull --rebase --autostash origin main`, then (background; 1,056 MB from
      YFCC100M's public S3 bucket, the source DISBench itself uses; the cluster checked all 7,886 urls byte-identical):
         python3 scripts/fetch_apple_photos_test.py
      It verifies every file's sha256 against eval/apple_photos/manifest.tsv and ends "DONE: 7886/7886 ...". It is
      resumable; rerun until 7886/7886. Files land in data/public/apple_photos_test/images/<item_id>.jpg (git-ignored).
  (2) BEFORE SWITCHING LIBRARIES, record (journal) for Reza's library: Photos > Settings > General shows "This is the
      System Photo Library" (button greyed), and Photos > Settings > iCloud: is iCloud Photos on? We restore exactly
      this at the end.
  (a) MAC NEEDS REZA: create a NEW, SEPARATE library. Quit Photos; hold Option while clicking Photos in the Dock ->
      "Create New..." -> name find_pics_test, save in ~/Pictures (~/Pictures/find_pics_test.photoslibrary). It must
      NEVER be made the System Photo Library (only the System library syncs to iCloud; never click "Use as System Photo
      Library"). Check: Photos > Settings > General shows the "Use as System Photo Library" BUTTON as clickable (= this
      is not the system library) and the window title / File menu shows find_pics_test. Apple's page says switching
      libraries turns off iCloud Photos and Shared Albums for the session; that is why step (2) records the state.
      Also needed once: the "osascript wants to control Photos" Automation prompt (Allow).
  (b) Import: File > Import... > select data/public/apple_photos_test/images (the folder) > Import All New Items.
      Check the library shows 7,886 items (journal the count; if fewer, journal which are missing). Do not add
      captions, keywords, albums or locations.
  (c) Let Photos analyse: keep the Mac on power, awake (`caffeinate -dims &`) and Photos OPEN on find_pics_test.
      Every 15 min: `bash scripts/apple_photos_search.sh --one dog` and `--one beach`; journal time + counts
      (eval/apple_photos/analysis_wait.log, commit it). Done when BOTH counts are unchanged for 3 checks in a row
      (>= 45 min) AND the Photos search panel shows no "indexing"/"analysing" note. Journal the total wait.
      UNKNOWN WE MUST MEASURE: whether Photos analyses a non-system library at all (Apple's docs do not say; forum
      reports disagree). If "dog" is still 0 after 3 h awake on power: STOP, journal it, MAC NEEDS REZA for the
      fallback: a new local macOS user account (no Apple Account signed in), where Photos' library is the System
      library of that user and cannot sync because no iCloud account exists; fetch + import there and redo (c).
      Never make find_pics_test the System library of Reza's account.
  (d) Search. First test the exact AppleScript: `bash scripts/apple_photos_search.sh --one dog` prints n and the first
      10 filenames (must look like 4796302459.jpg). Then, for all 3 styles:
         bash scripts/apple_photos_search.sh keyword        # dog, car, bicycle, beach, sunset, food
         bash scripts/apple_photos_search.sh phrase_of      # "photos of a dog", ..., "photos of food"
         bash scripts/apple_photos_search.sh phrase_with    # "photos with a dog", ..., "photos with food"
      AppleScript vs the search bar (Apple Intelligence search may only be in the UI): for "dog" and
      "photos with a dog", also type the text in Photos' Search field, press Return, and journal the count Photos
      shows. If it differs from the AppleScript count by more than a few: in the results view press Cmd-A, then
         osascript -e 'tell application "Photos" to get filename of every media item of (get selection)'
      (if that form errors, loop over `selection` like scripts/apple_photos_search.sh) and write the same 4-column
      TSV for all 6 queries of that style as eval/apple_photos/results_apple_ui_<style>.tsv (order = selection order).
      Journal also: macOS build, Photos version, whether Apple Intelligence is on (System Settings > Apple
      Intelligence & Siri), and the language/region.
  (e) `git add eval/apple_photos/results_apple_*.tsv eval/apple_photos/analysis_wait.log`, commit, push main. These
      are public-data results (filenames of public Flickr photos): fine to commit. The cluster then runs
      `python eval/score_apple_photos.py score` and fills RESULTS 40.
  (f) Afterwards: quit Photos, then hold Option while opening Photos and choose Reza's own library (Photos reopens the
      LAST USED library on a normal launch, which would be find_pics_test). Check Settings > General says "This is the
      System Photo Library" and Settings > iCloud matches what step (2) recorded; if iCloud Photos is now off and was
      on before: MAC NEEDS REZA (turning it back on is his call). Journal "Photos is back on <library name>" so Reza
      knows which library is open. Keep find_pics_test (7,886 public photos, ~1 GB) until RESULTS 40 is filled.
- [M30] URGENT, CHECK NOW: WHICH MODEL IS THE PHOTO JUDGE? The judge every phone number is about is Qwen3-VL-4B
  (Judge.visionJudgeCandidateID, engine "qwen3vl"). Qwen3.5-4B-4bit is the PLANNER (Judge.modelID); if the engine is
  "qwen" it is ALSO used as the photo judge (weaker: RESULTS 32). The default became "qwen3vl" on 10-07 (2444b69),
  but `engine` is stored in UserDefaults, so an older install's "qwen" survives reinstalls. Read the top-left Model
  menu: it must say qwen3vl. If it says qwen: switch it to the Qwen3-VL photo judge (it downloads ~2.5-3 GB more),
  journal which one was active, and whether both models fit under the new memory cap (planner unloads? Judge.swift).
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
