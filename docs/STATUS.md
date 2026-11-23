# find pics: where things stand (updated 2026-11-22)

This is the one current status page. PLAN.md (last updated 10-16), MORNING_REPORT.md (10-04) and the older parts of
README.md are history; where they disagree with this page, this page wins. Day-by-day detail is in JOURNAL.md, and the
cluster <-> Mac task queue is docs/MAC_INBOX.md.

## What it is
An iPhone app. You type what you want ("me looking heavier vs me looking fit", "food photos", "my dog Max at the
beach") and it finds those photos and videos on the phone, makes them into albums when you tap Save, and says how
complete the answer is ("found at least X%, 95% confidence"). Nothing is uploaded. Target: the App Store.

## How a search works
1. **When the app first reads the library** (after a one-time ~6 GB model download): every photo gets a fingerprint
   (a short list of numbers for what is in it) and its faces are found. This is what makes the first answer fast.
2. **Every night on the charger** (new 11-22): a small model writes a one-line description of each photo, newest
   first, ~2 s per photo, so a 148k library takes several nights. Searches score described photos from the description
   too, which is sharper than the fingerprint alone. Progress shows on the search screen.
3. **At search time**: photos are ranked by fingerprint (+ description), a photo-checking model (Qwen3-VL-4B) checks
   the top of the ranking and a random sample of the rest, and the completeness bound is computed from that sample
   (Clopper-Pearson; exact math, relative to the checker's yes/no answers).
4. **Person searches** use faces for WHO and the checker for HOW THEY LOOK, on a crop with a red box around them.
   Facts ("me at the beach") now run the same rounds and bound as every other search; looks ("me looking tired")
   walk events and stop at a first answer after 480 checks; a look PAIR ("heavier vs fit") asks both questions of each
   photo in one pass. Crops are read 4 ahead while the checker works (new 11-22).

## Measured (denominators in JOURNAL.md)
- Quick answer (no checker): mean recall per request .66-.73; objects .80, time/place .83, person+look .67,
  multi-step .54. It misses ~27% of matches and ~25% of what it shows is wrong; the checker fixes both.
- Checker agrees with careful review on 295 of 298 top-ranked photos.
- Descriptions alone (facts-only): AUC .859 on 300 pre-registered photos (bar .70).
- Phone: repeat search 10-11 s (16 s on a new phrasing; first guess on screen at ~6 s). Checker 3.17 s per photo
  when warm. Person photo ~11 s before 11-22 (read+crop ~7.95 s, check 3.17 s); read-ahead should hide most of
  the read: NOT yet measured.
- Tests: FindPicsCore 201 XCTest + 29 swift-testing pass on the Mac (11-22). Python 175/179 on the Mac (the 4 need
  av / pyarrow / reverse_geocoder installed locally).

## Done
- App builds and runs on Reza's iPhone 18 Pro with the increased-memory entitlement (paid team; profile valid to 2027-10).
- Search, people, pets/things from example photos, places, follow-ups, albums (add-only, dry-run safe), completeness note.
- Release cleanup (11-22): developer tools compiled out of store builds (`FINDPICS_DEV`), privacy manifest,
  encryption key, honest download size (~6 GB now + 1.8 GB later on Wi-Fi), safe 4.6 GB memory guard.
- Night pass wired into the charger task (11-22).
- Person-search read-ahead, bounded fact searches, one-walk look pairs (11-22).

## Open, in order
1. **Overnight phone test** - does iOS run the charger task; photos described per night; heat; memory. Read the
   NIGHTPASS lines in Documents/memcheck.txt (`scripts/pull_device_log.sh`).
2. **Person search on the phone** - time to first answer for "me looking heavier vs me looking fit" with read-ahead
   and the one-walk pair (was ~90 min for one album).
3. **Faster descriptions** - measure batching several photos per model call on the Mac.
4. **App Store** (Reza): App Store Connect record, privacy-policy and support URLs (drafts in docs/APP_STORE.md),
   screenshots, replace `SelfCheck/face.png` (DigiFace-1M, non-commercial; Self-check is hidden in store builds but
   the file is still bundled), a legal view on face matching (BIPA/GDPR). (Open-source credits in About: done 11-23.)
5. Supported phones: iPhone 15 Pro or newer (8 GB memory); set the minimum OS and device rule before submission.

## Known gaps
- The phone's descriptions differ from the cluster's by ~18% of words (runtime, not pixels); parked, small enough.
- Only Reza can build the app today: the bundled Core ML models come from the cluster (`ios/.../Models/README.txt`).
- The repo is public and holds CLAUDE.md (cluster paths, account names) and a face-embedding test fixture whose
  origin should be confirmed synthetic.
