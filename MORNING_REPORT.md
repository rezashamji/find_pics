# find_pics: report for the night of 10-03 (newest first)

Code: github.com/rezashamji/find_pics (main). Numbers: [eval/RESULTS.md](eval/RESULTS.md) (sections 18-21 are new).

## Evening 10-03: what you asked for, done
- **Times of day** ("photos between 8 and 11pm", "in the morning", "at night"): hours computed in code from your words,
  using the local clock where each photo was taken (Apple's own spreadsheet stores UTC, so 9pm in Boston would have
  read 1am; that is handled). Benchmark: F1 0.133, the best so far.
- **Unknown people**: the chat shows the face sheet and you answer "Jay is 4"; it redoes the search with Jay.
  **"Me with X"** works: Nic + Patricia on the test library -> 20 photos, all 20 show both (checked by eye).
- **Two moments** ("after A but before B") and moments inside a trip ("before the first Van Gogh photo": 0.06 -> 1.00).
- **"Which face is you"** without Apple's names: `findpics people` -> pick your group once. Same results as Apple tags.
- **A web page** (`findpics web`): text box + photo grids, tap to enlarge, red x to mark wrong. Screens checked.
- **Next on your side:** Apple's email -> download the 1 GB zips with a download-all add-on in the cluster's browser.

## Today 10-03 (while you were out): the chat planner, stress-tested
- **What:** the language model wrote ~800 realistic search requests in 7 voices (busy parent with typos, grandparent,
  student slang, travel photographer, ...). Every one went through the real planner; I read ~1,000 plans by eye across
  rounds. Crashes: 3 found, all fixed (a half-typed message used to end the chat; now it asks you to rephrase).
- **Biggest fixes:** relative dates are now computed by code, not the model (it thought Saturday was Thursday; "last
  weekend" was 8 days; "Fourth of July" a whole year); it no longer guesses who "my sister" is (it picked Sara) or
  invents what she looks like; questions the judge can't answer from pixels ("the house we bought", "taken with a
  telephoto lens", "Is this person Jay?") are rewritten or dropped; city/country names go to GPS instead of the judge;
  "photos from the wedding" returns the whole wedding.
- **Checks:** 30 scripted conversations 29/30 (the 1 miss behaves correctly at search time); end-to-end searches
  unchanged (bread, Kevin Bacon heavy/fit, a person from 3 photos: 318/318 right after checking photos by eye, dog Max).
- **Known gaps:** times of day ("between 8pm and 2am"), personal periods ("fall break"), and "last summer" said in
  October (I keep last year's summer).

## Overnight (you asleep; everything below is measured and photo-checked)
- **Your demo's ranking step works better than reported.** The public "weight transformation" photos were mislabeled
  (8 of Seth Rogen's 8 worst-ranked "heavy" photos were slim co-stars). With the product's face filter, ranking a
  person's photos from heaviest to leanest scores AUC 0.81-0.92 (Pratt 0.921, Hill 0.813, Rogen 0.863). Comparing two
  photos side by side was worse, so the product keeps its method.
- **"This specific thing / place" search:** averaging similarity over your reference photos is the best of everything
  tried (products 0.731, landmarks 0.693; now the default). The judge as re-ranker, a second image model, and keypoint
  matching all made it worse. The largest image model adds only 2-3 points.
- **Phone:** a model a third of the size (PE-Core B) keeps everyday search quality (0.92 vs 0.93, incl. small objects);
  the 4-bit 9B judge tracks the full 9B. Those are the phone candidates.
- **Generalization (72 queries, current planner, streaming):** the first round finds 0.85 of what checking every photo
  finds; the "found at least X%" bound held 68/68. The weak "clothing/color" queries were test-generation errors (3/3).
- **Bugs found by testing and fixed:** dates dropped when the planner reworded "from 2015 to 2018" or left the phrase
  empty, and "2019 and 2020" ending on Jan 1 2020; "only from Paris" sent to the judge instead of GPS; race-script jobs
  cancelling each other (now an atomic lock).
- **Tried and reverted:** judging 3 frames per video. It looked like +19 points, but looking at the frames showed most of
  the "new" videos were judge mistakes (5/8). One frame per video stays.
- **Scale:** indexing your library (147k photos + 40k videos) is about 11 GPU-hours, under an hour on 16 GPUs.
- **The whole phone stack works on a real search:** small image model (PE-Core B) + 4-bit judge found about the same
  photos as today's full-size stack (bread 0.886 vs 0.893 of the full judge's answers; dog 0.978 vs 0.985; sunglasses
  0.662 vs 0.588), and ~95% of what it returned matched the full judge. Phone speed/memory: not measured yet (needs a device).
- **"Find my dog Max" now works end to end** through the real chat: his 3 photos ranked first; "only the ones outdoors"
  dropped exactly his 2 indoor shots. Testing it for real found 4 bugs unit tests had missed (the face detector fires on
  dog faces, so pets were being matched as people) -- all fixed; people-by-photo still finds the same 322 photos + 4 videos.

- **Your library is one command away:** `scripts/run_private_copy.sh` takes Apple's data-copy zips through reading (16
  parallel jobs), indexing (16 GPUs) and a plumbing report only (counts, dates, GPS, faces, image sizes; no searches).
  It passed a full dry run on a fake export (deleted items skipped, dates from Apple's CSV, 0 errors).
- **Phone sizes:** small image model 94 MB + text model 355 MB with 8-bit weights; the 4-bit judge (~5-6 GB) is the
  heavy part. On-device speed and correctness still need a Mac/iPhone.

## Still needed from you
1. privacy.apple.com -> Request a copy of your data -> iCloud Photos -> 10 GB parts.
2. Optional: 3-5 photos of yourself from both eras.

---

# find_pics: report for 2026-10-03 (newest first; the 10-02 report is below)

Code: github.com/rezashamji/find_pics (main). Numbers: [eval/RESULTS.md](eval/RESULTS.md) sections 12-17. Every step:
[JOURNAL.md](JOURNAL.md).

## What changed overnight (10-02 evening to 10-03 morning)
- **One chat box, no modes.** `findpics chat` loads the models once; every message (first or follow-up) goes to one
  planner that sees the conversation. Follow-ups that work end to end, checked by eye: "drop the sandwiches and
  burgers" / "actually keep the sandwiches" (bread 734 -> 530 -> 661); "only the ones where I'm outdoors" on the
  heavier/fit demo (149 -> 52, 33 -> 7). "The day after the wedding"-style requests (offset windows).
- **Every search streams** until the judge has looked at every photo; "found at least X%" holds whenever you stop
  (0 of 115 rounds overclaimed). The judge is 1.9x faster (27.8 photos/s on one GPU).
- **"Find my dog Max" from 3 photos** (pets, objects): best matches first by image similarity, the judge only removes
  clear mismatches. Top-3 precision 72-95% depending on how many look-alike dogs are in the library. Not certified.
- **Phone judge candidate:** the 9B judge with 4-bit weights stays close to the full 9B on 4 concepts across all
  19,218 photos (e.g. christmas tree 6 missed / 6 extra of 85); the 4B and 2B are clearly worse.
- **What "truth" means:** the judge, human labels and a bigger model all make mistakes on disputed photos; checked by
  eye, christmas-tree precision is 15/21 (false alarms on plain conifers), recall 15/15. Rewriting the judge's
  question with definitions or look-alike lists was tested and rejected (it loses real matches).
- **Rejected after measuring:** tile vectors for small objects (gain was the judge's own errors; no gain against human
  labels); "two-step" word rules for the planner (would have broken 35 real DISBench queries).
- **Hard real-user queries (DISBench):** F1 0.112 with the merged planner; the worst cases need "the same building/
  person/scarf across photos", which is the "who = anything" problem.
- **Your export route:** Apple's own data copy (privacy.apple.com, read-only; nothing logs into iCloud). The reader for
  those zips is built and tested (`findpics apple-copy`). Your library stays the final exam: plumbing check only.

## What I need from you
1. When convenient: privacy.apple.com -> Request a copy of your data -> iCloud Photos -> 10 GB parts.
2. Optional: 3-5 photos of yourself (heavier and fit eras) as reference faces.

---

# find_pics: morning report (2026-10-02)

Full history: [JOURNAL.md](JOURNAL.md) (every decision, failure and job id) and `git log`.
Measured numbers: [eval/RESULTS.md](eval/RESULTS.md). Research: [research/00_SYNTHESIS.md](research/00_SYNTHESIS.md).

## 1. What you have this morning
- A working tool, `findpics`, on the cluster. One English sentence goes in. Out come:
  - albums (a folder of links plus a manifest; Apple Photos albums on a Mac with `--apple-apply`);
  - a local review page;
  - a receipt: how many items were scanned, how many the AI judge looked at, and a completeness bound with its caveats.
- Tested end to end on a public test library with known answers: 19,218 photos and videos, including 6 people
  photographed across 32-39 years.
- **Not run on your photos.** No upload arrived in `data/private/` overnight, and the GitHub repo
  `rezashamji/find_pics` doesn't exist yet, so everything is committed locally only.

## 2. The answer to your research question (Feynman order)
1. **Apple doesn't use a VLM for People.** It uses face + upper-body embeddings (vectors where the same person lands
   close together), a deliberately strict clustering pass, and a background run overnight while charging. New photos
   are matched against stored example faces near capture time.
2. **Why it misses your dad:** it is designed to. It only merges faces it is very sure of and holds the rest under
   "Review More Photos", and it never tells you how many.
3. **Why "bread" fails:** classic search stores a few words per photo from a fixed list (~1,300 tags), each with a
   score cutoff tuned for precision. We measured that kind of cutoff on 264 verified bread photos: it keeps only 28-31%
   of them. Ranking every photo by a stored vector and judging the top recovers 98% within the top 5×.
4. **What already exists:** natural-language photo search itself (Apple on recent iPhones, Google Ask Photos, the open-source
   Immich). **What doesn't:** any tool that tells you how complete its answer is. That is the product's wedge.

## 3. What was measured (denominators in eval/RESULTS.md)
- **Concepts:**
  - dog: 1,560 of 1,586 found; stated "at least 97.9%", truth 98.4%.
  - bread: 214 of 264; stated "at least 70.5%", truth 81.1%.
  - The stated lower bound was at or below the truth in 6 of 6 concept results (3 runs x bread, dog) and 290 of 300 simulations.
- **People** (face vectors only):
  - Kevin Bacon: 328 of 336, with 1 wrong item.
  - Drew Barrymore: 469 of 490, 0 wrong, plus 19 "possible" items for review, of which 11 are really her.
  - Nicolas Cage 1995-2005: 163 of 168. The 11 "wrong" items were all actually Cage (I looked); the labels were wrong.
- **"Looks overweight"** (CelebA): judge AUC 0.90. The external labels are themselves noisy and look racially skewed,
  and the judge confuses age with weight.
- **Matching across eras:** references from only the most recent photos, then confident matches chained in as new
  references. Oldest-third recall rose: Cage 0.88 → 0.95, Aykroyd 0.80 → 0.90, Drew 0.92 → 0.98. Wrong matches were
  unchanged. This is now the default.
- **Within one person** (the transformation question; 132 CelebA people): mean AUC 0.64-0.66. Where a real visible
  change exists, the judge separates the eras, but this dataset has few such people. Your own photos are the real test.

### Added after you woke up (you asked me to confirm everything)
- **Videos of a person:** on an export in Apple's exact format (HEIC, iCloud previews, HEVC `.mov`), "every photo and
  video of Kevin Bacon" found all 4 videos containing him (none tagged; found from faces in sampled frames) and none of
  the 6 others. Photos: 317/324; the 1 "wrong" photo looks like him in a group shot.
- **People no model has seen:** synthetic rendered identities (DigiFace-1M), 8 references each: 98.7% of each
  person's photos found (18,855/19,110). Wrong matches were higher than on real photos (44 per person among 21k).
- **Android:** Google renamed Takeout's metadata files in late 2024 (`.supplemental-metadata.json`, sometimes clipped).
  The old code would have silently lost every date. Fixed and tested on all 3 naming schemes.
- **Scale:** 150,000 items: everything except the AI judge takes under 2 s per query (1.8 GB RAM). The judge is the
  cost: ~1,600 calls, ~35 s on one GPU.
- **Learning new things from you:** 3 example photos were *worse* than a text description for nameable things (recall
  0.69 vs 0.88). Learning from your clicks helped only where text was weak (Christmas tree 0.25 → 0.50). Your clicks
  are how it would learn *your* taste ("bad photos of me"), but that is not proven.

- **People with documented weight loss** (Chris Pratt -60 lb, Jonah Hill -40 lb, Seth Rogen -30 lb; I labeled all 166
  heavy-era faces by eye because IMDB's labels were ~30% wrong):
  - **Face matching across the change:** with references from the lean era only, 112/114 heavy-era photos found;
    114/114 with cross-era expansion. Big weight change did not break identity.
  - **The judge** ranks heavy-era above lean-era photos: AUC 0.81-0.87 (0.5 = chance).
  - **The album logic I had shipped was wrong for this:** it put 1 of Pratt's 52 heavy-era photos in "heavier". The
    judge's raw scores differ by person, so albums now rank each photo against *that person's own* photos. Measured:
    "heavier" gets 78/114 heavy-era photos with 29/131 lean-era leaking in; "fit" gets 93/131 lean-era with 19/114 leaking.
  - **Consequence:** a relative ranking always has a "heavier half", even for someone who never changed weight. For
    your 10-year change that's the point. It's a ranking of your own photos, not a judgment against other people.

## 4. Things I got wrong overnight and fixed (because I looked at raw outputs, not just scores)
- **Reference contamination.** Co-stars' faces became "Drew Barrymore" references. Fixed with a consensus filter:
  14 wrong matches went to 0.
- **The VLM is not a face recognizer.** It said "same person" 191 times for Kevin Bacon and was right 4 times.
  Identity is now from face vectors only.
- **Planner date leak.** It put your "past 6 months" on the *heavier* album too, which would have silently dropped every
  heavy-era photo. Dates are now code-checked against your exact words.
- **Invented appearance filters.** The planner added "blonde hair, blue eyes" to "every photo of Drew". Code now strips
  them.
- **My first completeness estimator was a heuristic.** Replaced with the e-discovery elusion test (exact binomial bound).
- **Group photos distracted the judge.** Appearance is now judged on a head+torso crop of the matched person.
  "Cage heavier" went from 56 photos (many driven by other people in the frame) to 27.
- **One photo was in both "heavier" and "fit".** Each photo now goes only to the album whose question the judge
  answered most confidently.
- **Every face match showed similarity 1.00.** Cross-era expansion let accepted faces match themselves. Membership was
  right, but the number was meaningless. Self-matches are now ignored.
- **"Best" albums returned nearly everything** (145 of 153). They are now curated: confident yes only, top quarter.
- **The judge's yes-cutoff was too loose.** On "dog on a beach", the 0.5-0.7 band was mostly wrong (dog portraits, a
  birthday cake). It's now 0.7; on labeled data that cost 6 of 1,586 dogs and 9 of 264 bread photos.
- **The README install was broken for strangers.** Unpinned, the resolver installed a 2023 vLLM that can't build, and
  the Mac extra resolved for macOS 13. Fixed and verified: a clean install from the README passes all 18 tests and
  runs a GPU query.

## 5. To run it on your library (about 30 minutes of your time)
1. On your Mac: [docs/MAC_EXPORT.md](docs/MAC_EXPORT.md) (read-only export and upload).
2. On the cluster: `bash scripts/run_private.sh "<your exact name in Apple People>"`.
   This indexes on the GPUs, then runs the transformation request. Output goes to `data/private/albums_*/index.html`.
   Tested tonight end to end on public data with Kevin Bacon standing in for you: scan, then 5 GPU shards (5-7 min),
   then the query (3-7 min). For 150k photos, expect about 16 shards and 15-30 min of indexing, if GPUs are free.
   To see how your sentence was understood before anything runs, first try
   `findpics ask data/private/index "<sentence>" --out data/private/plan_check --plan-only` (inside a GPU job).
3. Open the review page, click wrong photos red and right ones green, then export reviews.json and run
   `findpics apply-reviews <albums_dir> reviews.json`. It removes the album links you marked wrong (never the photos;
   the remover refuses anything that isn't an album link) and records "you checked N: X correct, Y wrong" in each
   manifest. Not done yet: folding that human check into the completeness statement itself.
4. On your Mac, to write the albums into Photos: [docs/MAC_RUN.md](docs/MAC_RUN.md) (`--apple-apply`; additive only).

## 6. Needs your decision
- **Two files outside find_pics/** were written tonight by library side effects. My cleanup was blocked by the
  permission system, so they're yours to decide:
  1. tonight's appended lines in `~/.config/vllm/usage_stats.json`;
  2. a Rust toolchain at `~/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu`. It was installed by a package build;
     your default toolchain is untouched.
  - Prevention is in place: caches are pinned inside the folder, and Slurm jobs run with HOME inside find_pics.
- **GitHub:** create `rezashamji/find_pics` (private), then I push `main` only. Make it public after you review.
  Found at 07:50 in a final check: contact sheets of public-dataset images (celebrity photos, CelebA faces) had been
  committed. CelebA's license forbids redistribution, so I removed them from `main`'s entire history (nothing had been
  pushed). The old history sits in a local branch, `pre-scrub-backup`; delete it once you've confirmed with
  `git branch -D pre-scrub-backup && rm -rf .git/refs/original`.
- **License:** the code is MIT. The InsightFace face models are non-commercial: fine for personal use, but a company
  needs the swaps in research/03.

## 7. Honest limits
- Never run on real iPhone data. The Mac path (MLX judge, Photos album writer) is written but untested.
- Person completeness can't be certified by the AI judge. It's reported as the measured test-library rate plus a
  review sample.
- "Looks heavier" is the judge's opinion on a relative scale, not a measurement.
- GPUs: H100s/H200s weren't available at our priority tonight (estimated starts were 07:00 and 15:00), so everything ran on A100s.
