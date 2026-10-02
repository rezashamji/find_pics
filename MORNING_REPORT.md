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
