# find_pics — overnight build plan

## STATUS (update every wake-up) — written 2026-10-02 ~15:00 before compaction; amended ~15:45
Reza said: "do it all, continue post compact". Work autonomously; report results after raw-look audits.

### OVERNIGHT PLAN 10-03 (Reza asleep; full autonomy; big/long GPU jobs first; continue after compaction without prompting)
Goal: Reza's demo (me heavier vs me fit, photos+videos) + "find anything" on a phone; backend right first (Phase A).
Loop each wake: squeue -> read finished results -> raw look (background subagent for image sheets) -> journal ->
commit+push -> submit the next queued job so GPUs never idle -> ScheduleWakeup (watchers wake earlier).
Queue, in order (mark [x] when done, write results to JOURNAL + RESULTS):
[x] 1. Transformation PAIRWISE judge (done 10:10: no clear win; product unchanged) (eval/eval_transformation_pairwise.py): same person, two photos side by side,
       "which looks heavier?" (both orders); within-person AUC heavy vs lean vs single-photo AUC 0.81-0.87.
       If clearly better: product person-appearance albums rank by pairwise wins (engine), test, regress.
[x] 2. Same-instance across photos (#4) (done 11:45: vectors win; judge, DINOv2, SIFT all worse; product uses mean over refs): eval_instance things/places with OWLv2 crops + PE-Core/DINOv2 crop vectors +
       side-by-side veto; then a planner/engine "same as the anchor's X" operation if it helps; DISBench cross-photo queries.
[x] 3. Generalization re-run (done 12:40: round-1 0.85, bound 68/68; attribute queries weak) with the CURRENT planner (converse) + streaming on the 72 generated queries (round-1 recall
       vs oracle), raw look at misses.
[ ] 4. Regression after any engine change: scripts/regress.sh (bread chat + bacon demo) + DISBench shards (race_sbatch).
[ ] 5. If time: 3-hop chains (anchor of an anchor); review-page polish; README/RESULTS/MORNING_REPORT refresh.
Never: touch Reza's photos, lower judge resolution, run >30 s in foreground, push anything but main.

### ~09:20 10-03 STATUS (latest; Reza back)
Since 06:00: planner 14/14 as executed (place names in filters -> place filter); DISBench regression found + fixed:
no-condition albums crashed, unanswerable "only..." filters not dropped, copied exclusions; race_sbatch mutual-cancel
fixed with an atomic lock (scripts/race_guard.sh); DISBench v5 = parity with best (F1 0.111, recall 0.362, 0 errors).
27B cache blobs removed (102 -> 50 GB). Tiles stay off (no gain vs human labels). 62 tests.
NEXT (biggest open backend problem, #4 "who/what = anything" across photos): a "same instance" operation -- given the
anchor photos of a thing (building, scarf, car, person without tags), find other photos containing the SAME instance via
detector crops + instance vectors + side-by-side judge veto; measure on things/places sets and on DISBench's
cross-photo queries. Also open: 3-hop chains.
Waiting on Reza: privacy.apple.com data-copy request.

### ~06:00 10-03 STATUS
Done since 01:45 (all pushed): judge vs EYE-checked truth (christmas tree precision 15/21, recall 15/15; labels and
model-adjudication both unreliable on disputes); LLM-defined and look-alike questions REJECTED (recall loss); 4-bit and
FP8 9B judges ~ bf16 9B (phone candidate: 4-bit); "find my dog" from 3 refs: rank by image vector (top-3 precision
72-95%), judge = veto at 0.2, album best-first, not certified; Apple data-copy ingester (`apple-copy`); offset windows;
filter_question ("only the ones where ...", found by end-to-end regression, fixed, raw-looked); RESULTS 12-17.
Running: fp_planners (14 conversations incl. "only outdoors").
Next: (1) tiles vs eye-checked truth (small objects) before switching on; (2) 27B blobs cleanup via hf cache tooling;
(3) multi-hop chains beyond offsets (deferred: rare in real use); (4) Phase B only after Reza's go-ahead.
Waiting on Reza: privacy.apple.com request (whenever), then the data-copy download route.

### ~01:45 10-03 STATUS
Done tonight: export route = Apple data copy (Reza will request later; ingester `findpics apple-copy` built+tested);
GitHub pushed; streaming tile replay (gain partly judge errors -> tiles on HOLD); offset windows; planner degrades
unanswerable questions; RESULTS.md 12-16. Rule: nothing >30 s in the foreground (memory + CLAUDE.md).
Running (watchers bi4o9bzg0, next one): fp_qdefs (planner-defined questions vs human labels), fp_adjudicate_27b
(27B third opinion), fp_full_9b_int4 / fp_full_9b_fp8 (phone-judge precision), fp_petjudge2 (strict same-dog question).
When they land: `PYTHONPATH=src python eval/analyze_truth.py` -> judge vs adjudicated truth; raw-look samples of each
outcome (background subagent); then re-measure tiles vs adjudicated truth; pet: pick cut from per-pair scores.

### ~00:45 10-03 STATUS
Repo pushed: github.com/rezashamji/find_pics (main only). Full plan given to Reza (phases A-D): A = backend right with big
models (judge vs truth, small objects, hard queries, any-identity), B = phone feasibility, C = app, D = licensing.
Reza's library = final exam (plumbing check only until confident). Reza is starting the Apple export.
Running: fp_dis_uni2_s0..7 (DISBench, fixed planner, saves got_ids -> raw look returned photos), fp_full_9b_int4
(4-bit 9B, 4 concepts x 19,218), fp_full_9b_fp8 (H100/H200 only), fp_adjudicate_27b (27B third opinion on judge-vs-label
disagreements).
Next (A, in order): (1) adjudicated truth -> judge precision/recall vs truth; error types; fixes (sharper questions,
second check on yes-photos) measured; (2) tiles on in the product index + re-measure first-round recall on small
objects; (3) chains of >2 hops + metadata-based relations; DISBench raw look; (4) animal/object identity (detector crops
+ MegaDescriptor) + check whether eval_pet_judge ever ran.

### ~20:20 STATUS (Reza turned the session off here)
Running/queued on their own (no session needed; check `squeue -u rshamji`, logs in slurm/logs/):
- Race jobs (scripts/race_sbatch.sh, 6 copies each, first to start cancels the rest, lower job id wins):
  fp_dis_uni_s0..7 (DISBench, merged planner, 8 shards; LAST shard writes eval/disbench/unified_summary.txt),
  fp_gen_fix_2_9 / 33_45 / 62_69 (6 generalization queries rerun without the red-box/person artifact; last one writes
  eval/general/analyze_latest.txt), fp_planners (12+1 scripted conversations, 72 gen, 122 DISBench plans ->
  eval/planners/*.json; check stdout in the newest slurm/logs/fp_planners_*.out).
  Queue position at 20:15: ~#340 of 424 in kempner_requeue (lab fairshare low); other partitions at the lab's 96-GPU cap.
- gpu_test: fp_full_2b_then_9bfp8 (2B, then FP8 9B) + fp_full_Qwen3_5-4B (eval/eval_small_oracle.py: small judges on ALL 19,218 photos for bread,
  christmas_tree, sunglasses, dog -> eval/judge_size/full_*.parquet) = false-yes rate at real prevalence.
Next session, in order:
1. DISBench: read unified_summary.txt; compare with agent (R 0.424, F1 0.096) and baseline (F1 0.033); RAW LOOK at a
   few queries (gt photos vs returned, full res) before any claim.
2. Planner test: 13 conversations (incl. partial undo); apply place_or_look when scoring (see journal 17:35).
3. Generalization: analyze_latest.txt; event/occasion dimension after the fix.
4. Small judges: compare full_* with eval/oracle/<qid>.parquet (9B); raw look at disagreements; pick a phone cut.
   Remember: the 9B "oracle" over-calls conifers as christmas trees (oracle != truth).
5. FP8 9B (Reza OK 20:30) queued on gpu_test: compare full_Qwen3.5-9B-fp8_* vs eval/oracle (bf16 9B) per item.
   Waiting on Reza:
   GitHub repo creation; his Apple export; PhotoBench photos (his call); two outside files.

### ~19:10 STATUS
- Product path = `findpics chat` (models once) / `ask`: converse.py planner + stream_plan (rounds until every photo is
  judged; union-bound alpha; growing tail samples). Conversation test passed 3/3 turns with raw look. 50 tests.
- Judge loader 1.9x faster (measured). Use as many GPUs as possible; shard evals; gpu_test overflow (CLAUDE.md).
- Waiting (GPU cap full): fp_dis_uni array 49973750 (8 shards; merge watcher b9tdwoowx), fp_planners 49973865 on
  gpu_test (watcher b2z6w3zn4). Compare uni vs agent (R 0.424, F1 0.096) vs baseline (F1 0.033); raw look per query.
- Next CPU ideas: raw look of the "event or occasion" generalization queries (oracle median 0); RESULTS.md update;
  phone: smaller encoder (Core ML towers are 1.3 GB).

### ~15:45 change of direction (Reza): ONE conversational path, no mode flags, no separate refine system
- Done: src/findpics/converse.py (planner sees conversation + current plan -> whole new plan; optional anchor/window/
  exclude per album; effort_phrase = "look harder"; judge cache; per-item overrides). refine.py, --multistep,
  --exhaustive removed. 39 tests pass. Commit fdb1092.
- Waiting: fp_planners 49950289 (does the merged planner match/beat old ones? 12 conversations, 72 gen, 122 DISBench),
  fp_disbench_uni 49950290 (end-to-end), fp_chat 49950443 (3-turn conversation on testlib). Watchers b8ec80qny, b6kz7z9nb.
  If the merged planner regresses on single-step queries: fix the prompt, do NOT reintroduce flags.
- Then: delete agent.make_plan/execute once DISBench baseline/agent jobs have run; raw look tiles gains + 2B judge misses.

### Jobs in flight (check: squeue -u rshamji; job ids also in .cache/tmp/*.jobid)
- Generalization oracle: array 49932711 (72 queries generated from library content -> eval/general/oracle_*.parquet).
  When done: `PYTHONPATH=src python eval/eval_general.py analyze` (CPU) -> per-dimension table; then RAW LOOK at
  full res (>=900 px) of fast-missed / oracle-yes samples per dimension; degenerate queries (match most photos: "no
  text", "clear and in color") reported separately.
- Tile vectors: 49938023 (eval/eval_tiles.py embed) with watcher that runs analyze -> does max-over-2x2/3x3 tiles
  raise recall of oracle-confirmed small objects within top-K? If yes -> add tile vectors to index.py + engine
  (look_scores = max over tiles) and re-measure fast-vs-oracle.
- Oracle concepts 8-11 (49929692) + 12-19 (49938104) -> then `python eval/eval_oracle.py analyze` for all 20
  (15 concepts + dog_on_beach, person_sunglasses, birthday_candles, eating_pizza, bike_street).
- DISBench index: 49941173 (16 shards) -> baseline eval job 49941813 (eval/eval_disbench.py, starts afterok) -> per query,
  scope = that user's photos (path contains user id), run planner+engine fast mode, score vs answer ids
  (recall/precision/F1); this is the BASELINE for hard multi-step queries.

- Multi-step planner v1 written (src/findpics/agent.py: anchor -> window -> target + exclusion; 28 tests pass); DISBench agent eval job 49942224 (afterok index) -> compare eval/disbench/agent.json vs baseline.json, raw-look per query.

### Next build steps (agreed with Reza, in order)
1. Multi-step agent planner: plan -> search -> read results/metadata -> derive windows (date/place/event) -> search
   again. Same upgrade gives conversational REFINEMENT ("remove these / only smiling / add more like this / also
   videos"): session state (albums + judged tables + prior plan), planner emits EDIT ops; removals only drop album
   links; each edit reports counts; receipt updated. Measure on DISBench (vs baseline) + refinement unit tests.
2. Tile vectors in product if the experiment supports it.
3. Raw-look sheets (full res) for things/places/copies instance results; rerun copies to get per-copy-type breakdown.
4. Individual animals still unsolved (~0.60 R-precision; crops and MegaDescriptor-B-224 did not help). Ideas: larger
   MegaDescriptor only if a clean test set exists; judge side-by-side for animals (measure); user clicks.
5. Phone/Mac path (not started): MLX judge + MPS encoders untested; needs a Mac.

### Done since the overnight build (see JOURNAL for numbers)
person identity on unseen + weight-change (114/114); videos 4/4; --ref (3 photos = 168 tags); within-person relative
ranking for appearance albums; --exhaustive; place/GPS support; Takeout 2024 names; oracle-vs-fast for 8 concepts
(fast recovers 79-98%; gap = real small background objects, verified at full res); instance identity table (copies
0.90, things 0.69, places 0.68, dogs ~0.60); research 05 query space; PhotoBench GT (photos need a request: ASK REZA).

### Waiting on Reza
photo export upload; GitHub repo creation; PhotoBench photo request (his call); 2 outside-folder files; more lab GPUs.
---

## 1. The goal in one paragraph
A tool anyone can run on their own photo library that answers a natural-language request like
"find every photo and video of Reza where he looks heavy, and the best athletic ones from the last
6 months, and make two albums", using only open-source models (no per-photo API cost), never
modifying or deleting the user's photos, and telling the user honestly how many items it scanned and
how complete the result is likely to be, as an estimate with a confidence interval rather than an
unqualified "found them all".

### Morning acceptance criteria (what "done" means at ~08:00)
1. `research/00_SYNTHESIS.md`: what Apple/Google do, what already exists, what does not, method choice. Every claim cited.
2. Working `findpics` package: `index` (folder or Apple Photos export) and `ask "<natural language>"`, with album output
   (folder of symlinks + JSON manifest; Apple Photos album writer, dry-run by default).
3. Evaluated on a public test library with known ground truth: person recall/precision, concept recall ("bread"),
   and whether the recall estimator's confidence interval actually contains the true recall (calibration).
   Every metric has a denominator, and contact sheets were viewed by Claude.
4. If Reza's export arrived: the transformation query run on his library, two albums produced, every image in both
   albums viewed by Claude, a sample of rejects viewed, comparison against Apple's own "Reza" tagging.
5. README a stranger can follow on a Mac. Code on GitHub (private until Reza reviews).
6. `MORNING_REPORT.md`: what works, what doesn't, numbers with denominators, raw examples, next steps.

---

## 2. Working hypothesis of the method (to be confirmed or overturned by P1 research)

Core idea: **index once, query many times.** This is Reza's "compute metadata when the photo is taken"
idea, with one change: the stored metadata is a vector (a list of ~1,000 numbers that encodes what the
image looks like), not a list of words. Words are lossy: if the indexer never wrote "bread", a later
search for "bread" cannot find it. A vector can be compared against any query text later, including words
nobody anticipated.

Per item (photo, or sampled video frame), computed once on GPU:
- **Faces**: detector finds face boxes, then a face-recognition network turns each face into an identity
  vector (same person -> vectors point the same way, cosine similarity high).
- **Whole-image vector**: a CLIP/SigLIP-type model (trained so an image and a caption describing it map to
  nearby vectors) gives one vector per image; text queries are embedded by the same model's text tower.
- **Metadata**: capture date, GPS, media type, duration, Apple's own person/label tags if exported.

Per query:
1. **Planner**: a small local LLM fills a fixed JSON template from the request (person, attributes, date range,
   media type, album names). Fixed template + injected request = the "stagnant prompt" Reza described.
2. **Candidate generation (cheap, all N items)**: date/media filters (exact); person match = max cosine of the
   item's faces vs the person's reference vectors (from a few labeled photos, ideally spanning eras);
   attribute ranking = cosine of image vector vs text vectors. Cost: one matrix-vector product over N.
   For N=150k and 1,152-dim vectors that's ~1.7e8 multiply-adds: milliseconds. No O(1) structure needed.
3. **Verification (expensive, only candidates)**: an open VLM (vision-language model: looks at an image and answers
   text questions) answers a fixed rubric per candidate ("Is the person in the red box the same as the reference?",
   "Rate 1-5 how heavy this person looks"). Only hundreds to low thousands of items, not 150k.
4. **Recall audit**: draw a stratified random sample from the items NOT returned (strata by model score, so the
   sample concentrates where misses are likely), have the VLM judge them, and spot-check the judge with
   human/Claude labels. Estimate missed matches -> recall with a confidence interval. This is the
   "elusion test" from legal e-discovery (method to be confirmed in research/04).
5. **Album assembly**: write results as albums; report items scanned, candidates verified, sample sizes, recall CI.

Cost check for "VLM on every photo": 150k items at ~20 img/s on one H100 is ~2 h per query; on a laptop at ~1 img/s
it's ~40 h. So a VLM over everything is not viable per query on consumer hardware, while embeddings are computed
once (~minutes on GPU, ~1-3 h on a Mac) and reused for every future query.

Honesty caveat to carry into the product: identity recall ("is Reza in this photo") has an objective ground truth.
"Looks heavy / looks bad" does not; there, "recall" means agreement with a judge (VLM, checked against Reza/Claude
labels on a sample) and the report must say so.

---

## 3. Phases

Times are targets (ET, 2026-10-02). If a phase slips, cut scope inside it; keep P6, P8, P9 alive.

### P0 Setup (00:45-01:15) — DONE when committed
- Folder skeleton, `env.sh` (all caches inside), `.gitignore`, `CLAUDE.md`, this plan, `JOURNAL.md`, git init.
- Ask Reza for: Mac export + upload, GitHub repo creation, privacy OK. (See section 7.)

### P1 Deep research (01:00-02:30, background agents)
- `research/01_apple_google_internals.md`: how Apple/Google people recognition + search work, when it runs, why it misses, why "bread" fails.
- `research/02_competition_landscape.md`: Immich, Ente, PhotoPrism, Queryable, Google Ask Photos, research prototypes. Verdict: what does not exist.
- `research/03_models_sota.md`: face det/rec, text-image embeddings, VLMs, video, licenses, Mac support.
- `research/04_recall_estimation_and_platform_access.md`: elusion tests, PPI, sample sizes; osxphotos, PhotoKit, icloudpd, Google API.
- Then Claude writes `research/00_SYNTHESIS.md` (Feynman order: what exists -> what's missing -> why our method).
- Done when: synthesis written, stack chosen, decision logged in JOURNAL. If research says the product already exists
  in full, say so in the synthesis and narrow to the missing wedge rather than rebuilding.

### P2 Environment (01:15-02:30)
- `uv` venv at `envs/fp` (Python 3.11 or 3.12 via uv-managed python inside `.cache/uv-python`).
- torch (CUDA 12.x wheels), transformers, open_clip_torch, insightface + onnxruntime-gpu, pillow, pillow-heif,
  opencv-python-headless, av (PyAV) and/or decord, pandas, pyarrow, numpy, scikit-learn, hdbscan, fastapi/uvicorn, rich.
- Separate `envs/vllm` if vLLM pins conflict with the main env.
- Static ffmpeg binary into `tools/bin`.
- Smoke test on a GPU node via `srun` (imports, CUDA visible, one model forward pass).
- Check whether compute nodes have outbound internet (model downloads). If not, download on login node.
- Download chosen weights into `.cache/huggingface` / `models/`.

### P3 Public test library with ground truth (01:30-03:00) — Reza's suggestion, and the right call
Build `data/public/testlib/` that mimics a personal library (~20-50k items), where we KNOW the answers:
- Identity across years/appearance: a cross-age celebrity set (e.g., CACD, 2,000 people over ~10 years) or another
  identity-labeled set available on HF. Pick 3-5 "family members" with many photos each.
- Heavier/leaner appearance: CelebA has a "Chubby" attribute label -> ground truth for a heavy-looking judgment.
- Concepts like "bread": Open Images has a "Bread" class with image-level labels; COCO has sandwich/pizza/donut/cake.
- Distractors: thousands of unrelated photos (COCO / Open Images).
- Videos: a small identity-labeled or concept-labeled public video set (candidates: VoxCeleb-style talking heads,
  CC0 stock clips). Videos must go through the same pipeline (frame sampling).
- Write `testlib/ground_truth.json` (query -> set of true item ids). Licenses noted in `data/public/LICENSES.md`.

### P4 Indexer (02:00-03:30)
- `src/findpics/ingest.py`: walk a folder or osxphotos export; read EXIF/sidecar dates; decode JPEG/HEIC/PNG; videos ->
  frames (e.g., 1 frame / 2 s plus scene changes, capped per video).
- `src/findpics/index.py`: faces (det + embedding + box + quality), image embedding, metadata -> Parquet + .npy shards.
- Slurm array job over shards; resumable (skip shards already written). Requeue-safe.
- Watch: throughput (items/s), decode failures (count and sample them), GPU memory, empty-face rate.
- Done when: test library fully indexed, failure count logged with examples viewed.

### P5 Query engine (03:00-05:00)
- `planner.py`: NL -> JSON plan via small local LLM with a fixed template; schema-validated; fallback rules.
- `people.py`: person = set of reference face vectors (from user-labeled examples or Apple's tags); cluster library faces
  so one confirmation labels many; match threshold chosen from evaluation, not guessed.
- `search.py`: filters + embedding ranking -> candidates.
- `verify.py`: VLM rubric scoring of candidates (vLLM batch on GPU; mlx-vlm on Mac later).
- `audit.py`: recall estimation with stratified sampling + CI; report scanned counts.
- `albums.py`: folder-of-symlinks + manifest; Apple Photos writer (create album + add only; dry-run default).
- `cli.py`: `findpics index`, `findpics ask`, `findpics review` (local web page to accept/reject; accepted/rejected labels feed the audit).

### P6 Evaluation on public library (04:00-06:00)
- Person queries: recall and precision vs threshold, by era (does a 2004 face match a 2013 reference?).
- Concept queries ("bread"): embedding search vs a fixed-label baseline (simulate Apple-style taxonomy+threshold) -> evidence for why bread fails.
- Composite query: person + "heavy-looking" + date range.
- Recall estimator calibration: across many queries and repeated samples, how often does the 95% CI contain the true recall? Target ~95%. If not, fix the estimator.
- Claude views contact sheets of: top results, borderline items, a random sample of misses, the audit sample. Writes what it saw in `eval/RESULTS.md`.

### P7 Reza's library (whenever the upload lands; poll `data/private/apple_export/`)
- Index on GPU (all photos; videos if uploaded).
- Reference set for "Reza": Apple's own "Reza" tags from the osxphotos metadata/sidecars (+ clustering to catch era changes).
- Compare: items Apple tagged as Reza vs items we find; Claude views the disagreements (both directions).
- Run the transformation query -> two albums. Claude views EVERY image in both albums and a random sample of rejects,
  writes per-image notes in `data/private/audits/` (never committed). Iterate thresholds/rubric if wrong.

### P8 Product packaging (05:00-07:30)
- `pyproject.toml`, install instructions for Mac (Apple Silicon: PyTorch MPS / MLX) and Linux GPU.
- Apple Photos path: osxphotos read; album write via osxphotos/photoscript (create + add only), dry-run unless `--apply`.
- Android path: Google Takeout / any folder (Google Photos API no longer gives full-library access; to confirm in research/04).
- Honest README: what it does, what it doesn't, privacy (all local), license constraints of model weights.
- Unit tests (`tests/`): planner schema, date filters, recall math against simulated known answers, no-delete guarantee
  (grep + runtime guard), album writer dry-run.
- Push to GitHub (private) once repo exists.

### P9 Morning report (07:30-08:30)
- `MORNING_REPORT.md`: results with denominators, raw examples (public ones inline; private ones as local paths),
  failures, what Reza must do next, and a demo command.

---

## 4. Slurm playbook

Why Slurm: this session runs on `holylogin07`, a login node with no GPU; heavy CPU/GPU work on login nodes gets killed
and is against cluster policy. Slurm is the scheduler that hands out GPU nodes: you describe the job (GPUs, CPUs,
memory, time) in an `sbatch` script, it queues, runs on a compute node, and writes logs.

- Accounts: `kempner_mzitnik_lab` (Kempner partitions), `mzitnik_lab` (FASRC partitions).
- Partitions: `kempner_h100` (4x H100 80GB/node), `kempner_h200` (4x H200/node), `kempner_requeue` (preemptible, often faster to start),
  `gpu_h200` (account `mzitnik_lab`). Cap 16 GPUs concurrently.
- Template: `slurm/templates/gpu_job.sbatch` (to write in P2): `#SBATCH -p kempner_h100 -A kempner_mzitnik_lab --gres=gpu:1 -c 16 --mem=128G -t 04:00:00 -o slurm/logs/%x_%j.out`,
  body does `source env.sh` then runs a python module.
- Embarrassingly parallel indexing: `--array=0-K%16` (K shards, max 16 at once). Each task writes its own shard;
  skips if shard exists -> resubmitting is idempotent; safe on the requeue partition.
- Interactive smoke tests: `salloc -p kempner_h100 -A kempner_mzitnik_lab --gres=gpu:1 -c 8 --mem=64G -t 1:00:00`
  then `srun --jobid <id> --overlap nvidia-smi`.
- Watch: `squeue -u rshamji` (state, reason: Priority/Resources = waiting), `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode`,
  `tail slurm/logs/*.out`. Things to watch for: OOM (MaxRSS near --mem, CUDA OOM in log), throughput (items/s printed
  every N batches), decode error counts, NaNs in embeddings, pending > 20 min (switch partition/requeue).
- Every submission gets a JOURNAL line: job id, purpose, partition, expected runtime.

## 5. Autonomy protocol (overnight)
- Dynamic `/loop` with ScheduleWakeup. Each wake: resume protocol in `CLAUDE.md`, advance the next task, journal, commit.
- When a job is running: background agents/jobs notify on completion; fallback wake-up 20-30 min.
- Never sit idle while waiting on a job: work on the next code task in parallel.
- After compaction: `CLAUDE.md` auto-loads; follow its resume protocol.

## 6. Journal + git conventions
- `JOURNAL.md`: append-only, `## HH:MM` headers, what was done / decided / why / job ids / failures. Raw observations from viewing images go here (public data) or `data/private/audits/` (private).
- Commit after each milestone with a message saying what changed and why. Push when remote exists.
- `git log` + `JOURNAL.md` = Reza's full history of the night.

## 7. Asks to Reza (before bed)
1. Mac: install osxphotos, run the read-only export + upload commands (in the chat message / `docs/MAC_EXPORT.md`).
2. GitHub: create an empty private repo `rezashamji/find_pics` (no README). Claude pushes via SSH (already works).
3. Confirm OK to store a downscaled copy of his library in `data/private/` (chmod 700) and that Claude will view
   thumbnails of his photos during audits.
4. Optional: approximate dates of heaviest period and when he got lean (used only as a sanity check, never as the classifier).

## 8. Risks
- Upload too slow -> Reza's library arrives partially; work on what arrives, report coverage honestly.
- Face recognition fails across large weight change -> use references from both eras + clustering + body/context cues; measure on test set.
- Model license (InsightFace weights are reportedly non-commercial) -> fine for a personal tool; document; offer a permissive alternative.
- VLM judge bias on body judgments -> report judge agreement with Claude/Reza labels; never present a subjective judgment as fact.
- Mac path can't be executed on this Linux cluster -> unit-test it with mocks, mark "untested on real Mac" in the report.
