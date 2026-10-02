# find_pics — overnight build plan

## STATUS (update every wake-up)
- 03:19: P0-P2 done. P1 research done (research/00_SYNTHESIS.md). P3 testlib built + indexed. P4 indexer works
  (8xA100, 113 s for 19k items). P5 engine/planner/judge/certificate/albums/review page work end to end via CLI.
  P6 evaluation mostly done (eval/RESULTS.md). Running: transformation demo on Nicolas Cage + video demo.
- Reza's photos: NOT uploaded (data/private empty). When they land: `bash scripts/run_private.sh "<Apple name>"`.
- GitHub repo: NOT created by Reza (git ls-remote -> not found). Committing locally.
- Next: audit transformation + video demos by eye; README numbers; MORNING_REPORT.md; Mac path notes; tests.
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
